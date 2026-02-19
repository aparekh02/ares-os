"""AdapterTrainer — hybrid online + offline training for the SteeringAdapter.

Online:  Every N cycles, sample recent traces with outcomes, compute
         reward-weighted loss on guidance category prediction.
Offline: Contrastive (good, bad) pairs + attention entropy regularization.
L_total = L_reward + lambda1 * L_contrastive + lambda2 * L_attention

All sampling and checkpointing calls are async (MongoDB-backed).
Gradient computation stays synchronous (PyTorch).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from mem_alpha.steering.adapter import SteeringAdapter
from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.schemas import MemorySlot

logger = logging.getLogger(__name__)


class AdapterTrainer:
    """Trains the SteeringAdapter using online and offline strategies."""

    def __init__(
        self,
        adapter: SteeringAdapter,
        trace_buffer,  # MGSTraceBuffer
        config: Optional[MGSConfig] = None,
        gridfs_bucket=None,  # AsyncIOMotorGridFSBucket
    ):
        self._adapter = adapter
        self._traces = trace_buffer
        self._config = config or MGSConfig()
        self._gridfs = gridfs_bucket
        self._online_optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=self._config.online_lr
        )
        self._offline_optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=self._config.offline_lr
        )
        self._update_count = 0

    async def online_update(self) -> float:
        """Run a few gradient steps on recent traces with outcomes.

        Returns the average loss across steps, or 0.0 if no data.
        """
        traces = await self._traces.sample_traces(
            self._config.online_batch_size,
            only_with_outcome=True,
        )
        if len(traces) < 2:
            return 0.0

        self._adapter.train()
        total_loss = 0.0
        steps = 0

        for trace in traces:
            loss = self._reward_weighted_loss(trace)
            if loss is None:
                continue
            self._online_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._adapter.parameters(), 1.0)
            self._online_optimizer.step()
            total_loss += loss.item()
            steps += 1

        self._adapter.eval()
        self._update_count += 1
        await self._maybe_checkpoint()

        avg = total_loss / max(steps, 1)
        logger.info("Online update #%d  steps=%d  avg_loss=%.4f", self._update_count, steps, avg)
        return avg

    async def offline_train(self) -> float:
        """Run offline training with contrastive pairs + entropy regularization.

        Returns average total loss across epochs.
        """
        pairs = await self._traces.sample_contrastive_pairs(
            self._config.offline_batch_size
        )
        if len(pairs) < 1:
            logger.info("Offline train: no contrastive pairs available")
            return 0.0

        self._adapter.train()
        epoch_losses = []

        for epoch in range(self._config.offline_epochs):
            epoch_loss = 0.0
            steps = 0

            for good_trace, bad_trace in pairs:
                loss = self._contrastive_step(good_trace, bad_trace)
                if loss is None:
                    continue
                self._offline_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._adapter.parameters(), 1.0)
                self._offline_optimizer.step()
                epoch_loss += loss.item()
                steps += 1

            avg_epoch = epoch_loss / max(steps, 1)
            epoch_losses.append(avg_epoch)
            logger.info("Offline epoch %d/%d  avg_loss=%.4f", epoch + 1, self._config.offline_epochs, avg_epoch)

        self._adapter.eval()
        self._update_count += 1
        await self._maybe_checkpoint()

        return sum(epoch_losses) / max(len(epoch_losses), 1)

    def _reward_weighted_loss(self, trace: dict) -> Optional[torch.Tensor]:
        """Compute reward-weighted cross-entropy on guidance categories."""
        slots_data = trace.get("memory_slots", [])
        steering = trace.get("steering_result")
        if not slots_data or steering is None:
            return None

        query_emb = trace.get("query_embedding", [])
        if not query_emb or len(query_emb) != self._config.embedding_dim:
            return None

        memory_slots = [MemorySlot(**s) for s in slots_data]
        inputs = self._adapter.prepare_input(query_emb, memory_slots)

        steering_vec, guidance_logits, attn_weights = self._adapter(**inputs)

        directives = steering.get("directives", [])
        categories = self._config.guidance_categories
        n_slots = guidance_logits.shape[1]

        target = torch.zeros(1, n_slots, dtype=torch.long)
        for i, d in enumerate(directives[:n_slots]):
            cat = d.get("category", categories[0])
            if cat in categories:
                target[0, i] = categories.index(cat)

        reward = trace.get("reward", 0.0)
        weight = abs(reward) + 0.1

        logits_flat = guidance_logits.view(-1, len(categories))
        target_flat = target.view(-1)
        ce_loss = F.cross_entropy(logits_flat, target_flat)

        if reward >= 0:
            loss = ce_loss * weight
        else:
            loss = -ce_loss * weight * 0.5

        return loss

    def _contrastive_step(
        self, good_trace: dict, bad_trace: dict
    ) -> Optional[torch.Tensor]:
        """Compute contrastive loss between a good and bad trace pair."""
        good_fwd = self._forward_trace(good_trace)
        bad_fwd = self._forward_trace(bad_trace)
        if good_fwd is None or bad_fwd is None:
            return None

        good_vec, good_logits, good_attn = good_fwd
        bad_vec, bad_logits, bad_attn = bad_fwd

        margin = self._config.contrastive_margin
        cos_sim = F.cosine_similarity(good_vec, bad_vec, dim=-1)
        l_contrastive = F.relu(cos_sim + margin).mean()

        categories = self._config.guidance_categories
        n_slots = good_logits.shape[1]

        directives = good_trace.get("steering_result", {}).get("directives", [])
        target = torch.zeros(1, n_slots, dtype=torch.long)
        for i, d in enumerate(directives[:n_slots]):
            cat = d.get("category", categories[0])
            if cat in categories:
                target[0, i] = categories.index(cat)

        logits_flat = good_logits.view(-1, len(categories))
        target_flat = target.view(-1)
        l_reconstruction = F.cross_entropy(logits_flat, target_flat)

        l_attention = self._attention_entropy(good_attn) + self._attention_entropy(bad_attn)

        loss = (
            l_reconstruction
            + self._config.lambda_contrastive * l_contrastive
            + self._config.lambda_attention * l_attention
        )
        return loss

    def _forward_trace(
        self, trace: dict
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass on a single trace, returning (vec, logits, attn)."""
        slots_data = trace.get("memory_slots", [])
        query_emb = trace.get("query_embedding", [])
        if not slots_data or not query_emb or len(query_emb) != self._config.embedding_dim:
            return None

        memory_slots = [MemorySlot(**s) for s in slots_data]
        inputs = self._adapter.prepare_input(query_emb, memory_slots)
        return self._adapter(**inputs)

    @staticmethod
    def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention distribution (encourage focus)."""
        a = attn.view(attn.shape[0], -1)
        a = a + 1e-8
        a = a / a.sum(dim=-1, keepdim=True)
        entropy = -(a * a.log()).sum(dim=-1).mean()
        return entropy

    async def _maybe_checkpoint(self) -> None:
        """Save checkpoint every 10 updates to GridFS."""
        if self._update_count % 10 != 0:
            return
        filename = f"adapter_{self._update_count:06d}.pt"
        if self._gridfs is not None:
            await self._adapter.save_gridfs(self._gridfs, filename, self._update_count)
        else:
            logger.debug("No GridFS bucket configured — skipping checkpoint")
