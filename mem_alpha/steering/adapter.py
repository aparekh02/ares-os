"""SteeringAdapter — the neural core of MGS (~1.15M params).

A small cross-attention network that learns which memories matter for each
query and compiles them into a steering vector + guidance logits.

Checkpoints can be saved/loaded via MongoDB GridFS (async) or local
filesystem (sync fallback for offline/testing use).
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.schemas import MemorySlot

logger = logging.getLogger(__name__)


class CrossAttentionBlock(nn.Module):
    """Query attends over memory bank via multi-head cross-attention + FFN."""

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,       # [B, 1, D]
        memory: torch.Tensor,      # [B, N, D]
        mask: Optional[torch.Tensor] = None,  # [B, N] True=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: query attends over memories
        attn_out, attn_weights = self.attn(
            query, memory, memory, key_padding_mask=mask
        )
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query, attn_weights


class SteeringAdapter(nn.Module):
    """The trainable adapter that converts memories into steering signals.

    Input:
        query_embedding  [B, 384]
        memory_bank      [B, N, 384]
        tier_ids         [B, N]       (0=core, 1=episodic, 2=semantic)
        confidence       [B, N]
        mask             [B, N]       (True where padded)

    Output:
        steering_vector  [B, 128]
        guidance_logits  [B, num_guidance_slots, num_categories]
        attention_weights [B, num_heads, 1, N]
    """

    def __init__(self, config: Optional[MGSConfig] = None):
        super().__init__()
        self.config = config or MGSConfig()
        c = self.config

        # Enrich memory representations with tier + confidence info (at 384-dim)
        self.tier_embed = nn.Embedding(c.num_tiers, c.embedding_dim)
        self.confidence_proj = nn.Linear(1, c.embedding_dim)

        # Project from embedding_dim (384) down to adapter_hidden (256)
        self.query_proj = nn.Linear(c.embedding_dim, c.adapter_hidden)
        self.memory_proj = nn.Linear(c.embedding_dim, c.adapter_hidden)

        # Cross-attention blocks operate at adapter_hidden (256) dim
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(c.adapter_hidden, c.adapter_heads, c.adapter_hidden)
            for _ in range(c.adapter_layers)
        ])

        # Steering head: adapter_hidden → steering_dim
        self.steering_head = nn.Linear(c.adapter_hidden, c.steering_dim)

        # Guidance decoder: steering vector → logits per slot per category
        self.guidance_decoder = nn.Linear(
            c.steering_dim, c.num_guidance_slots * c.num_categories
        )

        self._num_guidance_slots = c.num_guidance_slots
        self._num_categories = c.num_categories

    def forward(
        self,
        query_embedding: torch.Tensor,   # [B, D]
        memory_bank: torch.Tensor,        # [B, N, D]
        tier_ids: torch.Tensor,           # [B, N]
        confidence: torch.Tensor,         # [B, N]
        mask: torch.Tensor,               # [B, N]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Enrich memories with tier and confidence signals
        tier_emb = self.tier_embed(tier_ids)                    # [B, N, D]
        conf_emb = self.confidence_proj(confidence.unsqueeze(-1))  # [B, N, D]
        enriched = memory_bank + tier_emb + conf_emb            # [B, N, D]
        enriched = self.memory_proj(enriched)

        # Prepare query as [B, 1, D]
        query = self.query_proj(query_embedding).unsqueeze(1)   # [B, 1, D]

        # Run through cross-attention blocks
        all_attn_weights = []
        for block in self.blocks:
            query, attn_w = block(query, enriched, mask)
            all_attn_weights.append(attn_w)

        # Average attention weights across layers
        avg_attn = torch.stack(all_attn_weights).mean(dim=0)  # [B, 1, N]

        # Steering vector from the attended query
        query_squeezed = query.squeeze(1)                      # [B, D]
        steering_vector = self.steering_head(query_squeezed)   # [B, steering_dim]

        # Guidance logits
        raw = self.guidance_decoder(steering_vector)           # [B, slots*cats]
        guidance_logits = raw.view(-1, self._num_guidance_slots, self._num_categories)

        return steering_vector, guidance_logits, avg_attn

    def prepare_input(
        self,
        query_embedding: list[float],
        memory_slots: list[MemorySlot],
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Convert a query embedding + list of MemorySlots into padded tensors."""
        device = device or next(self.parameters()).device
        max_n = self.config.max_memory_slots
        dim = self.config.embedding_dim
        n = min(len(memory_slots), max_n)

        # Query
        q = torch.tensor([query_embedding], dtype=torch.float32, device=device)

        # Memory bank
        bank = torch.zeros(1, max_n, dim, dtype=torch.float32, device=device)
        tiers = torch.zeros(1, max_n, dtype=torch.long, device=device)
        confs = torch.zeros(1, max_n, dtype=torch.float32, device=device)
        pad_mask = torch.ones(1, max_n, dtype=torch.bool, device=device)

        for i in range(n):
            slot = memory_slots[i]
            if len(slot.embedding) == dim:
                bank[0, i] = torch.tensor(slot.embedding, dtype=torch.float32)
            tiers[0, i] = slot.tier_id
            confs[0, i] = slot.confidence
            pad_mask[0, i] = False  # not padded

        return {
            "query_embedding": q,
            "memory_bank": bank,
            "tier_ids": tiers,
            "confidence": confs,
            "mask": pad_mask,
        }

    # ── GridFS (async) persistence ──────────────────────────

    async def save_gridfs(self, gridfs_bucket, filename: str, update_count: int = 0) -> None:
        """Save checkpoint to MongoDB GridFS."""
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        buf.seek(0)
        await gridfs_bucket.upload_from_stream(
            filename,
            buf,
            metadata={"update_count": update_count, "timestamp": time.time()},
        )
        logger.info("Adapter checkpoint saved to GridFS: %s", filename)

    async def load_gridfs(self, gridfs_bucket, filename: str) -> None:
        """Load checkpoint from MongoDB GridFS."""
        buf = io.BytesIO()
        await gridfs_bucket.download_to_stream_by_name(filename, buf)
        buf.seek(0)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        logger.info("Adapter checkpoint loaded from GridFS: %s", filename)

    # ── Local filesystem (sync) persistence ───────────────

    def save_local(self, path: Path | str) -> None:
        """Save checkpoint to local filesystem."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Adapter checkpoint saved to %s", path)

    def load_local(self, path: Path | str) -> None:
        """Load checkpoint from local filesystem."""
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state)
        logger.info("Adapter checkpoint loaded from %s", path)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
