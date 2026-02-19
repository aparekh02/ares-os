"""SteeringObserver — orchestrates the MGS pipeline around each LLM call.

Lifecycle:
    observer = SteeringObserver(manager, config)
    await observer.connect(db)
    observer.start_background_trainer()

    # Before LLM call: inject learned guidance
    result = await observer.before_call(user_id, query)
    prompt_block = observer.get_prompt_block(result)

    # After LLM call: store trace, auto-train in background
    trace_id = await observer.after_call(user_id, query, response)

    # When feedback arrives: attach outcome, reinforce/demote memories
    await observer.attach_outcome(trace_id, outcome)

    # Shutdown
    await observer.stop()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Optional

import torch
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket

from mem_alpha.manager import MemoryManager
from mem_alpha.steering.adapter import SteeringAdapter
from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.guidance import GuidanceCompiler
from mem_alpha.steering.outcome import Outcome
from mem_alpha.steering.retriever import MemoryRetriever
from mem_alpha.steering.schemas import MGSTrace, SteeringResult
from mem_alpha.steering.trace_buffer import MGSTraceBuffer
from mem_alpha.steering.trainer import AdapterTrainer

logger = logging.getLogger(__name__)


class SteeringObserver:
    """Orchestrates the full MGS pipeline: store, train, inject.

    The agent never blocks on training — online updates run as fire-and-forget
    asyncio tasks, and deep offline training runs in a background thread.
    """

    def __init__(
        self,
        manager: MemoryManager,
        config: Optional[MGSConfig] = None,
    ):
        self._config = config or MGSConfig()
        self._manager = manager
        self._embedder = manager._embedder

        # Core components
        self._retriever = MemoryRetriever(manager, self._embedder, self._config)
        self._adapter = SteeringAdapter(self._config)
        self._compiler = GuidanceCompiler(self._config)

        # Wired in connect()
        self._trace_buffer: Optional[MGSTraceBuffer] = None
        self._trainer: Optional[AdapterTrainer] = None
        self._gridfs: Optional[AsyncIOMotorGridFSBucket] = None

        # State
        self._cycle_count = 0
        self._lock = asyncio.Lock()

        # Background training
        self._background_task: Optional[asyncio.Task] = None
        self._online_tasks: list[asyncio.Task] = []
        self._stopping = False

        logger.info(
            "SteeringObserver created  adapter_params=%d",
            self._adapter.param_count(),
        )

    async def connect(self, db: AsyncIOMotorDatabase) -> None:
        """Wire MongoDB collections and GridFS, load latest checkpoint."""
        # Traces collection
        traces_coll = db[self._config.traces_collection]
        self._trace_buffer = MGSTraceBuffer(traces_coll)
        await self._trace_buffer.ensure_indexes()

        # GridFS bucket for adapter checkpoints
        self._gridfs = AsyncIOMotorGridFSBucket(
            db, bucket_name=self._config.checkpoints_bucket
        )

        # Trainer
        self._trainer = AdapterTrainer(
            self._adapter, self._trace_buffer, self._config, self._gridfs
        )

        # Load latest checkpoint from GridFS
        await self._load_latest_checkpoint()

        logger.info("SteeringObserver connected to MongoDB")

    async def _load_latest_checkpoint(self) -> None:
        """Load the most recent adapter checkpoint from GridFS."""
        if self._gridfs is None:
            return
        cursor = self._gridfs.find().sort("uploadDate", -1).limit(1)
        files = await cursor.to_list(length=1)
        if files:
            filename = files[0]["filename"]
            try:
                await self._adapter.load_gridfs(self._gridfs, filename)
            except Exception as e:
                logger.warning("Failed to load checkpoint %s: %s", filename, e)

    # ── Inject ────────────────────────────────────────────────────

    async def before_call(
        self, user_id: str, query: str
    ) -> SteeringResult:
        """Retrieve memories, run adapter, return guidance for prompt injection.

        This is the INJECT step — the adapter has learned from past feedback
        and produces directives that steer the LLM's next response.
        """
        t0 = time.perf_counter()

        # Retrieve relevant memories from all tiers
        memory_slots, query_embedding, retrieval_ms = await self._retriever.retrieve(
            user_id, query
        )

        # Run adapter forward pass (lock prevents reads during weight updates)
        async with self._lock:
            adapter_t0 = time.perf_counter()
            inputs = self._adapter.prepare_input(query_embedding, memory_slots)

            with torch.no_grad():
                steering_vector, guidance_logits, attn_weights = self._adapter(**inputs)

            adapter_ms = (time.perf_counter() - adapter_t0) * 1000

        # Compile guidance directives
        directives = self._compiler.compile(guidance_logits, attn_weights, memory_slots)

        total_ms = (time.perf_counter() - t0) * 1000

        result = SteeringResult(
            directives=directives,
            steering_vector=steering_vector[0].tolist(),
            retrieval_latency_ms=retrieval_ms,
            adapter_latency_ms=adapter_ms,
            total_latency_ms=total_ms,
            num_memories_retrieved=len(memory_slots),
        )

        # Stash for after_call
        self._last_query = query
        self._last_user_id = user_id
        self._last_query_embedding = query_embedding
        self._last_memory_slots = memory_slots
        self._last_steering_result = result

        logger.debug(
            "Steering: %d directives, %d memories, %.1fms total",
            len(directives), len(memory_slots), total_ms,
        )
        return result

    # ── Store ─────────────────────────────────────────────────────

    async def after_call(
        self,
        user_id: str,
        query: str,
        response: str,
        action: Optional[str] = None,
    ) -> str:
        """Record the trace after an LLM call. Returns trace_id.

        Fires a non-blocking online training task every N cycles — the agent
        never waits for training to complete.
        """
        if self._trace_buffer is None:
            raise RuntimeError("SteeringObserver.connect() must be called before after_call()")

        trace_id = f"mgs-{int(time.time() * 1000)}-{random.randint(0, 9999):04d}"

        trace = MGSTrace(
            trace_id=trace_id,
            timestamp=time.time(),
            user_id=user_id,
            query=query,
            query_embedding=getattr(self, "_last_query_embedding", []),
            memory_slots=getattr(self, "_last_memory_slots", []),
            steering_result=getattr(self, "_last_steering_result", None),
            action=action,
            response=response,
        )

        await self._trace_buffer.record(trace)

        # Store as episodic memory
        if self._manager.episodic is not None:
            try:
                await self._manager.episodic.add(
                    user_id=user_id,
                    title=f"MGS decision: {query[:80]}",
                    content=f"Query: {query}\nResponse: {response[:500]}",
                    outcome="success",
                )
            except Exception as e:
                logger.warning("Failed to store episodic memory: %s", e)

        # Fire-and-forget online training every N cycles
        self._cycle_count += 1
        if self._cycle_count % self._config.online_update_every_n == 0:
            task = asyncio.create_task(self._online_update_background())
            self._online_tasks.append(task)
            task.add_done_callback(lambda t: self._online_tasks.remove(t) if t in self._online_tasks else None)

        return trace_id

    async def _online_update_background(self) -> None:
        """Run a light online adapter update without blocking the agent."""
        if self._trainer is None:
            return
        try:
            async with self._lock:
                loss = await self._trainer.online_update()
            if loss > 0:
                logger.info("Background online update complete, loss=%.4f", loss)
        except Exception as e:
            logger.warning("Background online update failed: %s", e)

    # ── Feedback ──────────────────────────────────────────────────

    async def attach_outcome(
        self, trace_id: str, outcome: Outcome
    ) -> None:
        """Backfill outcome onto a trace. Reinforce or demote memories based on feedback."""
        if self._trace_buffer is None:
            raise RuntimeError("SteeringObserver.connect() must be called before attach_outcome()")

        outcome.compute_reward()
        await self._trace_buffer.attach_outcome(trace_id, outcome)

        # Reinforce good patterns / store "avoid" patterns in semantic memory
        if self._manager.semantic is not None and outcome.reward != 0.0:
            trace_data = await self._trace_buffer.get_trace(trace_id)

            if trace_data:
                query = trace_data.get("query", "")
                user_id = trace_data.get("user_id", "")

                if outcome.reward > 5.0:
                    try:
                        await self._manager.promote_episode_to_semantic(
                            user_id=user_id,
                            episode_id=trace_id,
                            knowledge=f"Good pattern: {query[:200]}",
                            category="pattern",
                            confidence=min(0.9, 0.5 + outcome.reward / 20.0),
                        )
                    except Exception as e:
                        logger.warning("Failed to promote pattern: %s", e)

                elif outcome.reward < -5.0:
                    try:
                        await self._manager.semantic.add(
                            user_id=user_id,
                            content=f"Avoid: {query[:200]} (led to negative outcome)",
                            category="rule",
                            confidence=min(0.8, 0.5 + abs(outcome.reward) / 20.0),
                        )
                    except Exception as e:
                        logger.warning("Failed to store avoid pattern: %s", e)

    # ── Background Offline Training ──────────────────────────────

    def start_background_trainer(self) -> None:
        """Start the periodic deep offline training loop.

        Runs contrastive (good vs bad) training in a background thread
        every offline_train_interval_sec. The agent never blocks.
        """
        if self._background_task is not None:
            logger.warning("Background trainer already running")
            return
        if self._trainer is None:
            raise RuntimeError("SteeringObserver.connect() must be called before start_background_trainer()")

        self._stopping = False
        self._background_task = asyncio.create_task(self._offline_training_loop())
        logger.info(
            "Background offline trainer started (interval=%ds)",
            self._config.offline_train_interval_sec,
        )

    async def _offline_training_loop(self) -> None:
        """Persistent loop that runs deep offline training periodically."""
        interval = self._config.offline_train_interval_sec

        while not self._stopping:
            await asyncio.sleep(interval)

            if self._stopping:
                break

            try:
                # Check if we have enough traces to train on
                trace_count = await self._trace_buffer.size()
                if trace_count < self._config.min_traces_for_offline:
                    logger.debug(
                        "Offline train skipped: %d traces < %d minimum",
                        trace_count, self._config.min_traces_for_offline,
                    )
                    continue

                logger.info("Starting deep offline training...")

                # Run PyTorch training in a thread to avoid blocking the event loop
                async with self._lock:
                    loss = await asyncio.to_thread(self._run_offline_train_sync)

                logger.info("Offline training complete, avg_loss=%.4f", loss)

            except Exception as e:
                logger.warning("Offline training failed: %s", e)

    def _run_offline_train_sync(self) -> float:
        """Synchronous wrapper for offline training (runs in thread)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._trainer.offline_train())
        finally:
            loop.close()

    async def stop(self) -> None:
        """Cleanly shut down background training and save final checkpoint."""
        self._stopping = True

        # Cancel background offline trainer
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        # Wait for any in-flight online training tasks
        if self._online_tasks:
            await asyncio.gather(*self._online_tasks, return_exceptions=True)
            self._online_tasks.clear()

        # Save final checkpoint
        if self._gridfs is not None and self._trainer is not None:
            try:
                filename = f"adapter_final_{int(time.time())}.pt"
                await self._adapter.save_gridfs(
                    self._gridfs, filename, self._trainer._update_count
                )
                logger.info("Final checkpoint saved: %s", filename)
            except Exception as e:
                logger.warning("Failed to save final checkpoint: %s", e)

        logger.info("SteeringObserver stopped")

    # ── Properties ────────────────────────────────────────────────

    @property
    def adapter(self) -> SteeringAdapter:
        return self._adapter

    @property
    def trace_buffer(self) -> Optional[MGSTraceBuffer]:
        return self._trace_buffer

    @property
    def trainer(self) -> Optional[AdapterTrainer]:
        return self._trainer

    def get_prompt_block(self, result: SteeringResult) -> str:
        """Format a SteeringResult as a text block for system prompt injection."""
        return GuidanceCompiler.format_for_prompt(result.directives)
