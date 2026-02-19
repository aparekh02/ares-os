"""MGS Trace Buffer — async MongoDB persistence for steering traces.

Stores traces in a MongoDB collection (same cluster as the memory tiers).
Provides record(), attach_outcome(), sample_traces(), and
sample_contrastive_pairs() — all async.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from mem_alpha.steering.outcome import Outcome
from mem_alpha.steering.schemas import MGSTrace

logger = logging.getLogger(__name__)


class MGSTraceBuffer:
    """MongoDB-backed trace buffer for MGS decision cycles."""

    def __init__(self, collection: AsyncIOMotorCollection):
        self._collection = collection

    async def ensure_indexes(self) -> None:
        """Create indexes for efficient querying."""
        await self._collection.create_index("trace_id", unique=True)
        await self._collection.create_index("reward")
        await self._collection.create_index("timestamp")
        await self._collection.create_index([("outcome", 1), ("reward", 1)])
        logger.info("MGSTraceBuffer indexes ensured")

    async def record(self, trace: MGSTrace) -> str:
        """Insert a trace document. Returns the trace_id."""
        doc = trace.model_dump(mode="json")
        await self._collection.insert_one(doc)
        logger.debug("Recorded MGS trace %s  reward=%.2f", trace.trace_id, trace.reward)
        return trace.trace_id

    async def attach_outcome(self, trace_id: str, outcome: Outcome) -> None:
        """Attach an outcome to a previously recorded trace (backfill)."""
        outcome.compute_reward()
        await self._collection.update_one(
            {"trace_id": trace_id},
            {"$set": {"outcome": outcome.to_dict(), "reward": outcome.reward}},
        )
        logger.debug("Attached outcome to %s  reward=%.2f", trace_id, outcome.reward)

    async def size(self) -> int:
        return await self._collection.count_documents({})

    async def sample_traces(
        self,
        n: int,
        *,
        only_with_outcome: bool = False,
        min_reward: float | None = None,
        max_reward: float | None = None,
    ) -> list[dict[str, Any]]:
        """Sample n trace documents matching the filters using MongoDB $sample."""
        match_filter: dict[str, Any] = {}
        if only_with_outcome:
            match_filter["outcome"] = {"$ne": None}
        if min_reward is not None:
            match_filter.setdefault("reward", {})["$gte"] = min_reward
        if max_reward is not None:
            match_filter.setdefault("reward", {})["$lte"] = max_reward

        pipeline: list[dict] = []
        if match_filter:
            pipeline.append({"$match": match_filter})
        pipeline.append({"$sample": {"size": n}})
        pipeline.append({"$project": {"_id": 0}})

        return await self._collection.aggregate(pipeline).to_list(length=n)

    async def sample_contrastive_pairs(
        self,
        n: int,
        good_threshold: float = 0.0,
        bad_threshold: float = 0.0,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Sample (good, bad) trace pairs for contrastive training."""
        base_filter = {"outcome": {"$ne": None}}

        good_pipeline = [
            {"$match": {**base_filter, "reward": {"$gte": good_threshold}}},
            {"$sample": {"size": n}},
            {"$project": {"_id": 0}},
        ]
        bad_pipeline = [
            {"$match": {**base_filter, "reward": {"$lt": bad_threshold}}},
            {"$sample": {"size": n}},
            {"$project": {"_id": 0}},
        ]

        good_traces = await self._collection.aggregate(good_pipeline).to_list(length=n)
        bad_traces = await self._collection.aggregate(bad_pipeline).to_list(length=n)

        if not good_traces or not bad_traces:
            return []

        # Pair them up; if unequal lengths, cycle the shorter list
        pairs = []
        for i in range(min(n, len(good_traces))):
            b = bad_traces[i % len(bad_traces)]
            pairs.append((good_traces[i], b))

        return pairs

    async def get_trace(self, trace_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single trace by ID."""
        return await self._collection.find_one(
            {"trace_id": trace_id}, {"_id": 0}
        )
