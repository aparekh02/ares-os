"""MemoryRetriever — bridge between mem_alpha tiers and the steering adapter.

Wraps MemoryManager.search_all_tiers(), re-embeds results, and packages
them into MemorySlot objects sorted by score.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from mem_alpha.embeddings import Embedder
from mem_alpha.manager import MemoryManager
from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.schemas import MemorySlot

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Retrieves and packages memories for the steering adapter."""

    def __init__(
        self,
        manager: MemoryManager,
        embedder: Embedder,
        config: Optional[MGSConfig] = None,
    ):
        self._manager = manager
        self._embedder = embedder
        self._config = config or MGSConfig()

    async def retrieve(
        self, user_id: str, query: str
    ) -> tuple[list[MemorySlot], list[float], float]:
        """Retrieve memories, embed them, return (slots, query_embedding, latency_ms).

        Uses MemoryManager.search_all_tiers() which runs all 3 tiers in
        parallel via asyncio.gather (manager.py:51-55).
        """
        t0 = time.perf_counter()

        # Embed the query (reuse existing Embedder)
        query_embedding = self._embedder.embed(query)

        # Search all tiers in parallel
        results = await self._manager.search_all_tiers(
            user_id, query, limit_per_tier=self._config.limit_per_tier
        )

        # Flatten and convert to MemorySlots
        slots: list[MemorySlot] = []
        for tier_name, tier_results in results.items():
            for doc in tier_results:
                content = doc.get("content", "")
                # Re-embed: search results strip embeddings; re-embedding
                # ~15 results adds ~15ms, negligible vs LLM latency
                embedding = self._embedder.embed(content)
                confidence = doc.get("confidence", 0.5)
                if tier_name == "core":
                    confidence = 1.0  # core memories are always high-confidence

                slot = MemorySlot(
                    id=str(doc.get("_id", "")),
                    tier=tier_name,
                    content=content,
                    score=doc.get("score", 0.0),
                    confidence=confidence,
                    category=doc.get("category"),
                    embedding=embedding,
                    metadata={
                        k: v
                        for k, v in doc.items()
                        if k not in ("_id", "content", "embedding", "score", "confidence", "category", "tier")
                    },
                )
                slots.append(slot)

        # Sort by score descending, cap at max_memory_slots
        slots.sort(key=lambda s: s.score, reverse=True)
        slots = slots[: self._config.max_memory_slots]

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Retrieved %d memory slots for user=%s in %.1fms",
            len(slots), user_id, latency_ms,
        )
        return slots, query_embedding, latency_ms
