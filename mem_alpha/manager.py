import asyncio
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

from mem_alpha.config import MemAlphaConfig
from mem_alpha.embeddings import Embedder
from mem_alpha.tiers.core import CoreMemory
from mem_alpha.tiers.episodic import EpisodicMemory
from mem_alpha.tiers.semantic import SemanticMemory


class MemoryManager:
    """Orchestrates all three memory tiers."""

    def __init__(self, config: Optional[MemAlphaConfig] = None):
        self._config = config or MemAlphaConfig()
        self._client: Optional[AsyncIOMotorClient] = None
        self._db = None
        self._embedder = Embedder(self._config.embedding_model)
        self.core: Optional[CoreMemory] = None
        self.episodic: Optional[EpisodicMemory] = None
        self.semantic: Optional[SemanticMemory] = None

    async def connect(self, db_uri: Optional[str] = None):
        """Connect to MongoDB and initialize all tiers."""
        uri = db_uri or self._config.db_uri
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[self._config.db_name]

        self.core = CoreMemory(
            self._db[self._config.core_collection],
            self._embedder,
            self._config,
        )
        self.episodic = EpisodicMemory(
            self._db[self._config.episodic_collection],
            self._embedder,
            self._config,
        )
        self.semantic = SemanticMemory(
            self._db[self._config.semantic_collection],
            self._embedder,
            self._config,
        )

    async def search_all_tiers(
        self, user_id: str, query: str, limit_per_tier: int = 3
    ) -> dict[str, list[dict]]:
        """Search across all three tiers in parallel."""
        core_results, episodic_results, semantic_results = await asyncio.gather(
            self.core.search(user_id, query, limit=limit_per_tier),
            self.episodic.search(user_id, query, limit=limit_per_tier),
            self.semantic.search(user_id, query, limit=limit_per_tier),
        )
        return {
            "core": core_results,
            "episodic": episodic_results,
            "semantic": semantic_results,
        }

    async def get_context(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        max_core: int = 5,
        max_episodic: int = 3,
        max_semantic: int = 5,
    ) -> dict:
        """Get relevant context from all tiers for the current task."""
        core_keys = await self.core.get_all_keys(user_id)
        core_search, episodic_search, semantic_search = await asyncio.gather(
            self.core.search(user_id, query, limit=max_core),
            self.episodic.search(user_id, query, limit=max_episodic),
            self.semantic.search(user_id, query, limit=max_semantic),
        )

        return {
            "core": {"keys": core_keys, "relevant": core_search},
            "episodic": episodic_search,
            "semantic": semantic_search,
        }

    async def promote_episode_to_semantic(
        self,
        user_id: str,
        episode_id: str,
        knowledge: str,
        category: str = "pattern",
        confidence: float = 0.6,
    ) -> str:
        """Distill an episode into semantic knowledge.

        If very similar knowledge already exists (similarity > threshold),
        reinforces the existing entry instead of creating a duplicate.
        """
        episode = await self._db[self._config.episodic_collection].find_one(
            {"user_id": user_id, "episode_id": episode_id}
        )
        if not episode:
            raise ValueError(f"Episode {episode_id} not found for user {user_id}")

        existing = await self.semantic.search(user_id, knowledge, limit=1)
        threshold = self._config.semantic_dedup_threshold
        if existing and existing[0].get("score", 0) > threshold:
            await self.semantic.reinforce(
                existing[0]["_id"], source_episode_id=episode_id
            )
            return existing[0]["_id"]

        return await self.semantic.add(
            user_id=user_id,
            content=knowledge,
            category=category,
            confidence=confidence,
            source_episode_ids=[episode_id],
        )

    async def clear_user(self, user_id: str) -> dict[str, int]:
        """Clear all memories for a user across all tiers."""
        core_count, episodic_count, semantic_count = await asyncio.gather(
            self.core.clear_user(user_id),
            self.episodic.clear_user(user_id),
            self.semantic.clear_user(user_id),
        )
        return {
            "core": core_count,
            "episodic": episodic_count,
            "semantic": semantic_count,
        }

    async def close(self):
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
