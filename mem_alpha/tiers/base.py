from abc import ABC, abstractmethod
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection

from mem_alpha.config import MemAlphaConfig
from mem_alpha.embeddings import Embedder


class BaseMemoryTier(ABC):
    """Abstract base for all memory tiers."""

    def __init__(
        self,
        collection: AsyncIOMotorCollection,
        embedder: Embedder,
        config: MemAlphaConfig,
    ):
        self._collection = collection
        self._embedder = embedder
        self._config = config

    @abstractmethod
    async def add(self, user_id: str, **kwargs) -> str:
        """Store a memory. Returns the document ID."""
        ...

    @abstractmethod
    async def search(
        self, user_id: str, query: str, limit: int = 5, **filters
    ) -> list[dict]:
        """Vector search within this tier."""
        ...

    @abstractmethod
    async def get_recent(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get most recent entries."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a specific entry."""
        ...

    async def clear_user(self, user_id: str) -> int:
        """Delete all entries for a user in this tier."""
        result = await self._collection.delete_many({"user_id": user_id})
        return result.deleted_count

    def _build_vector_search_pipeline(
        self,
        query_embedding: list[float],
        user_id: str,
        limit: int,
        extra_filters: Optional[dict] = None,
        index_name: str = "vector_index",
    ) -> list[dict]:
        """Shared vector search pipeline builder."""
        filter_doc = {"user_id": user_id}
        if extra_filters:
            filter_doc.update(extra_filters)

        return [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": filter_doc,
                }
            },
            {
                "$addFields": {
                    "_id": {"$toString": "$_id"},
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
            {"$project": {"embedding": 0}},
        ]
