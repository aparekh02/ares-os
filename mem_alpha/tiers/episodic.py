from datetime import datetime
from typing import Optional
from uuid import uuid4

from bson import ObjectId

from mem_alpha.tiers.base import BaseMemoryTier


class EpisodicMemory(BaseMemoryTier):
    """Records complete task episodes with outcomes and actions."""

    async def add(
        self,
        user_id: str,
        title: str,
        content: str,
        actions: Optional[list[dict]] = None,
        outcome: str = "success",
        outcome_detail: str = "",
        tags: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a complete episode."""
        embed_text = f"{title}. {content}"
        embedding = self._embedder.embed(embed_text)
        now = datetime.utcnow()

        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "episode_id": str(uuid4()),
            "title": title,
            "content": content,
            "actions": actions or [],
            "outcome": outcome,
            "outcome_detail": outcome_detail,
            "embedding": embedding,
            "started_at": now,
            "ended_at": now,
            "duration_seconds": None,
            "tags": tags or [],
            "metadata": metadata or {},
        }

        result = await self._collection.insert_one(doc)
        return str(result.inserted_id)

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        outcome: Optional[str] = None,
        tags: Optional[list[str]] = None,
        **filters,
    ) -> list[dict]:
        query_embedding = self._embedder.embed(query)
        extra_filters = {}
        if outcome:
            extra_filters["outcome"] = outcome
        if tags:
            extra_filters["tags"] = {"$in": tags}

        pipeline = self._build_vector_search_pipeline(
            query_embedding,
            user_id,
            limit,
            extra_filters=extra_filters,
            index_name="episodic_vector_index",
        )
        results = []
        async for doc in self._collection.aggregate(pipeline):
            doc["tier"] = "episodic"
            results.append(doc)
        return results

    async def get_recent(self, user_id: str, limit: int = 10) -> list[dict]:
        cursor = (
            self._collection.find({"user_id": user_id}, {"embedding": 0})
            .sort("ended_at", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        for doc in results:
            doc["_id"] = str(doc["_id"])
        return results

    async def get_by_outcome(
        self, user_id: str, outcome: str, limit: int = 10
    ) -> list[dict]:
        """Get episodes filtered by outcome."""
        cursor = (
            self._collection.find(
                {"user_id": user_id, "outcome": outcome}, {"embedding": 0}
            )
            .sort("ended_at", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        for doc in results:
            doc["_id"] = str(doc["_id"])
        return results

    async def delete(self, memory_id: str) -> bool:
        result = await self._collection.delete_one({"_id": ObjectId(memory_id)})
        return result.deleted_count > 0
