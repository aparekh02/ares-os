from datetime import datetime
from typing import Optional

from bson import ObjectId

from mem_alpha.tiers.base import BaseMemoryTier


class SemanticMemory(BaseMemoryTier):
    """Distilled long-term knowledge with confidence and reinforcement."""

    async def add(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        confidence: float = 0.5,
        source_episode_ids: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a piece of distilled knowledge."""
        embedding = self._embedder.embed(content)
        now = datetime.utcnow()

        doc = {
            "user_id": user_id,
            "category": category,
            "content": content,
            "embedding": embedding,
            "confidence": confidence,
            "source_episode_ids": source_episode_ids or [],
            "reinforcement_count": 1,
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
            "metadata": metadata or {},
        }

        result = await self._collection.insert_one(doc)
        return str(result.inserted_id)

    async def reinforce(
        self, memory_id: str, source_episode_id: Optional[str] = None
    ) -> bool:
        """Reinforce existing knowledge, boosting confidence."""
        now = datetime.utcnow()
        boost = self._config.semantic_reinforcement_boost

        update_stages = [
            {
                "$set": {
                    "reinforcement_count": {"$add": ["$reinforcement_count", 1]},
                    "updated_at": now,
                    "confidence": {
                        "$min": [{"$add": ["$confidence", boost]}, 1.0]
                    },
                    "source_episode_ids": {
                        "$cond": {
                            "if": {"$eq": [source_episode_id, None]},
                            "then": "$source_episode_ids",
                            "else": {
                                "$concatArrays": [
                                    "$source_episode_ids",
                                    [source_episode_id],
                                ]
                            },
                        }
                    },
                }
            }
        ]

        result = await self._collection.update_one(
            {"_id": ObjectId(memory_id)}, update_stages
        )
        return result.modified_count > 0

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        **filters,
    ) -> list[dict]:
        query_embedding = self._embedder.embed(query)
        extra_filters = {}
        if category:
            extra_filters["category"] = category

        pipeline = self._build_vector_search_pipeline(
            query_embedding,
            user_id,
            limit,
            extra_filters=extra_filters,
            index_name="semantic_vector_index",
        )

        if min_confidence > 0.0:
            pipeline.append({"$match": {"confidence": {"$gte": min_confidence}}})

        results = []
        async for doc in self._collection.aggregate(pipeline):
            doc["tier"] = "semantic"
            await self._collection.update_one(
                {"_id": ObjectId(doc["_id"])},
                {"$set": {"last_accessed_at": datetime.utcnow()}},
            )
            results.append(doc)
        return results

    async def get_by_category(
        self, user_id: str, category: str, limit: int = 20
    ) -> list[dict]:
        """Get knowledge filtered by category, sorted by confidence."""
        cursor = (
            self._collection.find(
                {"user_id": user_id, "category": category}, {"embedding": 0}
            )
            .sort("confidence", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        for doc in results:
            doc["_id"] = str(doc["_id"])
        return results

    async def get_recent(self, user_id: str, limit: int = 10) -> list[dict]:
        cursor = (
            self._collection.find({"user_id": user_id}, {"embedding": 0})
            .sort("updated_at", -1)
            .limit(limit)
        )
        results = await cursor.to_list(length=limit)
        for doc in results:
            doc["_id"] = str(doc["_id"])
        return results

    async def delete(self, memory_id: str) -> bool:
        result = await self._collection.delete_one({"_id": ObjectId(memory_id)})
        return result.deleted_count > 0
