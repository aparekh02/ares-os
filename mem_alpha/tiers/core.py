from datetime import datetime, timedelta
from typing import Optional

from bson import ObjectId

from mem_alpha.tiers.base import BaseMemoryTier


class CoreMemory(BaseMemoryTier):
    """Key-value scratchpad memory. Stores working context by named keys."""

    async def add(
        self,
        user_id: str,
        key: str,
        content: str,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        ttl_hours: Optional[int] = None,
    ) -> str:
        """Upsert a core memory by key. Updates if user_id+key already exists."""
        embedding = self._embedder.embed(content)
        now = datetime.utcnow()

        ttl_expires = None
        hours = ttl_hours or self._config.core_default_ttl_hours
        if hours:
            ttl_expires = now + timedelta(hours=hours)

        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "key": key,
            "content": content,
            "embedding": embedding,
            "updated_at": now,
            "ttl_expires_at": ttl_expires,
            "metadata": metadata or {},
        }

        result = await self._collection.update_one(
            {"user_id": user_id, "key": key},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )

        if result.upserted_id:
            return str(result.upserted_id)
        existing = await self._collection.find_one(
            {"user_id": user_id, "key": key}, {"_id": 1}
        )
        return str(existing["_id"])

    async def get_by_key(self, user_id: str, key: str) -> Optional[dict]:
        """Retrieve a specific core memory by its key."""
        doc = await self._collection.find_one(
            {"user_id": user_id, "key": key}, {"embedding": 0}
        )
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def get_all_keys(self, user_id: str) -> list[str]:
        """List all core memory keys for a user."""
        cursor = self._collection.find({"user_id": user_id}, {"key": 1, "_id": 0})
        return [doc["key"] async for doc in cursor]

    async def search(
        self, user_id: str, query: str, limit: int = 5, **filters
    ) -> list[dict]:
        query_embedding = self._embedder.embed(query)
        pipeline = self._build_vector_search_pipeline(
            query_embedding, user_id, limit, index_name="core_vector_index"
        )
        results = []
        async for doc in self._collection.aggregate(pipeline):
            doc["tier"] = "core"
            results.append(doc)
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

    async def delete_by_key(self, user_id: str, key: str) -> bool:
        """Delete a core memory by its key name."""
        result = await self._collection.delete_one({"user_id": user_id, "key": key})
        return result.deleted_count > 0
