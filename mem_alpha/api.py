"""
mem-alpha API — primary interface for the RL-replacement framework.

Usage:
    from mem_alpha import MemAlpha

    ma = MemAlpha()
    await ma.connect()

    # One-shot wrap: inject → LLM call → store
    response, trace_id, result = await ma.wrap(
        "user_1", "my query", my_llm_fn
    )

    # Or step-by-step:
    result = await ma.inject("user_1", "my query")
    guidance = ma.get_prompt_block(result)
    # ... your LLM call with guidance ...
    trace_id = await ma.store("user_1", "my query", response)

    # Attach feedback when available
    await ma.feedback(trace_id, Outcome(...))

    await ma.shutdown()
"""

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional, Union

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from mem_alpha.config import MemAlphaConfig
from mem_alpha.manager import MemoryManager
from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.observer import SteeringObserver
from mem_alpha.steering.outcome import Outcome
from mem_alpha.steering.schemas import SteeringResult

logger = logging.getLogger(__name__)


class MemAlpha:
    """Primary interface for the mem-alpha RL-replacement framework.

    Three core operations:
        inject()   — retrieve memories + run adapter → guidance for LLM prompt
        store()    — record LLM response + context → trace + episodic memory
        feedback() — attach outcome/rating → reinforce or demote patterns

    Convenience:
        wrap()     — inject + call your LLM + store in one shot
    """

    def __init__(
        self,
        mem_config: Optional[MemAlphaConfig] = None,
        mgs_config: Optional[MGSConfig] = None,
    ):
        self._mem_config = mem_config or MemAlphaConfig()
        self._mgs_config = mgs_config or MGSConfig()
        self._manager: Optional[MemoryManager] = None
        self._observer: Optional[SteeringObserver] = None
        self._client: Optional[AsyncIOMotorClient] = None

    async def connect(
        self,
        db_uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """Connect to MongoDB and initialize all components.

        Args:
            db_uri:  MongoDB connection string (or reads MONGODB_URI from env)
            db_name: Database name (or reads MONGODB_DB_NAME from env)
        """
        load_dotenv()
        uri = db_uri or os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError(
                "No MongoDB URI provided. Set MONGODB_URI in .env or pass db_uri="
            )

        name = db_name or os.getenv("MONGODB_DB_NAME", self._mem_config.db_name)
        self._mem_config.db_name = name

        # Memory tiers
        self._manager = MemoryManager(self._mem_config)
        await self._manager.connect(uri)

        # Steering observer (adapter + trainer + trace buffer)
        self._client = AsyncIOMotorClient(uri)
        db = self._client[name]
        self._observer = SteeringObserver(self._manager, self._mgs_config)
        await self._observer.connect(db)

        # Background offline training
        self._observer.start_background_trainer()

        logger.info("MemAlpha connected (db=%s)", name)

    # ── WRAP (one-shot) ──────────────────────────────────────────

    async def wrap(
        self,
        user_id: str,
        query: str,
        llm_fn: Callable[[str, str], Union[str, Awaitable[str]]],
    ) -> tuple[str, str, SteeringResult]:
        """Wrap an LLM call with memory injection and storage.

        This is the simplest way to integrate — one call does everything:
        inject guidance → call your LLM → store the trace.

        Args:
            user_id: User/session identifier
            query:   The input query or prompt
            llm_fn:  Callable(guidance_block, query) -> response string.
                     Can be sync or async.

        Returns:
            (response, trace_id, steering_result)
        """
        self._ensure_connected()

        # 1. Inject: get guidance from memory
        result = await self.inject(user_id, query)
        guidance = self.get_prompt_block(result)

        # 2. Call: invoke the LLM with guidance
        response = llm_fn(guidance, query)
        if asyncio.iscoroutine(response) or asyncio.isfuture(response):
            response = await response

        # 3. Store: record the trace
        trace_id = await self.store(user_id, query, response)

        return response, trace_id, result

    # ── INJECT ───────────────────────────────────────────────────

    async def inject(self, user_id: str, query: str) -> SteeringResult:
        """Retrieve memories and generate guidance directives for prompt injection.

        Call this BEFORE your LLM call. The adapter has learned from past
        feedback and produces directives that steer the LLM's response.
        """
        self._ensure_connected()
        return await self._observer.before_call(user_id, query)

    def get_prompt_block(self, result: SteeringResult) -> str:
        """Format a SteeringResult as a text block for system prompt injection."""
        self._ensure_connected()
        return self._observer.get_prompt_block(result)

    # ── STORE ────────────────────────────────────────────────────

    async def store(
        self,
        user_id: str,
        query: str,
        response: str,
        action: Optional[str] = None,
    ) -> str:
        """Record an LLM response with its context. Returns trace_id.

        Automatically triggers non-blocking online training every N calls.
        """
        self._ensure_connected()
        return await self._observer.after_call(user_id, query, response, action)

    # ── FEEDBACK ─────────────────────────────────────────────────

    async def feedback(self, trace_id: str, outcome: Outcome) -> None:
        """Attach feedback/rating to a stored trace.

        The outcome's reward is computed from its deltas and weights.
        Good outcomes (reward > 5) promote patterns to semantic knowledge.
        Bad outcomes (reward < -5) store "avoid" rules.
        The adapter learns from these during background training.
        """
        self._ensure_connected()
        await self._observer.attach_outcome(trace_id, outcome)

    # ── Core Memory ──────────────────────────────────────────────

    async def set_core(self, user_id: str, key: str, content: str) -> str:
        """Store a core working memory entry (upserts by key)."""
        self._ensure_connected()
        return await self._manager.core.add(user_id, key=key, content=content)

    async def get_core(self, user_id: str, key: str) -> Optional[dict]:
        """Retrieve a core memory by key."""
        self._ensure_connected()
        return await self._manager.core.get_by_key(user_id, key)

    # ── Lifecycle ────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Stop background training, save final checkpoint, close connections."""
        if self._observer:
            await self._observer.stop()
        if self._manager:
            await self._manager.close()
        if self._client:
            self._client.close()
        logger.info("MemAlpha shut down")

    def _ensure_connected(self) -> None:
        if self._observer is None or self._manager is None:
            raise RuntimeError("MemAlpha.connect() must be called first")

    @property
    def manager(self) -> MemoryManager:
        """Direct access to the MemoryManager for advanced tier operations."""
        self._ensure_connected()
        return self._manager

    @property
    def observer(self) -> SteeringObserver:
        """Direct access to the SteeringObserver for advanced steering operations."""
        self._ensure_connected()
        return self._observer
