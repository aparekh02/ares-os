"""Pydantic data models for the MGS pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemorySlot(BaseModel):
    """One memory item prepared for the steering adapter."""

    id: str
    tier: str  # "core", "episodic", "semantic"
    content: str
    score: float = 0.0
    confidence: float = 0.5
    category: Optional[str] = None
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tier_id(self) -> int:
        """Map tier name to integer for embedding lookup."""
        return {"core": 0, "episodic": 1, "semantic": 2}.get(self.tier, 0)


class SteeringDirective(BaseModel):
    """One guidance instruction for the LLM prompt."""

    category: str  # e.g. "prioritize", "avoid"
    instruction: str
    confidence: float = 0.5
    source_tier: str = ""
    source_content: str = ""


class SteeringResult(BaseModel):
    """Full output of one steering cycle."""

    directives: list[SteeringDirective] = Field(default_factory=list)
    steering_vector: list[float] = Field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    adapter_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    num_memories_retrieved: int = 0


class MGSTrace(BaseModel):
    """One complete decision cycle for training."""

    trace_id: str
    timestamp: float
    user_id: str
    query: str
    query_embedding: list[float] = Field(default_factory=list)
    memory_slots: list[MemorySlot] = Field(default_factory=list)
    steering_result: Optional[SteeringResult] = None
    action: Optional[str] = None
    response: Optional[str] = None
    outcome: Optional[dict[str, Any]] = None
    reward: float = 0.0
