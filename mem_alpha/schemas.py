from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum


class MemoryTier(str, Enum):
    CORE = "core"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABANDONED = "abandoned"


class SemanticCategory(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    RULE = "rule"
    PATTERN = "pattern"
    SKILL = "skill"


class EpisodeAction(BaseModel):
    action: str
    result: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CoreMemoryDoc(BaseModel):
    id: Optional[str] = None
    user_id: str
    session_id: Optional[str] = None
    key: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_expires_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class EpisodicMemoryDoc(BaseModel):
    id: Optional[str] = None
    user_id: str
    session_id: Optional[str] = None
    episode_id: str
    title: str
    content: str
    actions: list[EpisodeAction] = Field(default_factory=list)
    outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS
    outcome_detail: str = ""
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SemanticMemoryDoc(BaseModel):
    id: Optional[str] = None
    user_id: str
    category: SemanticCategory = SemanticCategory.FACT
    content: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_episode_ids: list[str] = Field(default_factory=list)
    reinforcement_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    id: str
    tier: MemoryTier
    content: str
    score: float
    metadata: dict = Field(default_factory=dict)
