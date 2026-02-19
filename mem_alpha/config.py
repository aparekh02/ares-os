from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from mem_alpha.steering.config import MGSConfig


@dataclass
class MemAlphaConfig:
    """Configuration for the mem-alpha memory framework."""

    db_uri: str = ""
    db_name: str = "mem_alpha"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Collection names
    core_collection: str = "core_memories"
    episodic_collection: str = "episodic_memories"
    semantic_collection: str = "semantic_memories"

    # Core memory settings
    core_default_ttl_hours: Optional[int] = None
    core_max_entries_per_user: int = 100

    # Episodic memory settings
    episodic_max_actions_per_episode: int = 50

    # Semantic memory settings
    semantic_initial_confidence: float = 0.5
    semantic_reinforcement_boost: float = 0.1
    semantic_dedup_threshold: float = 0.92

    # Steering layer (backward-compatible defaults)
    steering_enabled: bool = False
    steering_config: Optional[MGSConfig] = None
