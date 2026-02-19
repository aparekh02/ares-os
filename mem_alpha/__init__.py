"""
mem-alpha: Tiered memory framework that replaces RL fine-tuning for LLM agents.

Usage:
    from mem_alpha import MemAlpha, Outcome

    ma = MemAlpha()
    await ma.connect()

    # One-shot: inject + call + store
    response, trace_id, result = await ma.wrap(user_id, query, my_llm_fn)

    # Attach feedback
    await ma.feedback(trace_id, Outcome(deltas={...}, weights={...}))
"""

from mem_alpha.api import MemAlpha
from mem_alpha.manager import MemoryManager
from mem_alpha.config import MemAlphaConfig
from mem_alpha.tools import create_memory_tools
from mem_alpha.tiers.core import CoreMemory
from mem_alpha.tiers.episodic import EpisodicMemory
from mem_alpha.tiers.semantic import SemanticMemory
from mem_alpha.schemas import MemoryTier, EpisodeOutcome, SemanticCategory

__all__ = [
    "MemAlpha",
    "MemoryManager",
    "MemAlphaConfig",
    "create_memory_tools",
    "CoreMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "MemoryTier",
    "EpisodeOutcome",
    "SemanticCategory",
]

# Steering layer (RL replacement) — graceful degradation if PyTorch not installed
try:
    from mem_alpha.steering import (
        MGSConfig,
        SteeringObserver,
        SteeringAdapter,
        GuidanceCompiler,
        Outcome,
    )
    from mem_alpha.steering.schemas import SteeringResult, SteeringDirective

    __all__ += [
        "MGSConfig",
        "SteeringObserver",
        "SteeringAdapter",
        "GuidanceCompiler",
        "Outcome",
        "SteeringResult",
        "SteeringDirective",
    ]
except ImportError:
    pass

__version__ = "0.1.0"
