"""Memory-Guided Steering (MGS) — RL replacement via mem_alpha.

A neural steering layer that replaces RL fine-tuning. Instead of changing
the LLM's weights, MGS injects memory-derived guidance into every prompt
via a small trainable adapter (~1M params).
"""

from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.outcome import Outcome
from mem_alpha.steering.schemas import (
    MGSTrace,
    MemorySlot,
    SteeringDirective,
    SteeringResult,
)

__all__ = [
    "MGSConfig",
    "Outcome",
    "MemorySlot",
    "SteeringDirective",
    "SteeringResult",
    "MGSTrace",
]

# PyTorch-dependent classes — graceful degradation if torch not installed
try:
    from mem_alpha.steering.adapter import SteeringAdapter
    from mem_alpha.steering.guidance import GuidanceCompiler
    from mem_alpha.steering.observer import SteeringObserver
    from mem_alpha.steering.retriever import MemoryRetriever
    from mem_alpha.steering.trace_buffer import MGSTraceBuffer
    from mem_alpha.steering.trainer import AdapterTrainer

    __all__ += [
        "SteeringAdapter",
        "GuidanceCompiler",
        "SteeringObserver",
        "MemoryRetriever",
        "MGSTraceBuffer",
        "AdapterTrainer",
    ]
except ImportError:
    pass
