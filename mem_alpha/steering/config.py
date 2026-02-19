"""MGS configuration — all hyperparameters for the steering layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MGSConfig:
    """Configuration for Memory-Guided Steering."""

    # Embedding dimensions (must match Embedder output)
    embedding_dim: int = 384

    # Adapter architecture
    adapter_hidden: int = 256
    adapter_heads: int = 4
    adapter_layers: int = 2
    steering_dim: int = 128

    # Memory slots
    max_memory_slots: int = 32
    num_guidance_slots: int = 5
    num_tiers: int = 3  # core, episodic, semantic

    # Guidance categories
    guidance_categories: list[str] = field(
        default_factory=lambda: [
            "prioritize",
            "follow_pattern",
            "avoid",
            "context",
            "constraint",
        ]
    )

    # Confidence threshold for including a directive
    guidance_confidence_threshold: float = 0.3

    # Online training
    online_update_every_n: int = 10
    online_lr: float = 1e-4
    online_batch_size: int = 8

    # Offline training
    offline_lr: float = 3e-5
    offline_epochs: int = 5
    offline_batch_size: int = 16
    contrastive_margin: float = 0.5

    # Loss weights
    lambda_contrastive: float = 0.3
    lambda_attention: float = 0.1

    # Background training
    offline_train_interval_sec: int = 300  # deep offline training every N seconds
    min_traces_for_offline: int = 20       # minimum traces before offline training runs

    # Retrieval
    limit_per_tier: int = 5

    # MongoDB persistence
    traces_collection: str = "mgs_traces"
    checkpoints_bucket: str = "mgs_checkpoints"

    @property
    def num_categories(self) -> int:
        return len(self.guidance_categories)
