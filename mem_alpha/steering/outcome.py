"""Domain-agnostic Outcome for MGS traces.

Uses flexible deltas + weights so any domain can define its own
improvement signals without hardcoded fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Outcome:
    """Observed result after an action — domain-agnostic version.

    Instead of fixed grid metrics, uses flexible dicts so any domain
    can define its own improvement signals.
    """

    deltas: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    human_approved: bool | None = None  # None = no feedback yet
    human_approval_bonus: float = 10.0
    reward: float = 0.0

    def compute_reward(self) -> float:
        """Weighted sum of deltas + human approval bonus."""
        r = 0.0
        for key, delta in self.deltas.items():
            weight = self.weights.get(key, 1.0)
            # Negative delta = improvement → positive reward contribution
            r -= delta * weight
        if self.human_approved is True:
            r += self.human_approval_bonus
        elif self.human_approved is False:
            r -= self.human_approval_bonus
        self.reward = r
        return r

    def to_dict(self) -> dict:
        return {
            "deltas": self.deltas,
            "weights": self.weights,
            "human_approved": self.human_approved,
            "reward": self.reward,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Outcome:
        o = cls(
            deltas=d.get("deltas", {}),
            weights=d.get("weights", {}),
            human_approved=d.get("human_approved"),
        )
        o.reward = d.get("reward", 0.0)
        return o
