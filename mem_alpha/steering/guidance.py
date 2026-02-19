"""GuidanceCompiler — decodes adapter logits into human-readable directives.

Converts raw guidance_logits + attention weights into SteeringDirective
objects and formats them for injection into the LLM system prompt.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from mem_alpha.steering.config import MGSConfig
from mem_alpha.steering.schemas import MemorySlot, SteeringDirective

logger = logging.getLogger(__name__)

# Templates for each guidance category
CATEGORY_TEMPLATES: dict[str, str] = {
    "prioritize": "Prioritize: {content}",
    "follow_pattern": "Follow this pattern: {content}",
    "avoid": "Avoid: {content}",
    "context": "Important context: {content}",
    "constraint": "Constraint: {content}",
}

CONFIDENCE_LABELS = {
    (0.8, 1.01): "HIGH",
    (0.5, 0.8): "MED",
    (0.0, 0.5): "LOW",
}


def _confidence_label(conf: float) -> str:
    for (lo, hi), label in CONFIDENCE_LABELS.items():
        if lo <= conf < hi:
            return label
    return "LOW"


class GuidanceCompiler:
    """Decodes guidance logits into directives for prompt injection."""

    def __init__(self, config: Optional[MGSConfig] = None):
        self._config = config or MGSConfig()

    def compile(
        self,
        guidance_logits: torch.Tensor,  # [1, num_slots, num_categories]
        attention_weights: torch.Tensor,  # [1, 1, N] or [1, heads, 1, N]
        memory_slots: list[MemorySlot],
    ) -> list[SteeringDirective]:
        """Decode logits into a list of SteeringDirectives."""
        categories = self._config.guidance_categories
        threshold = self._config.guidance_confidence_threshold

        # Softmax over categories for each guidance slot
        probs = F.softmax(guidance_logits[0], dim=-1)  # [num_slots, num_cats]

        # Flatten attention weights to [N]
        attn = attention_weights.detach()
        while attn.dim() > 1:
            attn = attn.mean(dim=0)
        # attn is now [N]

        n_memories = len(memory_slots)
        directives: list[SteeringDirective] = []

        for slot_idx in range(probs.shape[0]):
            # Best category for this slot
            cat_probs = probs[slot_idx]
            best_cat_idx = cat_probs.argmax().item()
            confidence = cat_probs[best_cat_idx].item()

            if confidence < threshold:
                continue

            category = categories[best_cat_idx]

            # Find the most-attended memory for this slot
            # Use slot_idx as an offset to diversify which memory each slot references
            if n_memories > 0:
                # Weight attention by slot offset to avoid all slots pointing to same memory
                shifted_attn = attn[:n_memories].clone()
                if slot_idx > 0 and n_memories > slot_idx:
                    # Slightly boost memories beyond the top-1
                    shifted_attn[slot_idx % n_memories] += 0.1
                best_mem_idx = shifted_attn.argmax().item()
                best_mem_idx = min(best_mem_idx, n_memories - 1)
                source = memory_slots[best_mem_idx]
            else:
                source = MemorySlot(id="none", tier="unknown", content="no memories available")

            template = CATEGORY_TEMPLATES.get(category, "{content}")
            instruction = template.format(content=source.content)

            directives.append(SteeringDirective(
                category=category,
                instruction=instruction,
                confidence=confidence,
                source_tier=source.tier,
                source_content=source.content,
            ))

        return directives

    @staticmethod
    def format_for_prompt(directives: list[SteeringDirective]) -> str:
        """Format directives as a text block for system prompt injection."""
        if not directives:
            return ""

        lines = ["=== MEMORY-GUIDED STEERING ==="]
        for d in directives:
            label = _confidence_label(d.confidence)
            cat_upper = d.category.upper()
            lines.append(
                f"[{label}] {cat_upper}: {d.instruction} "
                f"(source: {d.source_tier}, conf: {d.confidence:.2f})"
            )
        lines.append("=" * 30)
        return "\n".join(lines)
