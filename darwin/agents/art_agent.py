"""Art Agent – generates and evaluates visual art improvement proposals.

This agent focuses on:
* Prompt engineering for image generation models
* Style consistency analysis
* Composition quality heuristics
* Dataset curation for art fine-tuning
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from darwin.agents.base import BaseAgent, Proposal

# Aesthetic quality dimensions used for scoring
_AESTHETIC_DIMENSIONS = [
    "composition",
    "colour_harmony",
    "lighting",
    "detail_level",
    "style_consistency",
    "originality",
]

# Known style vocabularies
_STYLE_MODIFIERS = [
    "photorealistic", "impressionist", "cyberpunk", "watercolour",
    "studio lighting", "golden hour", "rule of thirds", "bokeh",
    "8k resolution", "award-winning photography",
]

_NEGATIVE_KEYWORDS = [
    "blurry", "low quality", "artefacts", "distorted", "overexposed",
    "underexposed", "pixelated", "noisy", "jpeg artefacts",
]


class ArtAgent(BaseAgent):
    """Agent that improves Darwin's art-generation capabilities."""

    @property
    def domain(self) -> str:
        return "art"

    def analyse(self, context: Dict[str, Any]) -> List[Proposal]:
        """Analyse art context and return improvement proposals.

        Context keys:
          * ``recent_prompts``     – list of recently used prompt strings
          * ``quality_scores``     – dict mapping prompt_id -> score (0-1)
          * ``style_profile``      – current target style (str)
          * ``dataset_stats``      – dict with dataset statistics
          * ``max_proposals``      – cap on returned proposals
        """
        max_proposals = int(context.get("max_proposals", 5))
        proposals: List[Proposal] = []

        proposals.extend(self._propose_prompt_improvements(context))
        proposals.extend(self._propose_negative_prompt_expansion(context))
        proposals.extend(self._propose_style_fine_tuning(context))
        proposals.extend(self._propose_dataset_curation(context))

        proposals.sort(key=lambda p: p.expected_improvement, reverse=True)
        return proposals[:max_proposals]

    # ------------------------------------------------------------------

    def _propose_prompt_improvements(self, context: Dict[str, Any]) -> List[Proposal]:
        proposals: List[Proposal] = []
        quality_scores: Dict[str, float] = context.get("quality_scores", {})
        recent_prompts: List[str] = context.get("recent_prompts", [])

        low_quality = [pid for pid, score in quality_scores.items() if score < 0.4]
        if low_quality:
            proposals.append(
                self._make_proposal(
                    title=f"Improve {len(low_quality)} low-quality prompt(s)",
                    description=(
                        f"{len(low_quality)} prompts scored below 0.4 on aesthetic quality. "
                        "Enrich them with composition keywords, lighting descriptors, and "
                        "style modifiers drawn from the high-quality prompt library."
                    ),
                    risk_score=0.1,
                    expected_improvement=0.15,
                    metadata={"low_quality_ids": low_quality[:10]},
                )
            )

        # Suggest adding style modifiers to prompts that lack them
        plain_prompts = [
            p for p in recent_prompts
            if not any(mod in p.lower() for mod in _STYLE_MODIFIERS)
        ]
        if plain_prompts:
            sample = _pick_best_modifier()
            proposals.append(
                self._make_proposal(
                    title="Enrich plain prompts with style modifiers",
                    description=(
                        f"{len(plain_prompts)} recent prompts contain no style modifiers. "
                        f"Appending keywords like \"{sample}\" typically improves aesthetic "
                        "scores by 10–20%."
                    ),
                    risk_score=0.05,
                    expected_improvement=0.12,
                    metadata={"plain_prompt_count": len(plain_prompts), "suggested_modifier": sample},
                )
            )
        return proposals

    def _propose_negative_prompt_expansion(self, context: Dict[str, Any]) -> List[Proposal]:
        """Suggest expanding the negative prompt to avoid common artefacts."""
        current_neg: str = context.get("negative_prompt", "")
        missing = [kw for kw in _NEGATIVE_KEYWORDS if kw not in current_neg]
        if not missing:
            return []
        return [
            self._make_proposal(
                title="Expand negative prompt to reduce artefacts",
                description=(
                    f"The current negative prompt is missing: {', '.join(missing[:5])}. "
                    "Adding these terms reduces common generation artefacts."
                ),
                diff=f"negative_prompt += ', {', '.join(missing)}'",
                risk_score=0.05,
                expected_improvement=0.08,
                metadata={"missing_keywords": missing},
            )
        ]

    def _propose_style_fine_tuning(self, context: Dict[str, Any]) -> List[Proposal]:
        """Propose a fine-tuning run on high-quality art samples."""
        dataset_stats: Dict[str, Any] = context.get("dataset_stats", {})
        n_samples = dataset_stats.get("high_quality_samples", 0)
        if n_samples < 100:
            return []
        return [
            self._make_proposal(
                title=f"Fine-tune style model on {n_samples} curated samples",
                description=(
                    f"We have accumulated {n_samples} high-quality art samples. "
                    "A LoRA fine-tuning run on these samples should improve style "
                    "consistency by approximately 15%."
                ),
                risk_score=0.3,
                expected_improvement=0.15,
                metadata={"n_samples": n_samples},
            )
        ]

    def _propose_dataset_curation(self, context: Dict[str, Any]) -> List[Proposal]:
        """Propose steps to grow the curated art dataset."""
        dataset_stats: Dict[str, Any] = context.get("dataset_stats", {})
        total = dataset_stats.get("total_samples", 0)
        if total == 0:
            return [
                self._make_proposal(
                    title="Bootstrap art dataset",
                    description=(
                        "No art dataset exists yet. Collect at least 1 000 diverse, "
                        "high-quality images to enable future fine-tuning."
                    ),
                    risk_score=0.1,
                    expected_improvement=0.2,
                )
            ]
        return []


def _pick_best_modifier() -> str:
    return random.choice(_STYLE_MODIFIERS)
