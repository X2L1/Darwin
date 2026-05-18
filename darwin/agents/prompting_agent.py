"""Prompting Agent – optimises system prompts and user-facing prompt templates.

Responsibilities:
* Analyses prompt → output quality correlations
* Proposes chain-of-thought or few-shot improvements
* Detects vague or under-specified prompt patterns
* Generates prompt A/B test plans
* Maintains a growing library of effective prompt primitives
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from darwin.agents.base import BaseAgent, Proposal

# Prompt quality heuristics
_VAGUE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(something|stuff|things|do it|make it better)\b", re.I),
    re.compile(r"^\s*(please\s+)?(help|fix|improve)\s*$", re.I),
]

_COT_MARKERS = [
    "let's think step by step",
    "step 1",
    "first,",
    "reasoning:",
    "therefore,",
]

_FEW_SHOT_MARKERS = ["example:", "for instance", "e.g.", "input:", "output:"]

_MIN_PROMPT_LENGTH = 20
_LOW_QUALITY_THRESHOLD = 0.45


class PromptingAgent(BaseAgent):
    """Agent that continuously improves the quality of Darwin's prompts."""

    @property
    def domain(self) -> str:
        return "prompting"

    def analyse(self, context: Dict[str, Any]) -> List[Proposal]:
        """Analyse prompt quality data and return improvement proposals.

        Context keys:
          * ``prompt_library``   – list of {"id", "text", "score"} dicts
          * ``response_scores``  – dict prompt_id -> output quality score
          * ``domain_prompts``   – dict domain -> list of prompt strings
          * ``max_proposals``    – cap on returned proposals
        """
        max_proposals = int(context.get("max_proposals", 8))
        proposals: List[Proposal] = []

        proposals.extend(self._analyse_vague_prompts(context))
        proposals.extend(self._propose_chain_of_thought(context))
        proposals.extend(self._propose_few_shot_examples(context))
        proposals.extend(self._propose_prompt_compression(context))
        proposals.extend(self._propose_ab_tests(context))

        proposals.sort(key=lambda p: p.expected_improvement, reverse=True)
        return proposals[:max_proposals]

    # ------------------------------------------------------------------

    def _analyse_vague_prompts(self, context: Dict[str, Any]) -> List[Proposal]:
        library: List[Dict[str, Any]] = context.get("prompt_library", [])
        vague: List[str] = []
        for entry in library:
            text: str = entry.get("text", "")
            if len(text) < _MIN_PROMPT_LENGTH:
                vague.append(entry.get("id", text[:20]))
                continue
            for pat in _VAGUE_PATTERNS:
                if pat.search(text):
                    vague.append(entry.get("id", text[:20]))
                    break
        if not vague:
            return []
        return [
            self._make_proposal(
                title=f"Clarify {len(vague)} vague prompt(s)",
                description=(
                    f"{len(vague)} prompts contain vague language or are too short. "
                    "Replace hedging phrases with specific, actionable instructions. "
                    "Add context: role, format, constraints, and examples."
                ),
                risk_score=0.05,
                expected_improvement=0.15,
                metadata={"vague_ids": vague[:10]},
            )
        ]

    def _propose_chain_of_thought(self, context: Dict[str, Any]) -> List[Proposal]:
        library: List[Dict[str, Any]] = context.get("prompt_library", [])
        response_scores: Dict[str, float] = context.get("response_scores", {})
        candidates: List[str] = []
        for entry in library:
            text: str = entry.get("text", "")
            pid: str = entry.get("id", "")
            score = response_scores.get(pid, 1.0)
            if score < _LOW_QUALITY_THRESHOLD:
                has_cot = any(m in text.lower() for m in _COT_MARKERS)
                if not has_cot:
                    candidates.append(pid)
        if not candidates:
            return []
        return [
            self._make_proposal(
                title=f"Add chain-of-thought to {len(candidates)} low-scoring prompt(s)",
                description=(
                    f"{len(candidates)} prompts score below {_LOW_QUALITY_THRESHOLD} and lack "
                    "chain-of-thought guidance. Prepending 'Let\\'s think step by step:' "
                    "or explicit reasoning steps typically improves output quality by 15–30%."
                ),
                risk_score=0.05,
                expected_improvement=0.2,
                metadata={"candidate_ids": candidates[:10]},
            )
        ]

    def _propose_few_shot_examples(self, context: Dict[str, Any]) -> List[Proposal]:
        library: List[Dict[str, Any]] = context.get("prompt_library", [])
        response_scores: Dict[str, float] = context.get("response_scores", {})
        candidates: List[str] = []
        for entry in library:
            text: str = entry.get("text", "")
            pid: str = entry.get("id", "")
            score = response_scores.get(pid, 1.0)
            has_few_shot = any(m in text.lower() for m in _FEW_SHOT_MARKERS)
            if score < _LOW_QUALITY_THRESHOLD and not has_few_shot:
                candidates.append(pid)
        if not candidates:
            return []
        return [
            self._make_proposal(
                title=f"Add few-shot examples to {len(candidates)} prompt(s)",
                description=(
                    f"{len(candidates)} prompts may benefit from 2–3 concrete input→output "
                    "examples. Few-shot prompting typically reduces ambiguity and increases "
                    "output quality by 10–25%."
                ),
                risk_score=0.05,
                expected_improvement=0.18,
                metadata={"candidate_ids": candidates[:10]},
            )
        ]

    def _propose_prompt_compression(self, context: Dict[str, Any]) -> List[Proposal]:
        """Flag prompts that are unnecessarily long and could be compressed."""
        library: List[Dict[str, Any]] = context.get("prompt_library", [])
        verbose: List[str] = [
            entry.get("id", "")
            for entry in library
            if len(entry.get("text", "")) > 800
        ]
        if not verbose:
            return []
        return [
            self._make_proposal(
                title=f"Compress {len(verbose)} overly long prompt(s)",
                description=(
                    f"{len(verbose)} prompts exceed 800 characters. Very long prompts can "
                    "dilute attention. Summarise repeated instructions and remove filler text."
                ),
                risk_score=0.1,
                expected_improvement=0.07,
                metadata={"verbose_ids": verbose[:10]},
            )
        ]

    def _propose_ab_tests(self, context: Dict[str, Any]) -> List[Proposal]:
        """Propose A/B test plans for high-impact prompt variants."""
        domain_prompts: Dict[str, List[str]] = context.get("domain_prompts", {})
        plans: List[str] = []
        for domain, prompts in domain_prompts.items():
            if len(prompts) >= 2:
                plans.append(domain)
        if not plans:
            return []
        return [
            self._make_proposal(
                title=f"Run prompt A/B tests across {len(plans)} domain(s)",
                description=(
                    f"Domains with multiple prompt variants ({', '.join(plans[:5])}) should "
                    "be tested head-to-head. Route 50% of traffic to each variant and compare "
                    "output quality scores over 100 samples."
                ),
                risk_score=0.1,
                expected_improvement=0.1,
                metadata={"domains": plans},
            )
        ]
