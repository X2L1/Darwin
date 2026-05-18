"""Fusion/Integrator Agent – merges proposals from all domain agents.

The Fusion agent:
1. Receives proposals from every domain agent.
2. De-duplicates overlapping ideas.
3. Resolves conflicts between proposals that touch the same resource.
4. Ranks the merged set by a combined utility score.
5. Produces a unified "improvement roadmap" for the current cycle.

No external service required.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from darwin.agents.base import Proposal, ProposalStatus

logger = logging.getLogger(__name__)

# Similarity threshold above which two proposals are considered duplicates
_DUPLICATE_SIMILARITY = 0.75


class FusionAgent:
    """Merges and prioritises proposals from all domain agents."""

    def fuse(
        self,
        proposals: List[Proposal],
        knowledge_context: str = "",
    ) -> List[Proposal]:
        """Merge *proposals* into a de-duplicated, ranked roadmap.

        Args:
            proposals:         Raw proposals from all domain agents.
            knowledge_context: Optional context block from the Knowledge Base
                               (used to boost proposals that align with primary
                               references supplied by the user).

        Returns:
            A ranked list of unique proposals ready for the validation step.
        """
        if not proposals:
            return []

        # 1. De-duplicate
        unique = self._deduplicate(proposals)
        logger.info("Fusion: %d proposals → %d after dedup", len(proposals), len(unique))

        # 2. Resolve conflicts (proposals touching the same file/resource)
        resolved = self._resolve_conflicts(unique)
        logger.info("Fusion: %d proposals after conflict resolution", len(resolved))

        # 3. Boost proposals that align with user's primary reference materials
        if knowledge_context:
            resolved = self._boost_from_references(resolved, knowledge_context)

        # 4. Compute combined utility score and rank
        ranked = self._rank(resolved)
        return ranked

    # ------------------------------------------------------------------
    # De-duplication
    # ------------------------------------------------------------------

    def _deduplicate(self, proposals: List[Proposal]) -> List[Proposal]:
        """Remove near-duplicate proposals using title/description similarity."""
        unique: List[Proposal] = []
        for proposal in proposals:
            if not any(_similarity(proposal, kept) >= _DUPLICATE_SIMILARITY for kept in unique):
                unique.append(proposal)
        return unique

    # ------------------------------------------------------------------
    # Conflict resolution
    # ------------------------------------------------------------------

    def _resolve_conflicts(self, proposals: List[Proposal]) -> List[Proposal]:
        """When multiple proposals target the same file/resource, keep the best one."""
        # Group by primary file target (from metadata)
        groups: Dict[str, List[Proposal]] = defaultdict(list)
        no_file: List[Proposal] = []
        for p in proposals:
            target = p.metadata.get("file") or p.metadata.get("source_path") or ""
            if target:
                groups[target].append(p)
            else:
                no_file.append(p)

        resolved: List[Proposal] = list(no_file)
        for target, group in groups.items():
            if len(group) == 1:
                resolved.extend(group)
            else:
                # Keep highest-utility proposal; log others as superseded
                best = max(group, key=lambda p: p.expected_improvement - p.risk_score * 0.5)
                resolved.append(best)
                for other in group:
                    if other is not best:
                        logger.debug(
                            "Proposal '%s' superseded by '%s' (same target: %s)",
                            other.title, best.title, target,
                        )
        return resolved

    # ------------------------------------------------------------------
    # Knowledge-base boosting
    # ------------------------------------------------------------------

    def _boost_from_references(
        self, proposals: List[Proposal], context: str
    ) -> List[Proposal]:
        """Increase expected_improvement for proposals aligned with primary references."""
        context_lower = context.lower()
        for proposal in proposals:
            key_words = _keywords(proposal.title + " " + proposal.description)
            overlap = sum(1 for w in key_words if w in context_lower)
            if overlap > 2:
                boost = min(overlap * 0.02, 0.15)
                proposal.expected_improvement = min(proposal.expected_improvement + boost, 1.0)
                proposal.metadata["kb_boost"] = round(boost, 3)
        return proposals

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _rank(self, proposals: List[Proposal]) -> List[Proposal]:
        """Rank proposals by utility = expected_improvement - 0.4 * risk_score."""
        return sorted(
            proposals,
            key=lambda p: p.expected_improvement - 0.4 * p.risk_score,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarise(self, proposals: List[Proposal]) -> Dict[str, Any]:
        """Return a human-readable summary of the fused roadmap."""
        by_domain: Dict[str, int] = defaultdict(int)
        for p in proposals:
            by_domain[p.domain] += 1
        avg_improvement = (
            sum(p.expected_improvement for p in proposals) / len(proposals)
            if proposals else 0.0
        )
        return {
            "total_proposals": len(proposals),
            "by_domain": dict(by_domain),
            "avg_expected_improvement": round(avg_improvement, 3),
            "top_proposals": [
                {"title": p.title, "domain": p.domain, "improvement": p.expected_improvement}
                for p in proposals[:5]
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _similarity(a: Proposal, b: Proposal) -> float:
    """Approximate title + description similarity using word overlap (Jaccard)."""
    def words(p: Proposal) -> set:
        return set((p.title + " " + p.description).lower().split())

    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _keywords(text: str) -> List[str]:
    import re
    return re.findall(r"[a-z]{4,}", text.lower())
