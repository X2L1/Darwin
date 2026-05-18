"""Review gate – decides whether a proposal needs human review.

Proposals above the configured risk threshold are held for human approval
before being merged.  Low-risk proposals are auto-approved.

No external service needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from darwin.agents.base import Proposal, ProposalStatus

logger = logging.getLogger(__name__)


@dataclass
class ReviewDecision:
    proposal_id: str
    approved: bool
    reviewed_by: str = "auto"        # "auto" or human reviewer id
    notes: str = ""
    reviewed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ReviewGate:
    """Determines which proposals require human review and records decisions.

    Proposals are written to a local pending-reviews file so a human can
    inspect and approve/reject them via the CLI or API.

    Usage::

        gate = ReviewGate(risk_threshold=0.7)
        decision = gate.evaluate(proposal)
        if decision.approved:
            # safe to merge
    """

    def __init__(
        self,
        risk_threshold: float = 0.7,
        review_store: str | Path = "data/reviews",
    ) -> None:
        self.risk_threshold = risk_threshold
        self._store = Path(review_store)
        self._store.mkdir(parents=True, exist_ok=True)
        self._pending_path = self._store / "pending.json"
        self._decisions_path = self._store / "decisions.jsonl"

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, proposal: Proposal) -> ReviewDecision:
        """Return a ReviewDecision for *proposal*.

        * If risk_score < threshold → auto-approved.
        * Otherwise → queued for human review (returns unapproved decision).
        """
        if proposal.risk_score < self.risk_threshold:
            decision = ReviewDecision(
                proposal_id=proposal.proposal_id,
                approved=True,
                reviewed_by="auto",
                notes=f"Auto-approved (risk {proposal.risk_score:.2f} < {self.risk_threshold})",
            )
        else:
            self._queue_for_review(proposal)
            decision = ReviewDecision(
                proposal_id=proposal.proposal_id,
                approved=False,
                reviewed_by="pending",
                notes=(
                    f"Queued for human review (risk {proposal.risk_score:.2f} ≥ "
                    f"{self.risk_threshold}).  Run `darwin review list` to inspect."
                ),
            )
        self._record_decision(decision)
        return decision

    def evaluate_batch(self, proposals: List[Proposal]) -> List[ReviewDecision]:
        return [self.evaluate(p) for p in proposals]

    # ------------------------------------------------------------------
    # Human review workflow
    # ------------------------------------------------------------------

    def list_pending(self) -> List[Dict[str, Any]]:
        """Return all proposals currently pending human review."""
        if not self._pending_path.exists():
            return []
        try:
            return json.loads(self._pending_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def resolve(self, proposal_id: str, approved: bool, reviewer: str = "human", notes: str = "") -> bool:
        """Record a human decision for *proposal_id*.

        Returns True if the proposal was found in the pending queue.
        """
        pending = self.list_pending()
        remaining = [p for p in pending if p["proposal_id"] != proposal_id]
        if len(remaining) == len(pending):
            return False   # not found
        self._pending_path.write_text(
            json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        decision = ReviewDecision(
            proposal_id=proposal_id,
            approved=approved,
            reviewed_by=reviewer,
            notes=notes,
        )
        self._record_decision(decision)
        logger.info("Review resolved: %s approved=%s by %s", proposal_id, approved, reviewer)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _queue_for_review(self, proposal: Proposal) -> None:
        pending = self.list_pending()
        if not any(p["proposal_id"] == proposal.proposal_id for p in pending):
            pending.append(proposal.to_dict())
            self._pending_path.write_text(
                json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Proposal queued for review: %s", proposal.title)

    def _record_decision(self, decision: ReviewDecision) -> None:
        with self._decisions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(decision)) + "\n")
