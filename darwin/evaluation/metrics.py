"""Metrics collection for the Darwin self-improvement system.

All metrics are stored locally as JSON lines in data/logs/metrics.jsonl.
No external monitoring service required.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Metrics recorded for a single improvement cycle."""

    cycle_id: str
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0

    # Agent-level
    n_proposals_total: int = 0
    n_proposals_validated: int = 0
    n_proposals_rejected: int = 0
    n_proposals_merged: int = 0

    # Model quality
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None

    # Domain-specific KPIs
    code_coverage: Optional[float] = None
    art_quality_score: Optional[float] = None
    video_coherence: Optional[float] = None
    prompting_quality: Optional[float] = None
    research_qa_accuracy: Optional[float] = None

    # Knowledge Base
    kb_entry_count: int = 0

    # Safety
    n_high_risk_blocked: int = 0
    n_human_reviews_requested: int = 0

    extra: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and persists metrics locally as JSON lines.

    Usage::

        collector = MetricsCollector()
        m = CycleMetrics(cycle_id="c001", n_proposals_total=12)
        collector.record(m)
        recent = collector.load_recent(n=10)
    """

    def __init__(self, log_dir: str | Path = "data/logs") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._log_dir / "metrics.jsonl"

    def record(self, metrics: CycleMetrics) -> None:
        """Append *metrics* to the local JSONL store."""
        row = asdict(metrics)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def load_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the *n* most recent metric rows."""
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in reversed(lines[-n * 2:]):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return list(reversed(rows[-n:]))

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all recorded cycles."""
        rows = self.load_recent(n=1_000)
        if not rows:
            return {"total_cycles": 0}
        total = len(rows)
        avg_proposals = sum(r.get("n_proposals_total", 0) for r in rows) / total
        avg_merged = sum(r.get("n_proposals_merged", 0) for r in rows) / total
        losses = [r["eval_loss"] for r in rows if r.get("eval_loss") is not None]
        return {
            "total_cycles": total,
            "avg_proposals_per_cycle": round(avg_proposals, 1),
            "avg_merged_per_cycle": round(avg_merged, 1),
            "latest_eval_loss": losses[-1] if losses else None,
            "best_eval_loss": min(losses) if losses else None,
        }
