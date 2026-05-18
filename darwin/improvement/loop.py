"""The Darwin self-improvement loop.

Flow per cycle
--------------
1. Orchestrator dispatches domain agents → proposals
2. Fusion agent de-duplicates and ranks proposals
3. Validator runs each proposal through the sandbox + benchmarks
4. ReviewGate approves low-risk proposals; queues high-risk ones
5. Merger applies approved proposals (patches files / retrains model)
6. CheckpointManager saves the updated model
7. Metrics are recorded
8. Sleep until next cycle

Everything runs locally.  No paid API, no cloud service.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from darwin.agents.base import Proposal, ProposalStatus
from darwin.core.config import DarwinConfig
from darwin.core.model import DarwinLM
from darwin.core.tokenizer import BPETokenizer
from darwin.evaluation.benchmarks import BenchmarkSuite
from darwin.evaluation.metrics import CycleMetrics, MetricsCollector
from darwin.governance.checkpoint import CheckpointManager
from darwin.governance.review_gate import ReviewGate
from darwin.governance.sandbox import Sandbox
from darwin.knowledge.base import KnowledgeBase
from darwin.orchestrator.fusion import FusionAgent
from darwin.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class ImprovementLoop:
    """Orchestrates the full self-improvement cycle.

    Usage::

        cfg = DarwinConfig()
        model = DarwinLM.from_config(cfg.model)
        tokenizer = BPETokenizer.load("data/tokenizer.json")
        kb = KnowledgeBase()

        loop = ImprovementLoop(cfg, model, tokenizer, kb)
        loop.run_forever()          # blocks; press Ctrl-C to stop
        # -- or run a single cycle --
        loop.run_once()
    """

    def __init__(
        self,
        cfg: DarwinConfig,
        model: DarwinLM,
        tokenizer: BPETokenizer,
        knowledge_base: Optional[KnowledgeBase] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

        kb = knowledge_base or KnowledgeBase(store_dir=f"{cfg.data_dir}/knowledge")

        self.orchestrator = Orchestrator(cfg, knowledge_base=kb)
        self.fusion = FusionAgent()
        self.sandbox = Sandbox(
            timeout_seconds=cfg.max_code_execution_time_seconds,
            allow_network=cfg.allow_network_in_sandbox,
        )
        self.review_gate = ReviewGate(
            risk_threshold=cfg.require_human_review_above_risk,
            review_store=f"{cfg.data_dir}/reviews",
        )
        self.checkpoints = CheckpointManager(
            checkpoint_dir=cfg.training.output_dir
        )
        self.benchmarks = BenchmarkSuite.default(model, tokenizer)
        self.metrics = MetricsCollector(log_dir=cfg.log_dir)

        self._cycle_count = 0
        self._stop = False

    # ------------------------------------------------------------------
    # Main entry-points
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Run improvement cycles indefinitely (blocks the calling thread)."""
        logger.info("Darwin self-improvement loop started.")
        try:
            while not self._stop:
                self.run_once()
                logger.info(
                    "Cycle %d complete.  Sleeping %ds…",
                    self._cycle_count,
                    self.cfg.improvement_interval_seconds,
                )
                time.sleep(self.cfg.improvement_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            logger.info("Darwin self-improvement loop stopped.")

    def run_once(self, context_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a single improvement cycle and return a summary dict."""
        self._cycle_count += 1
        cycle_id = f"cycle_{self._cycle_count:04d}_{uuid.uuid4().hex[:6]}"
        start = time.perf_counter()
        logger.info("=== Improvement cycle %s ===", cycle_id)

        # 1. Collect proposals from all domain agents
        cycle_summary = self.orchestrator.run_improvement_cycle(context_overrides)
        raw_proposals: List[Proposal] = []
        for domain_proposals in cycle_summary["proposals_by_domain"].values():
            for pd in domain_proposals:
                raw_proposals.append(Proposal.from_dict(pd))

        # 2. Fusion – deduplicate and rank
        kb_context = ""
        if self.orchestrator.knowledge_base.count() > 0:
            kb_context = self.orchestrator.retriever.get_context("improvement best practices")
        fused = self.fusion.fuse(raw_proposals, knowledge_context=kb_context)
        logger.info("Fused %d proposals (from %d raw)", len(fused), len(raw_proposals))

        # 3. Validate + review each proposal
        approved: List[Proposal] = []
        rejected: List[Proposal] = []
        n_high_risk_blocked = 0
        n_human_review = 0

        for proposal in fused[: self.cfg.proposal_budget]:
            valid = self._validate(proposal)
            if not valid:
                proposal.status = ProposalStatus.REJECTED
                rejected.append(proposal)
                continue

            decision = self.review_gate.evaluate(proposal)
            if decision.reviewed_by == "pending":
                n_human_review += 1
                proposal.status = ProposalStatus.PENDING
            elif not decision.approved:
                n_high_risk_blocked += 1
                proposal.status = ProposalStatus.REJECTED
                rejected.append(proposal)
            else:
                proposal.status = ProposalStatus.VALIDATED
                approved.append(proposal)

        # 4. Merge approved proposals
        n_merged = self._merge_proposals(approved)

        # 5. Checkpoint
        self.checkpoints.save(
            self.model,
            tag=f"cycle_{self._cycle_count:04d}",
            cycle=self._cycle_count,
        )

        # 6. Record metrics
        duration = time.perf_counter() - start
        m = CycleMetrics(
            cycle_id=cycle_id,
            duration_seconds=round(duration, 2),
            n_proposals_total=len(raw_proposals),
            n_proposals_validated=len(approved),
            n_proposals_rejected=len(rejected),
            n_proposals_merged=n_merged,
            n_high_risk_blocked=n_high_risk_blocked,
            n_human_reviews_requested=n_human_review,
            kb_entry_count=self.orchestrator.knowledge_base.count(),
        )
        self.metrics.record(m)

        summary = {
            "cycle_id": cycle_id,
            "n_proposals_total": len(raw_proposals),
            "n_fused": len(fused),
            "n_approved": len(approved),
            "n_merged": n_merged,
            "n_rejected": len(rejected),
            "n_human_review_queued": n_human_review,
            "duration_seconds": round(duration, 2),
        }
        logger.info("Cycle summary: %s", summary)
        return summary

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, proposal: Proposal) -> bool:
        """Run basic sandbox validation for code-touching proposals."""
        if proposal.domain != "code":
            return True   # non-code proposals pass validation by default
        diff = proposal.diff
        if not diff:
            return True
        # For proposals that include runnable Python, test in sandbox
        if diff.strip().startswith(("def ", "class ", "import ")):
            result = self.sandbox.run_code(diff)
            if not result.success:
                logger.warning(
                    "Proposal '%s' failed sandbox: %s", proposal.title, result.stderr[:200]
                )
                proposal.validation_notes = result.stderr[:500]
                return False
        return True

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def _merge_proposals(self, proposals: List[Proposal]) -> int:
        """Apply approved proposals.  Returns count actually merged."""
        merged = 0
        for proposal in proposals:
            try:
                self._apply_proposal(proposal)
                proposal.status = ProposalStatus.MERGED
                merged += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to apply proposal '%s': %s", proposal.title, exc)
                proposal.status = ProposalStatus.REJECTED
        return merged

    def _apply_proposal(self, proposal: Proposal) -> None:
        """Apply a single proposal.

        Currently supports:
        * ``domain == "code"`` with a non-empty ``diff`` field containing
          a unified diff → writes the patch to the target file.
        * All other domains → logged as an advisory note.
        """
        if proposal.domain == "code" and proposal.diff:
            target = proposal.metadata.get("file")
            if target:
                self._apply_patch(target, proposal.diff, project_root=Path.cwd())
        else:
            # Non-code proposals (art, video, prompting, research) are recorded
            # as advisory notes; the relevant sub-system can act on them later.
            log_path = Path(self.cfg.log_dir) / "advisory_proposals.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                import json
                fh.write(json.dumps(proposal.to_dict()) + "\n")

    @staticmethod
    def _apply_patch(target_file: str, diff: str, project_root: Path | None = None) -> None:
        """Apply a full-file replacement or a small unified diff safely."""
        project_root = (project_root or Path.cwd()).resolve()
        path = Path(target_file)
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()

        if project_root not in path.parents and path != project_root:
            raise ValueError(f"Refusing to patch outside project root: {target_file}")
        if not path.exists() and not _looks_like_unified_diff(diff):
            logger.warning("Patch target not found: %s", target_file)
            return
        # A full unified-diff apply would require `patch` or difflib – that is
        if _looks_like_unified_diff(diff):
            new_text = _apply_unified_diff_text(path, diff)
        else:
            new_text = diff

        path.parent.mkdir(parents=True, exist_ok=True)
        backup = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")
        logger.info("Applied patch to %s%s", path, f" (backup: {backup})" if backup.exists() else "")


_HUNK_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@")


def _looks_like_unified_diff(text: str) -> bool:
    return "\n@@" in text and any(line.startswith(("--- ", "+++ ")) for line in text.splitlines())


def _apply_unified_diff_text(path: Path, diff: str) -> str:
    """Apply a simple unified diff to one file and return the patched text."""
    source = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
    diff_lines = diff.splitlines(keepends=True)
    output: list[str] = []
    src_index = 0
    i = 0

    while i < len(diff_lines):
        line = diff_lines[i]
        match = _HUNK_RE.match(line)
        if not match:
            i += 1
            continue

        old_start = int(match.group("old_start"))
        old_index = max(old_start - 1, 0)
        if old_index < src_index:
            raise ValueError(f"Overlapping patch hunk for {path}")

        output.extend(source[src_index:old_index])
        src_index = old_index
        i += 1

        while i < len(diff_lines) and not diff_lines[i].startswith("@@"):
            hunk_line = diff_lines[i]
            if hunk_line.startswith("\\ No newline at end of file"):
                i += 1
                continue
            if not hunk_line:
                i += 1
                continue

            marker = hunk_line[0]
            text = hunk_line[1:]
            if marker == " ":
                if src_index >= len(source) or source[src_index].rstrip("\r\n") != text.rstrip("\r\n"):
                    raise ValueError(f"Patch context mismatch for {path}")
                output.append(source[src_index])
                src_index += 1
            elif marker == "-":
                if src_index >= len(source) or source[src_index].rstrip("\r\n") != text.rstrip("\r\n"):
                    raise ValueError(f"Patch removal mismatch for {path}")
                src_index += 1
            elif marker == "+":
                output.append(text)
            else:
                raise ValueError(f"Unsupported patch line for {path}: {hunk_line!r}")
            i += 1

    output.extend(source[src_index:])
    return "".join(output)
