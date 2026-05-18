"""Tests for governance components: Sandbox, ReviewGate, CheckpointManager."""

from __future__ import annotations

import tempfile

import torch
import pytest

from darwin.governance.review_gate import ReviewGate
from darwin.governance.sandbox import Sandbox
from darwin.agents.base import Proposal


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


class TestSandbox:
    def test_simple_code_runs(self) -> None:
        sb = Sandbox(timeout_seconds=10)
        result = sb.run_code("x = 1 + 1\nprint(x)")
        assert result.success

    def test_syntax_error_captured(self) -> None:
        sb = Sandbox(timeout_seconds=10)
        result = sb.run_code("def broken(:\n    pass")
        assert not result.success

    def test_exception_captured(self) -> None:
        sb = Sandbox(timeout_seconds=10)
        result = sb.run_code("raise ValueError('test error')")
        assert not result.success
        assert "test error" in result.stderr or "ValueError" in result.stderr

    def test_timeout(self) -> None:
        sb = Sandbox(timeout_seconds=2)
        result = sb.run_code("import time; time.sleep(60)")
        assert result.timed_out or not result.success


# ---------------------------------------------------------------------------
# ReviewGate
# ---------------------------------------------------------------------------


class TestReviewGate:
    def test_low_risk_auto_approved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = ReviewGate(risk_threshold=0.7, review_store=tmpdir)
            p = Proposal(title="Safe change", risk_score=0.2)
            decision = gate.evaluate(p)
            assert decision.approved
            assert decision.reviewed_by == "auto"

    def test_high_risk_queued(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = ReviewGate(risk_threshold=0.5, review_store=tmpdir)
            p = Proposal(title="Risky change", risk_score=0.8)
            decision = gate.evaluate(p)
            assert not decision.approved
            assert decision.reviewed_by == "pending"
            pending = gate.list_pending()
            assert any(item["proposal_id"] == p.proposal_id for item in pending)

    def test_resolve_approve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = ReviewGate(risk_threshold=0.5, review_store=tmpdir)
            p = Proposal(title="Review me", risk_score=0.9)
            gate.evaluate(p)
            ok = gate.resolve(p.proposal_id, approved=True, reviewer="alice")
            assert ok
            assert gate.list_pending() == []

    def test_resolve_nonexistent_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = ReviewGate(review_store=tmpdir)
            assert not gate.resolve("nonexistent-id", approved=True)
