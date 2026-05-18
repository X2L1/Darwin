"""Tests for domain agents and the orchestrator."""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List

import pytest

from darwin.agents.base import BaseAgent, Proposal
from darwin.agents.code_agent import CodeAgent
from darwin.agents.art_agent import ArtAgent
from darwin.agents.video_agent import VideoAgent
from darwin.agents.prompting_agent import PromptingAgent
from darwin.agents.research_agent import ResearchAgent
from darwin.orchestrator.fusion import FusionAgent


# ---------------------------------------------------------------------------
# Code Agent
# ---------------------------------------------------------------------------


class TestCodeAgent:
    def test_returns_proposals(self, tmp_path) -> None:
        agent = CodeAgent()
        result = agent.run({"source_root": str(tmp_path), "max_proposals": 5})
        assert result.error is None
        assert isinstance(result.proposals, list)

    def test_detects_complexity(self, tmp_path) -> None:
        # Write a highly complex function (complexity = 11 → above threshold of 10)
        complex_code = """\
def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                while i > 0:
                    if i > 5:
                        if i > 10:
                            if i > 15:
                                if i > 20:
                                    if i > 25:
                                        pass
                                    i -= 1
                            i -= 1
                    i -= 1
    elif x < 0:
        pass
    return x
"""
        py_file = tmp_path / "complex.py"
        py_file.write_text(complex_code)
        agent = CodeAgent()
        result = agent.run({"source_root": str(tmp_path), "focus": ["complexity"], "max_proposals": 10})
        titles = [p.title for p in result.proposals]
        assert any("complexity" in t.lower() or "complex_function" in t for t in titles)

    def test_detects_security_issues(self, tmp_path) -> None:
        py_file = tmp_path / "bad.py"
        py_file.write_text("eval(input())\n")
        agent = CodeAgent()
        result = agent.run({"source_root": str(tmp_path), "focus": ["security"], "max_proposals": 5})
        assert any("eval" in p.title or "Security" in p.title for p in result.proposals)


# ---------------------------------------------------------------------------
# Art Agent
# ---------------------------------------------------------------------------


class TestArtAgent:
    def test_returns_proposals_empty_context(self) -> None:
        agent = ArtAgent()
        result = agent.run({})
        assert result.error is None

    def test_proposes_bootstrap_dataset(self) -> None:
        agent = ArtAgent()
        result = agent.run({"dataset_stats": {"total_samples": 0}})
        assert any("bootstrap" in p.title.lower() or "dataset" in p.title.lower()
                   for p in result.proposals)

    def test_proposes_negative_prompt_expansion(self) -> None:
        agent = ArtAgent()
        result = agent.run({"negative_prompt": ""})
        titles = [p.title.lower() for p in result.proposals]
        assert any("negative" in t or "artefact" in t for t in titles)


# ---------------------------------------------------------------------------
# Video Agent
# ---------------------------------------------------------------------------


class TestVideoAgent:
    def test_empty_context(self) -> None:
        agent = VideoAgent()
        result = agent.run({})
        assert result.error is None

    def test_low_coherence_proposal(self) -> None:
        agent = VideoAgent()
        result = agent.run({"coherence_scores": [0.3, 0.2, 0.9, 0.1]})
        assert any("coherence" in p.title.lower() for p in result.proposals)

    def test_high_latency_proposal(self) -> None:
        agent = VideoAgent()
        result = agent.run({"pipeline_latency_s": 120.0})
        assert any("latency" in p.title.lower() for p in result.proposals)


# ---------------------------------------------------------------------------
# Prompting Agent
# ---------------------------------------------------------------------------


class TestPromptingAgent:
    def test_flags_vague_prompts(self) -> None:
        agent = PromptingAgent()
        library = [{"id": "p1", "text": "do something"}]
        result = agent.run({"prompt_library": library})
        assert any("vague" in p.title.lower() or "clarify" in p.title.lower()
                   for p in result.proposals)

    def test_suggests_cot(self) -> None:
        agent = PromptingAgent()
        library = [{"id": "p1", "text": "Solve this math problem: 2 + 2"}]
        scores = {"p1": 0.2}
        result = agent.run({"prompt_library": library, "response_scores": scores})
        assert any("chain" in p.title.lower() or "step" in p.title.lower()
                   for p in result.proposals)


# ---------------------------------------------------------------------------
# Research Agent
# ---------------------------------------------------------------------------


class TestResearchAgent:
    def test_missing_corpus_proposal(self, tmp_path) -> None:
        agent = ResearchAgent()
        result = agent.run({"corpus_dir": str(tmp_path / "nonexistent")})
        assert any("corpus" in p.title.lower() or "bootstrap" in p.title.lower()
                   for p in result.proposals)

    def test_synthetic_data_proposal(self) -> None:
        agent = ResearchAgent()
        result = agent.run({"synthetic_samples": 0, "topic_frequencies": {"math": 3}})
        assert any("synthetic" in p.title.lower() for p in result.proposals)


# ---------------------------------------------------------------------------
# Fusion Agent
# ---------------------------------------------------------------------------


class TestFusionAgent:
    def _make_proposal(self, title: str, domain: str = "code", risk: float = 0.1, imp: float = 0.2) -> Proposal:
        return Proposal(title=title, domain=domain, risk_score=risk, expected_improvement=imp)

    def test_deduplication(self) -> None:
        fusion = FusionAgent()
        p1 = self._make_proposal("Reduce complexity of foo in bar.py")
        p2 = self._make_proposal("Reduce complexity of foo in bar.py")  # duplicate
        p3 = self._make_proposal("Add docstring to baz")
        fused = fusion.fuse([p1, p2, p3])
        assert len(fused) == 2

    def test_ranking_by_utility(self) -> None:
        fusion = FusionAgent()
        low = self._make_proposal("Low value", imp=0.05)
        high = self._make_proposal("High value", imp=0.4)
        fused = fusion.fuse([low, high])
        assert fused[0].title == "High value"

    def test_empty_input(self) -> None:
        fusion = FusionAgent()
        assert fusion.fuse([]) == []

    def test_kb_boost(self) -> None:
        fusion = FusionAgent()
        p = self._make_proposal("Improve code quality transformer attention", imp=0.1)
        before = p.expected_improvement
        fused = fusion.fuse([p], knowledge_context="transformer attention best practices code quality review")
        assert fused[0].expected_improvement >= before
