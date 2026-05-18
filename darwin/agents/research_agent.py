"""Research Agent – discovers knowledge and proposes learning improvements.

All research is performed **locally** using:
* The built-in DarwinLM for summarisation and Q&A
* Local document stores (plain text / JSON files in data/)
* No paid APIs, no external subscriptions, no internet required

Responsibilities:
* Summarise recently ingested documents
* Detect knowledge gaps in the local corpus
* Propose new training data topics
* Benchmark understanding on local Q&A pairs
* Generate synthetic training samples from existing knowledge
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from darwin.agents.base import BaseAgent, Proposal

_MIN_CORPUS_DOCS = 10
_KNOWLEDGE_GAP_THRESHOLD = 0.4   # QA accuracy below this → gap


class ResearchAgent(BaseAgent):
    """Agent that expands and improves Darwin's local knowledge base."""

    @property
    def domain(self) -> str:
        return "research"

    def analyse(self, context: Dict[str, Any]) -> List[Proposal]:
        """Analyse local knowledge state and return improvement proposals.

        Context keys:
          * ``corpus_dir``        – path to local document store
          * ``qa_benchmark``      – list of {"question", "answer", "score"} dicts
          * ``topic_frequencies`` – dict topic -> document count
          * ``last_ingested``     – ISO timestamp of last document ingestion
          * ``synthetic_samples`` – number of synthetic training samples already generated
          * ``max_proposals``     – cap on returned proposals
        """
        max_proposals = int(context.get("max_proposals", 8))
        proposals: List[Proposal] = []

        proposals.extend(self._analyse_corpus_size(context))
        proposals.extend(self._analyse_knowledge_gaps(context))
        proposals.extend(self._propose_synthetic_data(context))
        proposals.extend(self._propose_topic_expansion(context))
        proposals.extend(self._propose_summarisation_pass(context))

        proposals.sort(key=lambda p: p.expected_improvement, reverse=True)
        return proposals[:max_proposals]

    # ------------------------------------------------------------------

    def _analyse_corpus_size(self, context: Dict[str, Any]) -> List[Proposal]:
        corpus_dir = context.get("corpus_dir", "data/corpus")
        path = Path(corpus_dir)
        if not path.exists():
            return [
                self._make_proposal(
                    title="Bootstrap local corpus",
                    description=(
                        "No local corpus directory found. Create `data/corpus/` and populate "
                        "it with plain-text or JSON documents (books, papers, code, etc.) "
                        "to enable knowledge-based improvements. All data stays local – "
                        "no external services required."
                    ),
                    risk_score=0.05,
                    expected_improvement=0.3,
                )
            ]
        n_docs = sum(1 for _ in path.rglob("*.txt")) + sum(1 for _ in path.rglob("*.json"))
        if n_docs < _MIN_CORPUS_DOCS:
            return [
                self._make_proposal(
                    title=f"Grow local corpus (currently {n_docs} docs, target ≥{_MIN_CORPUS_DOCS})",
                    description=(
                        f"The corpus contains only {n_docs} documents. "
                        "Ingest more plain-text sources (e.g. Project Gutenberg, "
                        "Wikipedia dumps, or local files) to broaden the knowledge base."
                    ),
                    risk_score=0.05,
                    expected_improvement=0.2,
                    metadata={"n_docs": n_docs},
                )
            ]
        return []

    def _analyse_knowledge_gaps(self, context: Dict[str, Any]) -> List[Proposal]:
        qa_benchmark: List[Dict[str, Any]] = context.get("qa_benchmark", [])
        if not qa_benchmark:
            return []
        low = [
            item for item in qa_benchmark
            if float(item.get("score", 1.0)) < _KNOWLEDGE_GAP_THRESHOLD
        ]
        if not low:
            return []
        topics = list({item.get("topic", "general") for item in low})
        return [
            self._make_proposal(
                title=f"Fill knowledge gaps in {len(topics)} topic(s)",
                description=(
                    f"{len(low)}/{len(qa_benchmark)} Q&A pairs scored below "
                    f"{_KNOWLEDGE_GAP_THRESHOLD:.0%}. "
                    f"Weak topics: {', '.join(topics[:5])}. "
                    "Ingest more documents on these topics and fine-tune on local Q&A pairs."
                ),
                risk_score=0.2,
                expected_improvement=0.25,
                metadata={"weak_topics": topics[:10], "n_weak": len(low)},
            )
        ]

    def _propose_synthetic_data(self, context: Dict[str, Any]) -> List[Proposal]:
        """Suggest generating synthetic Q&A pairs from the existing corpus."""
        topic_frequencies: Dict[str, int] = context.get("topic_frequencies", {})
        synthetic_samples = int(context.get("synthetic_samples", 0))
        if synthetic_samples >= 5_000:
            return []
        underrepresented = [
            topic for topic, count in topic_frequencies.items() if count < 5
        ]
        target = 5_000 - synthetic_samples
        return [
            self._make_proposal(
                title=f"Generate {target:,} synthetic training samples from local corpus",
                description=(
                    f"Only {synthetic_samples:,} synthetic samples exist (target: 5 000). "
                    "Use the DarwinLM (running locally) to generate question→answer pairs "
                    "from corpus documents. No external API needed."
                    + (
                        f" Prioritise underrepresented topics: {', '.join(underrepresented[:5])}."
                        if underrepresented else ""
                    )
                ),
                risk_score=0.1,
                expected_improvement=0.2,
                metadata={
                    "synthetic_samples": synthetic_samples,
                    "target": target,
                    "underrepresented_topics": underrepresented[:10],
                },
            )
        ]

    def _propose_topic_expansion(self, context: Dict[str, Any]) -> List[Proposal]:
        """Propose adding new topic areas that are absent from the corpus."""
        topic_frequencies: Dict[str, int] = context.get("topic_frequencies", {})
        # Core topics every general-purpose model should know
        expected_topics = {
            "mathematics", "physics", "chemistry", "biology",
            "history", "literature", "computer_science", "philosophy",
            "economics", "linguistics",
        }
        missing = expected_topics - set(topic_frequencies.keys())
        if not missing:
            return []
        return [
            self._make_proposal(
                title=f"Add {len(missing)} missing topic area(s) to corpus",
                description=(
                    f"The following broad topics are absent from the corpus: "
                    f"{', '.join(sorted(missing))}. "
                    "Source free materials (e.g. Wikipedia dumps, Project Gutenberg, "
                    "ArXiv preprints) and ingest them locally."
                ),
                risk_score=0.05,
                expected_improvement=0.15,
                metadata={"missing_topics": sorted(missing)},
            )
        ]

    def _propose_summarisation_pass(self, context: Dict[str, Any]) -> List[Proposal]:
        """Propose summarising long documents to create concise training examples."""
        corpus_dir = context.get("corpus_dir", "data/corpus")
        path = Path(corpus_dir)
        if not path.exists():
            return []
        long_docs = []
        for doc in path.rglob("*.txt"):
            try:
                size = doc.stat().st_size
                if size > 100_000:   # > 100 KB
                    long_docs.append(str(doc))
            except OSError:
                continue
        if not long_docs:
            return []
        return [
            self._make_proposal(
                title=f"Summarise {len(long_docs)} long document(s) for training",
                description=(
                    f"{len(long_docs)} documents exceed 100 KB. "
                    "Run the local DarwinLM in summarisation mode to produce "
                    "concise 512-token summaries suitable as training examples."
                ),
                risk_score=0.1,
                expected_improvement=0.1,
                metadata={"long_doc_count": len(long_docs), "examples": long_docs[:3]},
            )
        ]
