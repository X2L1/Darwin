"""Benchmark suites for evaluating Darwin across all domains.

Each benchmark is self-contained, local, and free.
No external evaluation service or API key required.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from darwin.core.model import DarwinLM
from darwin.core.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base benchmark
# ---------------------------------------------------------------------------


class BaseBenchmark:
    """Abstract base for all domain benchmarks."""

    name: str = "base"

    def run(self, model: DarwinLM, tokenizer: BPETokenizer, **kwargs: Any) -> Dict[str, float]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Language-model perplexity benchmark
# ---------------------------------------------------------------------------


class PerplexityBenchmark(BaseBenchmark):
    """Measures next-token prediction perplexity on a local text file.

    The text file should be a representative sample of the target domain.
    No external data source needed.
    """

    name = "perplexity"

    def __init__(self, data_path: str | Path, max_tokens: int = 8_192) -> None:
        self.data_path = Path(data_path)
        self.max_tokens = max_tokens

    def run(self, model: DarwinLM, tokenizer: BPETokenizer, **kwargs: Any) -> Dict[str, float]:
        if not self.data_path.exists():
            logger.warning("Perplexity benchmark data not found: %s", self.data_path)
            return {"perplexity": float("nan")}

        text = self.data_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, max_length=self.max_tokens + 1)
        if len(token_ids) < 2:
            return {"perplexity": float("nan")}

        device = next(model.parameters()).device
        input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            out = model(input_ids, labels=labels)
        loss = out["loss"].item()
        return {
            "perplexity": round(math.exp(min(loss, 20)), 3),
            "nll_loss": round(loss, 4),
        }


# ---------------------------------------------------------------------------
# Code quality benchmark
# ---------------------------------------------------------------------------


class CodeBenchmark(BaseBenchmark):
    """Measures code generation quality on a local set of completion tasks.

    Tasks are stored in ``data/benchmarks/code_tasks.json`` as a list of::

        {"prompt": "def add(a, b):", "expected_contains": ["return a + b"]}
    """

    name = "code"
    _DEFAULT_PATH = Path("data/benchmarks/code_tasks.json")

    def __init__(self, tasks_path: Optional[str | Path] = None) -> None:
        self.tasks_path = Path(tasks_path) if tasks_path else self._DEFAULT_PATH

    def run(self, model: DarwinLM, tokenizer: BPETokenizer, **kwargs: Any) -> Dict[str, float]:
        tasks = self._load_tasks()
        if not tasks:
            return {"pass_rate": 0.0, "n_tasks": 0}

        device = next(model.parameters()).device
        passed = 0
        for task in tasks:
            prompt = task.get("prompt", "")
            expected_tokens: List[str] = task.get("expected_contains", [])
            ids = tokenizer.encode(prompt, add_bos=True, max_length=model.cfg.max_seq_len - 64)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=64,
                    temperature=0.0,
                    top_k=1,
                    eos_token_id=tokenizer.eos_id,
                )
            generated = tokenizer.decode(out_ids[0, len(ids):].tolist())
            if all(exp in generated for exp in expected_tokens):
                passed += 1

        n = len(tasks)
        return {"pass_rate": round(passed / n, 4), "n_tasks": n, "n_passed": passed}

    def _load_tasks(self) -> List[Dict[str, Any]]:
        if not self.tasks_path.exists():
            return []
        try:
            return json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []


# ---------------------------------------------------------------------------
# Q&A accuracy benchmark (uses local knowledge base Q&A pairs)
# ---------------------------------------------------------------------------


class QABenchmark(BaseBenchmark):
    """Measures question-answering accuracy on local Q&A pairs.

    Pairs stored in ``data/benchmarks/qa_pairs.json``::

        [{"question": "...", "answer": "...", "topic": "..."}]
    """

    name = "qa"
    _DEFAULT_PATH = Path("data/benchmarks/qa_pairs.json")

    def __init__(self, pairs_path: Optional[str | Path] = None) -> None:
        self.pairs_path = Path(pairs_path) if pairs_path else self._DEFAULT_PATH

    def run(self, model: DarwinLM, tokenizer: BPETokenizer, **kwargs: Any) -> Dict[str, float]:
        pairs = self._load_pairs()
        if not pairs:
            return {"accuracy": 0.0, "n_pairs": 0}

        device = next(model.parameters()).device
        correct = 0
        for pair in pairs:
            prompt = f"Question: {pair['question']}\nAnswer:"
            answer_key = pair.get("answer", "").lower().strip()
            ids = tokenizer.encode(prompt, add_bos=True, max_length=model.cfg.max_seq_len - 32)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=32,
                    temperature=0.0,
                    top_k=1,
                    eos_token_id=tokenizer.eos_id,
                )
            generated = tokenizer.decode(out[0, len(ids):].tolist()).lower().strip()
            if answer_key and answer_key[:20] in generated:
                correct += 1

        n = len(pairs)
        return {"accuracy": round(correct / n, 4), "n_pairs": n, "n_correct": correct}

    def _load_pairs(self) -> List[Dict[str, Any]]:
        if not self.pairs_path.exists():
            return []
        try:
            return json.loads(self.pairs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkSuite:
    """Runs all registered benchmarks and returns an aggregated report."""

    def __init__(self, model: DarwinLM, tokenizer: BPETokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._benchmarks: List[BaseBenchmark] = []

    def register(self, benchmark: BaseBenchmark) -> None:
        self._benchmarks.append(benchmark)

    def run_all(self) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for bm in self._benchmarks:
            start = time.perf_counter()
            try:
                scores = bm.run(self.model, self.tokenizer)
            except Exception as exc:  # noqa: BLE001
                logger.error("Benchmark %s failed: %s", bm.name, exc, exc_info=True)
                scores = {"error": 1.0}
            elapsed = round(time.perf_counter() - start, 2)
            scores["duration_seconds"] = elapsed
            results[bm.name] = scores
            logger.info("Benchmark %s: %s", bm.name, scores)
        return results

    @classmethod
    def default(cls, model: DarwinLM, tokenizer: BPETokenizer) -> "BenchmarkSuite":
        """Create a suite with the standard set of benchmarks."""
        suite = cls(model, tokenizer)
        suite.register(QABenchmark())
        suite.register(CodeBenchmark())
        return suite
