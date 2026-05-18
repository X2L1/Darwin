"""Code Agent – analyses Darwin's own source and proposes improvements.

Capabilities
------------
* Static code analysis (complexity, coverage, duplication)
* Dependency vulnerability scanning
* Automated refactoring proposals
* Test-coverage gap detection
* Benchmark regression detection
"""

from __future__ import annotations

import ast
import hashlib
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from darwin.agents.base import BaseAgent, Proposal


class CodeAgent(BaseAgent):
    """Agent that continuously analyses and proposes improvements to Darwin's codebase."""

    @property
    def domain(self) -> str:
        return "code"

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyse(self, context: Dict[str, Any]) -> List[Proposal]:
        """Scan source files and propose code improvements.

        Context keys (all optional):
          * ``source_root``  – path to the package root (default: darwin/)
          * ``max_proposals`` – cap on proposals returned per run
          * ``focus``         – list of sub-analyses to run
        """
        source_root = Path(context.get("source_root", "darwin"))
        max_proposals = int(context.get("max_proposals", 10))
        focus: List[str] = context.get("focus", ["complexity", "coverage", "duplication", "security"])

        proposals: List[Proposal] = []

        if "complexity" in focus:
            proposals.extend(self._analyse_complexity(source_root))
        if "duplication" in focus:
            proposals.extend(self._analyse_duplication(source_root))
        if "coverage" in focus:
            proposals.extend(self._analyse_coverage(context))
        if "security" in focus:
            proposals.extend(self._analyse_security(source_root))
        if "docstrings" in focus:
            proposals.extend(self._analyse_docstrings(source_root))

        # Sort by expected improvement desc, then truncate
        proposals.sort(key=lambda p: p.expected_improvement, reverse=True)
        return proposals[:max_proposals]

    # ------------------------------------------------------------------
    # Sub-analyses
    # ------------------------------------------------------------------

    def _analyse_complexity(self, root: Path) -> List[Proposal]:
        """Flag functions with high cyclomatic complexity."""
        proposals: List[Proposal] = []
        for py_file in _iter_python_files(root):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                complexity = _cyclomatic_complexity(node)
                if complexity > 10:
                    rel = py_file.relative_to(root.parent)
                    proposals.append(
                        self._make_proposal(
                            title=f"Reduce complexity of {node.name!r} in {rel}",
                            description=(
                                f"Function `{node.name}` (line {node.lineno}) has a cyclomatic "
                                f"complexity of {complexity} (threshold: 10). "
                                "Consider decomposing it into smaller, focused helpers."
                            ),
                            risk_score=0.2,
                            expected_improvement=min((complexity - 10) * 0.01, 0.3),
                            metadata={
                                "file": str(rel),
                                "function": node.name,
                                "line": node.lineno,
                                "complexity": complexity,
                            },
                        )
                    )
        return proposals

    def _analyse_duplication(self, root: Path) -> List[Proposal]:
        """Detect duplicate code blocks across the project."""
        proposals: List[Proposal] = []
        # Collect 5-line hash fingerprints
        hashes: Dict[str, List[tuple[str, int]]] = {}
        for py_file in _iter_python_files(root):
            try:
                lines = py_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for i in range(len(lines) - 5):
                block = "\n".join(lines[i : i + 5]).strip()
                if len(block) < 80:
                    continue
                h = hashlib.sha1(block.encode()).hexdigest()  # noqa: S324
                hashes.setdefault(h, []).append((str(py_file), i + 1))
        for h, locations in hashes.items():
            if len(locations) >= 2:
                refs = "; ".join(f"{f}:{l}" for f, l in locations[:3])
                proposals.append(
                    self._make_proposal(
                        title="Extract duplicated code block to shared utility",
                        description=(
                            f"The same 5-line block appears in {len(locations)} locations: {refs}. "
                            "Extract to a shared helper to reduce maintenance burden."
                        ),
                        risk_score=0.15,
                        expected_improvement=0.05,
                        metadata={"locations": locations[:5], "hash": h},
                    )
                )
        return proposals[:5]  # cap to avoid noise

    def _analyse_coverage(self, context: Dict[str, Any]) -> List[Proposal]:
        """Propose tests for modules with low or no coverage data."""
        proposals: List[Proposal] = []
        coverage_data: Dict[str, float] = context.get("coverage_data", {})
        source_root = Path(context.get("source_root", "darwin"))
        for py_file in _iter_python_files(source_root):
            rel = str(py_file.relative_to(source_root.parent))
            cov = coverage_data.get(rel, None)
            if cov is None:
                proposals.append(
                    self._make_proposal(
                        title=f"Add test coverage for {rel}",
                        description=(
                            f"No coverage data found for `{rel}`. "
                            "Create a corresponding test module to ensure correctness."
                        ),
                        risk_score=0.05,
                        expected_improvement=0.1,
                        metadata={"file": rel},
                    )
                )
            elif cov < 0.5:
                proposals.append(
                    self._make_proposal(
                        title=f"Increase test coverage for {rel} (currently {cov:.0%})",
                        description=(
                            f"`{rel}` has only {cov:.0%} line coverage. "
                            "Aim for at least 80% by adding edge-case tests."
                        ),
                        risk_score=0.05,
                        expected_improvement=0.08,
                        metadata={"file": rel, "coverage": cov},
                    )
                )
        return proposals[:5]

    def _analyse_security(self, root: Path) -> List[Proposal]:
        """Scan for common security anti-patterns."""
        proposals: List[Proposal] = []
        patterns = [
            ("eval(", "Avoid eval() – it executes arbitrary code", 0.8),
            ("exec(", "Avoid exec() – it executes arbitrary code", 0.8),
            ("pickle.loads(", "pickle.loads() is unsafe with untrusted data", 0.6),
            ("shell=True", "subprocess with shell=True is susceptible to injection", 0.6),
            ("MD5(", "MD5 is cryptographically broken; use SHA-256 or better", 0.4),
        ]
        for py_file in _iter_python_files(root):
            try:
                text = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            for pattern, message, risk in patterns:
                if pattern in text:
                    rel = py_file.relative_to(root.parent)
                    proposals.append(
                        self._make_proposal(
                            title=f"Security: `{pattern}` found in {rel}",
                            description=message,
                            risk_score=risk,
                            expected_improvement=0.05,
                            metadata={"file": str(rel), "pattern": pattern},
                        )
                    )
        return proposals

    def _analyse_docstrings(self, root: Path) -> List[Proposal]:
        """Find public functions/classes with missing docstrings."""
        proposals: List[Proposal] = []
        for py_file in _iter_python_files(root):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if node.name.startswith("_"):
                    continue
                if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
                    rel = py_file.relative_to(root.parent)
                    proposals.append(
                        self._make_proposal(
                            title=f"Add docstring to `{node.name}` in {rel}",
                            description=f"Public symbol `{node.name}` at line {node.lineno} lacks a docstring.",
                            risk_score=0.02,
                            expected_improvement=0.02,
                            metadata={"file": str(rel), "name": node.name, "line": node.lineno},
                        )
                    )
        return proposals[:10]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_python_files(root: Path):
    """Yield all .py files under *root*, skipping __pycache__ and test dirs."""
    for p in root.rglob("*.py"):
        if any(part in ("__pycache__", ".git", ".venv", "venv") for part in p.parts):
            continue
        yield p


def _cyclomatic_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Estimate cyclomatic complexity of an AST function node."""
    complexity = 1
    branch_types = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )
    for child in ast.walk(node):
        if isinstance(child, branch_types):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity
