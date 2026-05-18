"""Retriever – injects relevant Knowledge Base context into agent prompts.

Given a query string the Retriever:
1. Searches the Knowledge Base (TF-IDF, local, free).
2. Formats the top-k results as a compact context block.
3. Returns the block ready to be prepended to any agent context dict
   or model prompt.

No external service, no API key, no cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from darwin.knowledge.base import KnowledgeBase


class Retriever:
    """Retrieves relevant chunks from the local Knowledge Base.

    Usage::

        kb = KnowledgeBase()
        retriever = Retriever(kb)
        context_block = retriever.get_context("transformer attention mechanism")
        # → a formatted string with the most relevant passages
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        top_k: int = 3,
        max_chars_per_chunk: int = 600,
        primary_only: bool = False,
    ) -> None:
        self.kb = knowledge_base
        self.top_k = top_k
        self.max_chars_per_chunk = max_chars_per_chunk
        self.primary_only = primary_only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a formatted context block for *query*.

        The block looks like::

            === Reference Context ===
            [1] my_notes.txt (score: 0.82)
            ...relevant passage...

            [2] lecture.mp4 transcript (score: 0.71)
            ...relevant passage...
            === End of Context ===
        """
        results = self.kb.search(
            query, top_k=top_k or self.top_k, primary_only=self.primary_only
        )
        if not results:
            return ""

        lines: List[str] = ["=== Reference Context ==="]
        for i, hit in enumerate(results, 1):
            snippet = hit.get("snippet", "")[:self.max_chars_per_chunk]
            title = hit.get("title", "unknown")
            score = hit.get("score", 0.0)
            lines.append(f"\n[{i}] {title}  (relevance: {score:.2f})")
            lines.append(snippet)
        lines.append("\n=== End of Context ===")
        return "\n".join(lines)

    def get_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return raw search result dicts (entry_id, title, score, snippet)."""
        return self.kb.search(query, top_k=top_k or self.top_k, primary_only=self.primary_only)

    def enrich_agent_context(
        self,
        query: str,
        agent_context: Dict[str, Any],
        context_key: str = "reference_context",
    ) -> Dict[str, Any]:
        """Inject the retrieved context block into *agent_context* and return it.

        Args:
            query:         Query string used to retrieve relevant passages.
            agent_context: The existing context dict passed to an agent.
            context_key:   Key under which the context block will be stored.

        Returns:
            The same dict with the ``context_key`` field populated.
        """
        block = self.get_context(query)
        if block:
            agent_context[context_key] = block
        return agent_context
