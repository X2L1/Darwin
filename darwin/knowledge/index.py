"""TF-IDF search index for the Darwin Knowledge Base.

Implemented entirely in pure Python + stdlib – no external search service,
no Elasticsearch, no paid vector database.  Scores documents using
classic TF-IDF with cosine similarity.

For large corpora (> 100 k entries) you can drop in a free local vector
store (e.g. FAISS or Chroma) without changing the public API.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from darwin.knowledge.base import KnowledgeEntry

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.UNICODE)
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more",
    "most", "other", "some", "such", "than", "too", "very", "just",
    "this", "that", "these", "those", "it", "its", "as", "if",
}


def _tokenise(text: str) -> List[str]:
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOP_WORDS and len(tok) > 1
    ]


# ---------------------------------------------------------------------------
# TF-IDF scorer
# ---------------------------------------------------------------------------


def _tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency (normalised)."""
    counts = Counter(tokens)
    n = max(len(tokens), 1)
    return {term: count / n for term, count in counts.items()}


def _idf(term: str, documents: List[List[str]], n_docs: int) -> float:
    """Inverse document frequency with smoothing."""
    df = sum(1 for doc in documents if term in doc)
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def _cosine(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two term-weight dicts."""
    dot = sum(vec_a.get(t, 0.0) * w for t, w in vec_b.items())
    mag_a = math.sqrt(sum(w * w for w in vec_a.values())) or 1.0
    mag_b = math.sqrt(sum(w * w for w in vec_b.values())) or 1.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Public scoring function
# ---------------------------------------------------------------------------


def score_entries(
    entries: "List[KnowledgeEntry]",
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Score *entries* against *query* using TF-IDF + cosine similarity.

    Returns a list of result dicts (entry metadata + score), sorted
    by relevance descending.
    """
    if not entries or not query.strip():
        return []

    query_tokens = _tokenise(query)
    if not query_tokens:
        return []

    # Tokenise all entry texts (use first 5 chunks to stay fast)
    doc_tokens: List[List[str]] = [
        _tokenise(" ".join(e.chunks[:5])) for e in entries
    ]
    n_docs = len(entries)

    # Build IDF for query terms (only)
    idf_cache: Dict[str, float] = {
        term: _idf(term, doc_tokens, n_docs) for term in set(query_tokens)
    }

    # Query TF-IDF vector
    q_tf = _tf(query_tokens)
    q_vec = {term: q_tf[term] * idf_cache.get(term, 1.0) for term in q_tf}

    results: List[Dict[str, Any]] = []
    for entry, doc_tok in zip(entries, doc_tokens):
        doc_tf = _tf(doc_tok)
        doc_vec = {
            term: doc_tf[term] * idf_cache.get(term, 1.0)
            for term in doc_tok
            if term in idf_cache
        }
        sim = _cosine(q_vec, doc_vec)
        if sim > 0:
            results.append(
                {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "source_path": entry.source_path,
                    "media_type": entry.media_type,
                    "score": round(sim, 4),
                    "snippet": entry.chunks[0][:300] if entry.chunks else "",
                    "tags": entry.tags,
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
