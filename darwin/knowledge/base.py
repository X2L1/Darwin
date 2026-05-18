"""Core data structures for the Darwin Knowledge Base.

The Knowledge Base is a local, file-backed store of reference entries.
Each entry represents one ingested document (text file, subtitle track,
video metadata file, etc.).

Storage layout::

    data/knowledge/
        index.json          ← manifest of all entries
        entries/
            <entry_id>.json ← per-entry metadata + chunked text

No external database is required.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "index.json"
_ENTRIES_DIR = "entries"


@dataclass
class KnowledgeEntry:
    """A single reference document stored in the Knowledge Base."""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    source_path: str = ""          # original file path supplied by the user
    media_type: str = "text"       # "text" | "video" | "audio" | "pdf" | "markdown"
    language: str = "en"
    # Full text content (split into chunks for large docs)
    chunks: List[str] = field(default_factory=list)
    # Metadata extracted from the source (duration, fps, author, …)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_primary_reference: bool = True   # user-supplied refs are always primary
    added_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    word_count: int = 0
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def full_text(self) -> str:
        """Return all chunks joined by newlines."""
        return "\n".join(self.chunks)

    def summary(self, max_chars: int = 200) -> str:
        """Return a short human-readable summary of the entry."""
        text = self.full_text[:max_chars].replace("\n", " ")
        if len(self.full_text) > max_chars:
            text += "…"
        return f"[{self.media_type}] {self.title}: {text}"


class KnowledgeBase:
    """Local, file-backed Knowledge Base.

    All data is stored under *store_dir* (default: ``data/knowledge``).
    The manifest (``index.json``) tracks every ingested entry; individual
    entry payloads live in ``entries/<id>.json``.

    Usage::

        kb = KnowledgeBase()
        kb.add_entry(entry)
        results = kb.search("transformer architecture")
        kb.remove_entry(entry_id)
    """

    def __init__(self, store_dir: str | Path = "data/knowledge") -> None:
        self._dir = Path(store_dir)
        self._entries_dir = self._dir / _ENTRIES_DIR
        self._manifest_path = self._dir / _MANIFEST_FILENAME
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entries_dir.mkdir(parents=True, exist_ok=True)
        self._manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_entry(self, entry: KnowledgeEntry) -> None:
        """Persist *entry* and register it in the manifest."""
        entry_path = self._entries_dir / f"{entry.entry_id}.json"
        entry_path.write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._manifest[entry.entry_id] = {
            "entry_id": entry.entry_id,
            "title": entry.title,
            "source_path": entry.source_path,
            "media_type": entry.media_type,
            "added_at": entry.added_at,
            "word_count": entry.word_count,
            "tags": entry.tags,
            "is_primary_reference": entry.is_primary_reference,
        }
        self._save_manifest()
        logger.info("Knowledge entry added: %s (%s)", entry.title, entry.entry_id)

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Load and return a single entry by id."""
        path = self._entries_dir / f"{entry_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return KnowledgeEntry.from_dict(data)

    def remove_entry(self, entry_id: str) -> bool:
        """Delete an entry from the store. Returns True if it existed."""
        path = self._entries_dir / f"{entry_id}.json"
        if path.exists():
            path.unlink()
        existed = entry_id in self._manifest
        self._manifest.pop(entry_id, None)
        self._save_manifest()
        if existed:
            logger.info("Knowledge entry removed: %s", entry_id)
        return existed

    def list_entries(
        self,
        media_type: Optional[str] = None,
        primary_only: bool = False,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return manifest rows, optionally filtered."""
        rows = list(self._manifest.values())
        if media_type:
            rows = [r for r in rows if r["media_type"] == media_type]
        if primary_only:
            rows = [r for r in rows if r.get("is_primary_reference")]
        if tag:
            rows = [r for r in rows if tag in r.get("tags", [])]
        return rows

    def iter_entries(self) -> Iterator[KnowledgeEntry]:
        """Yield fully-loaded KnowledgeEntry objects for all stored entries."""
        for entry_id in list(self._manifest.keys()):
            entry = self.get_entry(entry_id)
            if entry:
                yield entry

    def count(self) -> int:
        return len(self._manifest)

    # ------------------------------------------------------------------
    # Simple keyword search (full-text, no external service)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        primary_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the *top_k* most relevant entries for *query*.

        Uses a lightweight TF-IDF-style scorer implemented entirely in
        Python – no external library or service required.
        """
        from darwin.knowledge.index import score_entries

        candidates = [
            self.get_entry(eid)
            for eid, meta in self._manifest.items()
            if (not primary_only or meta.get("is_primary_reference"))
        ]
        valid = [e for e in candidates if e is not None]
        return score_entries(valid, query, top_k=top_k)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt manifest – starting fresh.")
        return {}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
