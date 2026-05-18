"""Ingestor – converts user-supplied files into KnowledgeEntry objects.

Supported source types (all processed **locally**, zero cost):

| Extension / type         | How it's handled                                     |
|--------------------------|------------------------------------------------------|
| .txt / .md / .rst / .csv | Read as UTF-8 text                                   |
| .pdf                     | Text extracted with pdfminer.six (optional dep)      |
| .html / .htm             | Tags stripped with stdlib html.parser                |
| .srt / .vtt              | Subtitle tracks → clean transcript text              |
| .json / .jsonl           | Serialised as pretty text                            |
| Video files              | Subtitle/caption sidecar + ffprobe metadata (local)  |
| Audio files              | Transcript via openai-whisper (optional, local only) |
| Any other extension      | Attempted UTF-8 read; skipped if binary              |

No paid API, no network call, no key required for any format.
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from darwin.knowledge.base import KnowledgeEntry

logger = logging.getLogger(__name__)

# Chunk size in characters
_CHUNK_SIZE = 1_500
_CHUNK_OVERLAP = 150

# Video extensions that Darwin understands
_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_SUBTITLE_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa"}


class Ingestor:
    """Convert a file or directory into one or more :class:`~darwin.knowledge.base.KnowledgeEntry` objects.

    Usage::

        ingestor = Ingestor()
        entries = ingestor.ingest("/path/to/my_notes.txt")
        entries = ingestor.ingest("/path/to/lecture.mp4")
        entries = ingestor.ingest("/path/to/docs/")     # whole directory

    All returned entries are ready to be added to a :class:`~darwin.knowledge.base.KnowledgeBase`.
    """

    def __init__(self, chunk_size: int = _CHUNK_SIZE, chunk_overlap: int = _CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def ingest(
        self,
        path: str | Path,
        tags: Optional[List[str]] = None,
        language: str = "en",
        is_primary_reference: bool = True,
    ) -> List[KnowledgeEntry]:
        """Ingest a file or directory and return a list of KnowledgeEntry objects.

        For directories every supported file inside is ingested recursively.

        Note: *path* is a user-supplied local file path.  ``resolve()`` is called
        to canonicalise symlinks and eliminate ``..`` traversal before any I/O.
        Reading arbitrary local files is the intended purpose of this method.
        """
        # Canonicalise path to prevent directory-traversal via symlinks
        path = Path(path).resolve()
        tags = tags or []

        # Validate that the resolved path actually exists before proceeding
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_dir():
            entries: List[KnowledgeEntry] = []
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    try:
                        entries.extend(
                            self.ingest(child, tags=tags, language=language,
                                        is_primary_reference=is_primary_reference)
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Skipping %s: %s", child, exc)
            return entries

        if not path.is_file():
            raise FileNotFoundError(f"Path not found: {path}")

        ext = path.suffix.lower()

        if ext in _VIDEO_EXTENSIONS:
            return self._ingest_video(path, tags, language, is_primary_reference)
        if ext in _AUDIO_EXTENSIONS:
            return self._ingest_audio(path, tags, language, is_primary_reference)
        if ext in _SUBTITLE_EXTENSIONS:
            return [self._ingest_subtitle(path, tags, language, is_primary_reference)]
        if ext == ".pdf":
            return [self._ingest_pdf(path, tags, language, is_primary_reference)]
        if ext in {".html", ".htm"}:
            return [self._ingest_html(path, tags, language, is_primary_reference)]
        if ext in {".json", ".jsonl"}:
            return [self._ingest_json(path, tags, language, is_primary_reference)]
        # Default: treat as plain text
        return [self._ingest_text(path, tags, language, is_primary_reference)]

    # ------------------------------------------------------------------
    # Plain text / Markdown / CSV / RST
    # ------------------------------------------------------------------

    def _ingest_text(
        self,
        path: Path,
        tags: List[str],
        language: str,
        primary: bool,
    ) -> KnowledgeEntry:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeError(f"Cannot read {path}: {exc}") from exc

        media_type = "markdown" if path.suffix.lower() in {".md", ".rst"} else "text"
        return self._build_entry(
            title=path.name,
            source_path=str(path),
            media_type=media_type,
            text=text,
            metadata={"size_bytes": path.stat().st_size},
            tags=tags,
            language=language,
            primary=primary,
        )

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _ingest_html(self, path: Path, tags: List[str], language: str, primary: bool) -> KnowledgeEntry:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = _strip_html(raw)
        return self._build_entry(
            title=path.name,
            source_path=str(path),
            media_type="text",
            text=text,
            metadata={},
            tags=tags,
            language=language,
            primary=primary,
        )

    # ------------------------------------------------------------------
    # PDF (requires pdfminer.six, optional)
    # ------------------------------------------------------------------

    def _ingest_pdf(self, path: Path, tags: List[str], language: str, primary: bool) -> KnowledgeEntry:
        text = _extract_pdf_text(path)
        return self._build_entry(
            title=path.name,
            source_path=str(path),
            media_type="pdf",
            text=text,
            metadata={"size_bytes": path.stat().st_size},
            tags=tags,
            language=language,
            primary=primary,
        )

    # ------------------------------------------------------------------
    # JSON / JSONL
    # ------------------------------------------------------------------

    def _ingest_json(self, path: Path, tags: List[str], language: str, primary: bool) -> KnowledgeEntry:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # JSONL – try line by line
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            text = "\n".join(lines)
        return self._build_entry(
            title=path.name,
            source_path=str(path),
            media_type="text",
            text=text,
            metadata={},
            tags=tags,
            language=language,
            primary=primary,
        )

    # ------------------------------------------------------------------
    # Subtitle files (.srt / .vtt)
    # ------------------------------------------------------------------

    def _ingest_subtitle(
        self, path: Path, tags: List[str], language: str, primary: bool
    ) -> KnowledgeEntry:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = _parse_subtitle(raw, path.suffix.lower())
        return self._build_entry(
            title=path.stem + " (transcript)",
            source_path=str(path),
            media_type="video",
            text=text,
            metadata={"subtitle_format": path.suffix.lower()},
            tags=tags + ["transcript"],
            language=language,
            primary=primary,
        )

    # ------------------------------------------------------------------
    # Video files
    # ------------------------------------------------------------------

    def _ingest_video(
        self, path: Path, tags: List[str], language: str, primary: bool
    ) -> List[KnowledgeEntry]:
        entries: List[KnowledgeEntry] = []

        # 1. Try a sidecar subtitle file (e.g. video.srt alongside video.mp4)
        for sub_ext in _SUBTITLE_EXTENSIONS:
            sub_path = path.with_suffix(sub_ext)
            if sub_path.exists():
                entries.append(
                    self._ingest_subtitle(sub_path, tags + ["sidecar"], language, primary)
                )

        # 2. Probe video metadata with ffprobe (free, local)
        meta = _ffprobe_metadata(path)

        # 3. Optionally transcribe with local Whisper (free, runs on CPU/GPU)
        transcript = _whisper_transcribe(path)
        if transcript:
            entries.append(
                self._build_entry(
                    title=path.stem + " (whisper transcript)",
                    source_path=str(path),
                    media_type="video",
                    text=transcript,
                    metadata=meta,
                    tags=tags + ["transcript", "whisper"],
                    language=language,
                    primary=primary,
                )
            )
        elif not entries:
            # Fallback: store just metadata as a text entry so the file is tracked
            meta_text = f"Video file: {path.name}\n" + "\n".join(
                f"{k}: {v}" for k, v in meta.items()
            )
            entries.append(
                self._build_entry(
                    title=path.stem + " (metadata only)",
                    source_path=str(path),
                    media_type="video",
                    text=meta_text,
                    metadata=meta,
                    tags=tags + ["metadata-only"],
                    language=language,
                    primary=primary,
                )
            )

        return entries

    # ------------------------------------------------------------------
    # Audio files
    # ------------------------------------------------------------------

    def _ingest_audio(
        self, path: Path, tags: List[str], language: str, primary: bool
    ) -> List[KnowledgeEntry]:
        transcript = _whisper_transcribe(path)
        if not transcript:
            transcript = f"Audio file: {path.name}\n(Install openai-whisper for free local transcription)"
        entry = self._build_entry(
            title=path.stem + " (transcript)",
            source_path=str(path),
            media_type="audio",
            text=transcript,
            metadata={},
            tags=tags + ["transcript"],
            language=language,
            primary=primary,
        )
        return [entry]

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        title: str,
        source_path: str,
        media_type: str,
        text: str,
        metadata: Dict[str, Any],
        tags: List[str],
        language: str,
        primary: bool,
    ) -> KnowledgeEntry:
        chunks = _split_into_chunks(text, self.chunk_size, self.chunk_overlap)
        word_count = len(text.split())
        return KnowledgeEntry(
            title=title,
            source_path=source_path,
            media_type=media_type,
            language=language,
            chunks=chunks,
            metadata=metadata,
            is_primary_reference=primary,
            word_count=word_count,
            tags=tags,
        )


# ---------------------------------------------------------------------------
# Helper functions (pure Python + optional free libs)
# ---------------------------------------------------------------------------


def _split_into_chunks(text: str, size: int, overlap: int) -> List[str]:
    """Split *text* into overlapping character-level chunks."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks or [""]


def _strip_html(raw: str) -> str:
    """Remove HTML tags and unescape entities."""
    # Match closing script/style tags with optional whitespace before '>'
    text = re.sub(r"<script[^>]*>.*?</script\s*>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style\s*>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _parse_subtitle(raw: str, fmt: str) -> str:
    """Extract clean dialogue text from SRT or VTT subtitle content."""
    # Remove VTT header
    if fmt == ".vtt":
        raw = re.sub(r"^WEBVTT.*?\n\n", "", raw, flags=re.S)
    # Remove timing lines  (e.g.  00:01:23,456 --> 00:01:26,789)
    raw = re.sub(r"\d{1,2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[.,]\d{3}.*", "", raw)
    # Remove numeric cue identifiers
    raw = re.sub(r"^\d+\s*$", "", raw, flags=re.M)
    # Remove VTT positioning tags
    raw = re.sub(r"<[^>]+>", "", raw)
    # Collapse whitespace
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return " ".join(lines)


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pdfminer.six (free, local)."""
    try:
        from pdfminer.high_level import extract_text  # type: ignore[import]

        return extract_text(str(path))
    except ImportError:
        logger.warning(
            "pdfminer.six not installed – PDF text extraction unavailable. "
            "Install with: pip install pdfminer.six"
        )
        return f"[PDF: {path.name} – install pdfminer.six for text extraction]"
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF extraction failed for %s: %s", path, exc)
        return f"[PDF: {path.name} – extraction error: {exc}]"


def _ffprobe_metadata(path: Path) -> Dict[str, Any]:
    """Extract video metadata using ffprobe (free, local, part of FFmpeg)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            return {
                "duration_s": float(fmt.get("duration", 0)),
                "size_bytes": int(fmt.get("size", 0)),
                "format_name": fmt.get("format_name", ""),
                "bit_rate": int(fmt.get("bit_rate", 0)),
                "streams": len(data.get("streams", [])),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        pass
    return {"file": str(path)}


def _whisper_transcribe(path: Path) -> str:
    """Transcribe audio/video using openai-whisper (free, runs locally on CPU/GPU).

    openai-whisper is open-source (MIT licence) and runs entirely on your
    machine – no API key and no cost.  Install with::

        pip install openai-whisper

    If not installed this function returns an empty string gracefully.
    """
    try:
        import whisper  # type: ignore[import]
    except ImportError:
        return ""
    try:
        logger.info("Transcribing %s with local Whisper (this may take a while)…", path.name)
        model = whisper.load_model("base")   # smallest free model; upgrade to "small", "medium", "large" as needed
        result = model.transcribe(str(path))
        return result.get("text", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Whisper transcription failed for %s: %s", path, exc)
        return ""
