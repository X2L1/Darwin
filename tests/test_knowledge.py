"""Tests for the Knowledge Base and Ingestor."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from darwin.knowledge.base import KnowledgeBase, KnowledgeEntry
from darwin.knowledge.index import score_entries, _tokenise
from darwin.knowledge.ingestor import Ingestor, _parse_subtitle, _strip_html
from darwin.knowledge.retriever import Retriever


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------


class TestKnowledgeBase:
    def test_add_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            entry = KnowledgeEntry(title="Test", source_path="/tmp/test.txt", chunks=["hello world"])
            kb.add_entry(entry)
            loaded = kb.get_entry(entry.entry_id)
            assert loaded is not None
            assert loaded.title == "Test"

    def test_remove_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            entry = KnowledgeEntry(title="ToRemove", chunks=["content"])
            kb.add_entry(entry)
            assert kb.count() == 1
            kb.remove_entry(entry.entry_id)
            assert kb.count() == 0

    def test_list_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            kb.add_entry(KnowledgeEntry(title="A", media_type="text", chunks=["a"]))
            kb.add_entry(KnowledgeEntry(title="B", media_type="video", chunks=["b"]))
            all_entries = kb.list_entries()
            assert len(all_entries) == 2
            text_only = kb.list_entries(media_type="text")
            assert len(text_only) == 1

    def test_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            kb.add_entry(KnowledgeEntry(title="Transformers", chunks=["attention mechanism transformer neural network"]))
            kb.add_entry(KnowledgeEntry(title="Cooking", chunks=["recipe flour butter eggs baking"]))
            results = kb.search("transformer attention", top_k=2)
            assert results[0]["title"] == "Transformers"

    def test_manifest_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb1 = KnowledgeBase(store_dir=tmpdir)
            kb1.add_entry(KnowledgeEntry(title="Persist", chunks=["data"]))
            kb2 = KnowledgeBase(store_dir=tmpdir)   # reload
            assert kb2.count() == 1


# ---------------------------------------------------------------------------
# Ingestor
# ---------------------------------------------------------------------------


class TestIngestor:
    def test_ingest_text_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "notes.txt"
            p.write_text("This is a test document about machine learning.", encoding="utf-8")
            ingestor = Ingestor()
            entries = ingestor.ingest(str(p))
            assert len(entries) == 1
            assert "machine learning" in entries[0].full_text

    def test_ingest_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "readme.md"
            p.write_text("# Title\n\nSome **markdown** content.", encoding="utf-8")
            ingestor = Ingestor()
            entries = ingestor.ingest(str(p))
            assert entries[0].media_type == "markdown"

    def test_ingest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.txt").write_text("File A content.")
            (Path(tmpdir) / "b.txt").write_text("File B content.")
            ingestor = Ingestor()
            entries = ingestor.ingest(tmpdir)
            assert len(entries) == 2

    def test_ingest_srt_subtitle(self) -> None:
        srt = "1\n00:00:01,000 --> 00:00:03,000\nHello world\n\n2\n00:00:04,000 --> 00:00:06,000\nThis is a subtitle\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "clip.srt"
            p.write_text(srt, encoding="utf-8")
            ingestor = Ingestor()
            entries = ingestor.ingest(str(p))
            assert "Hello world" in entries[0].full_text
            assert "subtitle" in entries[0].full_text

    def test_ingest_vtt_subtitle(self) -> None:
        vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHi there\n\n00:00:04.000 --> 00:00:06.000\nHow are you\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "clip.vtt"
            p.write_text(vtt, encoding="utf-8")
            ingestor = Ingestor()
            entries = ingestor.ingest(str(p))
            assert "Hi there" in entries[0].full_text

    def test_ingest_nonexistent_raises(self) -> None:
        ingestor = Ingestor()
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/nonexistent/path/file.txt")

    def test_chunk_splitting(self) -> None:
        ingestor = Ingestor(chunk_size=50, chunk_overlap=10)
        long_text = "word " * 200
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "long.txt"
            p.write_text(long_text, encoding="utf-8")
            entries = ingestor.ingest(str(p))
            assert len(entries[0].chunks) > 1


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class TestIndex:
    def test_tokenise_removes_stopwords(self) -> None:
        tokens = _tokenise("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_score_returns_ordered_results(self) -> None:
        entries = [
            KnowledgeEntry(title="A", chunks=["deep learning neural network training"]),
            KnowledgeEntry(title="B", chunks=["cooking recipes pasta tomato sauce"]),
            KnowledgeEntry(title="C", chunks=["machine learning gradient descent optimizer"]),
        ]
        results = score_entries(entries, "neural network training", top_k=3)
        titles = [r["title"] for r in results]
        # ML-related entries should appear; cooking entry (B) should score lower or be absent
        assert "A" in titles or "C" in titles
        if "B" in titles:
            assert titles.index("B") > min(
                titles.index("A") if "A" in titles else 99,
                titles.index("C") if "C" in titles else 99,
            )

    def test_empty_query(self) -> None:
        entries = [KnowledgeEntry(title="X", chunks=["text"])]
        assert score_entries(entries, "") == []

    def test_empty_entries(self) -> None:
        assert score_entries([], "query") == []


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class TestRetriever:
    def test_get_context_returns_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            kb.add_entry(KnowledgeEntry(title="Doc", chunks=["attention is all you need transformer"]))
            retriever = Retriever(kb, top_k=1)
            ctx = retriever.get_context("transformer attention")
            assert "Reference Context" in ctx
            assert "Doc" in ctx

    def test_get_context_empty_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            retriever = Retriever(kb)
            assert retriever.get_context("anything") == ""

    def test_enrich_agent_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(store_dir=tmpdir)
            kb.add_entry(KnowledgeEntry(title="Code Guide", chunks=["write clean modular python code"]))
            retriever = Retriever(kb, top_k=1)
            ctx: dict = {}
            enriched = retriever.enrich_agent_context("clean code", ctx)
            assert "reference_context" in enriched
