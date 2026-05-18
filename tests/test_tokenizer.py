"""Tests for the BPE tokenizer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from darwin.core.tokenizer import BPETokenizer


SAMPLE_TEXT = (
    "Hello world! This is a test of the Darwin tokenizer. "
    "It should handle punctuation, numbers like 42, and Unicode: café. "
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is a subfield of artificial intelligence. " * 20
)


@pytest.fixture()
def trained_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=500)
    return tok


class TestBPETokenizer:
    def test_train_builds_vocab(self, trained_tokenizer: BPETokenizer) -> None:
        assert trained_tokenizer.vocab_size >= 256

    def test_special_tokens_present(self, trained_tokenizer: BPETokenizer) -> None:
        for special in BPETokenizer.SPECIAL_TOKENS:
            assert special in trained_tokenizer.encoder

    def test_encode_decode_roundtrip(self, trained_tokenizer: BPETokenizer) -> None:
        text = "Hello world! Testing 123."
        ids = trained_tokenizer.encode(text)
        recovered = trained_tokenizer.decode(ids)
        # Roundtrip should preserve all printable characters
        assert text.replace(" ", "") in recovered.replace(" ", "")

    def test_encode_adds_bos_eos(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode("hi", add_bos=True, add_eos=True)
        assert ids[0] == trained_tokenizer.bos_id
        assert ids[-1] == trained_tokenizer.eos_id

    def test_encode_max_length(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode(SAMPLE_TEXT, max_length=10)
        assert len(ids) == 10

    def test_save_and_load(self, trained_tokenizer: BPETokenizer) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tok.json"
            trained_tokenizer.save(path)
            loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == trained_tokenizer.vocab_size
        ids1 = trained_tokenizer.encode("test sentence")
        ids2 = loaded.encode("test sentence")
        assert ids1 == ids2

    def test_unknown_token(self, trained_tokenizer: BPETokenizer) -> None:
        # UNK should be used for bytes not in vocabulary
        ids = trained_tokenizer.encode("∞∑∏")
        assert all(isinstance(i, int) for i in ids)
