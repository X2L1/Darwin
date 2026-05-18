"""Byte-Pair Encoding (BPE) tokenizer with special token support.

This is a self-contained, pure-Python BPE tokenizer that does not depend on
any third-party tokenizer library.  It can be trained from scratch on a corpus
and serialised to / deserialised from JSON.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

# Pre-tokenisation regex (GPT-2 style): splits on contractions, punctuation,
# whitespace-prefixed words, and runs of digits/letters.
_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+"""
    r"""| ?[^\s\w]+|\s+""",
    re.UNICODE,
)


def _bytes_to_unicode() -> Dict[int, str]:
    """Map raw bytes 0-255 to printable Unicode characters."""
    bs: list[int] = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs: list[int] = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENCODER: Dict[int, str] = _bytes_to_unicode()
_BYTE_DECODER: Dict[str, int] = {v: k for k, v in _BYTE_ENCODER.items()}


def _get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


# ---------------------------------------------------------------------------
# Tokenizer class
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Byte-Pair Encoding tokenizer.

    Usage
    -----
    Train from scratch::

        tok = BPETokenizer()
        tok.train(corpus_text, vocab_size=32_000)

    Encode / decode::

        ids = tok.encode("Hello, world!")
        text = tok.decode(ids)

    Save / load::

        tok.save("tokenizer.json")
        tok2 = BPETokenizer.load("tokenizer.json")
    """

    # Special tokens
    PAD = "<|pad|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    UNK = "<|unk|>"
    SEP = "<|sep|>"
    MASK = "<|mask|>"

    SPECIAL_TOKENS: List[str] = [PAD, BOS, EOS, UNK, SEP, MASK]

    def __init__(self) -> None:
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        self._cache: Dict[str, Tuple[str, ...]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int = 32_000) -> None:
        """Train BPE on *text*, building a vocabulary of *vocab_size* tokens."""
        # Start with byte-level vocabulary (256 tokens) + specials
        base_vocab: Dict[str, int] = {}
        for special in self.SPECIAL_TOKENS:
            base_vocab[special] = len(base_vocab)
        for ch in _BYTE_ENCODER.values():
            if ch not in base_vocab:
                base_vocab[ch] = len(base_vocab)

        # Build initial word frequencies
        word_freqs: Dict[Tuple[str, ...], int] = defaultdict(int)
        for token in _PAT.findall(text):
            encoded = "".join(_BYTE_ENCODER[b] for b in token.encode("utf-8"))
            word_freqs[tuple(encoded)] += 1

        merges: List[Tuple[str, str]] = []
        vocab = dict(base_vocab)
        n_merges = vocab_size - len(vocab)

        for _ in range(n_merges):
            pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
            for word, freq in word_freqs.items():
                for pair in _get_pairs(word):
                    pair_freqs[pair] += freq
            if not pair_freqs:
                break
            best = max(pair_freqs, key=pair_freqs.__getitem__)
            merges.append(best)
            merged = best[0] + best[1]
            if merged not in vocab:
                vocab[merged] = len(vocab)
            # Apply merge to corpus
            new_freqs: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_freqs.items():
                new_word = _apply_merge(word, best)
                new_freqs[new_word] = new_freqs.get(new_word, 0) + freq
            word_freqs = new_freqs

        self.encoder = vocab
        self.decoder = {v: k for k, v in vocab.items()}
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self._cache = {}

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _bpe(self, token: str) -> Tuple[str, ...]:
        if token in self._cache:
            return self._cache[token]
        word: Tuple[str, ...] = tuple(token)
        while True:
            pairs = _get_pairs(word)
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            word = _apply_merge(word, best)
        self._cache[token] = word
        return word

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode *text* to a list of token ids."""
        ids: List[int] = []
        if add_bos:
            ids.append(self.encoder[self.BOS])
        for token in _PAT.findall(text):
            encoded = "".join(_BYTE_ENCODER[b] for b in token.encode("utf-8"))
            for bpe_token in self._bpe(encoded):
                ids.append(self.encoder.get(bpe_token, self.encoder[self.UNK]))
        if add_eos:
            ids.append(self.encoder[self.EOS])
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token ids back to a string."""
        tokens: List[str] = []
        specials_set = set(self.SPECIAL_TOKENS)
        for i in ids:
            tok = self.decoder.get(i, self.UNK)
            if skip_special and tok in specials_set:
                continue
            tokens.append(tok)
        text_bytes = bytearray()
        for tok in tokens:
            for ch in tok:
                text_bytes.append(_BYTE_DECODER.get(ch, 0))
        return text_bytes.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "encoder": self.encoder,
            "bpe_ranks": [list(pair) for pair in self.bpe_ranks.keys()],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls()
        tok.encoder = {k: int(v) for k, v in data["encoder"].items()}
        tok.decoder = {v: k for k, v in tok.encoder.items()}
        tok.bpe_ranks = {
            (pair[0], pair[1]): i for i, pair in enumerate(data["bpe_ranks"])
        }
        return tok

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @property
    def pad_id(self) -> int:
        return self.encoder[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.encoder[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.encoder[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.encoder[self.UNK]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_merge(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    """Merge all occurrences of *pair* in *word* into a single token."""
    merged: List[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)
