"""Custom Darwin LLM runtime.

This module is the public inference wrapper for Darwin's own transformer
implementation. It does not call Ollama or any hosted API, so ordinary local
use is not subject to provider rate limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass
class LLMGeneration:
    text: str
    used_fallback: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class DarwinLLMEngine:
    """Inference wrapper around the custom PyTorch DarwinLM."""

    model_name = "darwin-custom-transformer"

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @property
    def source_files(self) -> list[str]:
        root = Path(__file__).resolve().parents[2]
        return [
            str(root / "darwin" / "core" / "model.py"),
            str(root / "darwin" / "core" / "tokenizer.py"),
            str(root / "darwin" / "core" / "trainer.py"),
            str(Path(__file__).resolve()),
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "provider": "local-custom",
            "rate_limited_by_provider": False,
            "parameters": self.model.num_parameters() if hasattr(self.model, "num_parameters") else None,
            "source_files": self.source_files,
        }

    @torch.no_grad()
    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        fallback: Optional[str] = None,
    ) -> LLMGeneration:
        if not getattr(self.tokenizer, "encoder", None):
            return LLMGeneration(
                text=fallback or self._fallback(prompt),
                used_fallback=True,
            )

        max_prompt_len = max(1, self.model.cfg.max_seq_len - max_new_tokens)
        ids = self.tokenizer.encode(prompt, add_bos=True, max_length=max_prompt_len)
        if not ids:
            return LLMGeneration(text=fallback or self._fallback(prompt), used_fallback=True)

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        self.model.eval()
        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=getattr(self.tokenizer, "eos_id", None),
        )

        new_tokens = out_ids[0, len(ids):].tolist()
        text = self.tokenizer.decode(new_tokens).strip()
        if not _looks_useful(text):
            return LLMGeneration(
                text=fallback or self._fallback(prompt),
                used_fallback=True,
                prompt_tokens=len(ids),
                completion_tokens=len(new_tokens),
            )

        return LLMGeneration(
            text=text,
            used_fallback=False,
            prompt_tokens=len(ids),
            completion_tokens=len(new_tokens),
        )

    def _fallback(self, prompt: str) -> str:
        return (
            "I am Darwin, running on the custom local transformer. "
            "My language weights may need more training, but the system can still inspect "
            "its source, use the knowledge base, and run internal improvement cycles."
        )


def _looks_useful(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    replacement_ratio = stripped.count("\ufffd") / max(len(stripped), 1)
    if replacement_ratio > 0.05:
        return False
    printable_ratio = sum(ch.isprintable() for ch in stripped) / max(len(stripped), 1)
    return printable_ratio > 0.9
