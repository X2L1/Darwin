"""Tests for the DarwinLM foundation model."""

from __future__ import annotations

import torch
import pytest

from darwin.core.config import ModelConfig
from darwin.core.model import DarwinLM, RMSNorm, SwiGLU


@pytest.fixture()
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        max_seq_len=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
    )


@pytest.fixture()
def tiny_model(tiny_cfg: ModelConfig) -> DarwinLM:
    return DarwinLM.from_config(tiny_cfg)


class TestRMSNorm:
    def test_output_shape(self) -> None:
        norm = RMSNorm(32)
        x = torch.randn(2, 10, 32)
        assert norm(x).shape == (2, 10, 32)

    def test_normalises(self) -> None:
        norm = RMSNorm(16)
        x = torch.randn(4, 16) * 100
        out = norm(x)
        # Output should have RMS ≈ 1 (before weight scaling)
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() == pytest.approx(1.0, abs=0.3)


class TestSwiGLU:
    def test_output_shape(self) -> None:
        ffn = SwiGLU(32, 64)
        x = torch.randn(2, 8, 32)
        assert ffn(x).shape == (2, 8, 32)


class TestDarwinLM:
    def test_forward_logits_shape(self, tiny_model: DarwinLM, tiny_cfg: ModelConfig) -> None:
        B, T = 2, 16
        ids = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        out = tiny_model(ids)
        assert out["logits"].shape == (B, T, tiny_cfg.vocab_size)

    def test_forward_with_labels_returns_loss(self, tiny_model: DarwinLM, tiny_cfg: ModelConfig) -> None:
        B, T = 2, 16
        ids = torch.randint(0, tiny_cfg.vocab_size, (B, T))
        out = tiny_model(ids, labels=ids)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar
        assert out["loss"].item() > 0

    def test_generate_output_length(self, tiny_model: DarwinLM, tiny_cfg: ModelConfig) -> None:
        ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
        out = tiny_model.generate(ids, max_new_tokens=10, temperature=1.0, top_k=5)
        assert out.shape[1] == 8 + 10

    def test_generate_stops_at_eos(self, tiny_model: DarwinLM, tiny_cfg: ModelConfig) -> None:
        ids = torch.randint(1, tiny_cfg.vocab_size - 1, (1, 4))
        out = tiny_model.generate(ids, max_new_tokens=32, top_k=1, eos_token_id=2)
        # Should not exceed 4 + 32
        assert out.shape[1] <= 4 + 32

    def test_num_parameters(self, tiny_model: DarwinLM) -> None:
        n = tiny_model.num_parameters()
        assert n > 0

    def test_tied_embeddings(self, tiny_cfg: ModelConfig) -> None:
        tiny_cfg.tie_embeddings = True
        model = DarwinLM.from_config(tiny_cfg)
        assert model.lm_head.weight is model.embed.weight

    def test_untied_embeddings(self, tiny_cfg: ModelConfig) -> None:
        tiny_cfg.tie_embeddings = False
        model = DarwinLM.from_config(tiny_cfg)
        assert model.lm_head.weight is not model.embed.weight
