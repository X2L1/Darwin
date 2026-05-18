"""Darwin Foundation Model – GPT-style causal language model.

Architecture highlights
-----------------------
* Rotary Position Embeddings (RoPE)
* Grouped-Query Attention (GQA)
* SwiGLU feed-forward blocks
* RMSNorm (instead of LayerNorm)
* Pre-norm residual connections
* Optional Flash Attention

The model is designed to be trained from scratch and fine-tuned by the
self-improvement loop.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from darwin.core.config import ModelConfig


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------


def _precompute_freqs_cis(head_dim: int, seq_len: int, base: float = 10_000.0) -> torch.Tensor:
    """Precompute cos/sin tables for RoPE. Returns tensor [seq_len, head_dim/2, 2]."""
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [seq_len, half]
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # [seq_len, half, 2]


def _apply_rotary(x: torch.Tensor, freqs: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Apply RoPE to query or key tensor.

    Args:
        x:      [B, H, T, head_dim]
    freqs:  [max_seq_len, head_dim/2, 2]  (cos/sin pairs)
    offset: starting position for cached generation

    Returns tensor of same shape as *x*.
    """
    B, H, T, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos = freqs[offset: offset + T, :, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, T, half]
    sin = freqs[offset: offset + T, :, 1].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """Multi-head / Grouped-Query causal self-attention with RoPE."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.groups = cfg.n_heads // cfg.n_kv_heads

        # Projections
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.attention_dropout)
        self.use_flash = cfg.use_flash_attention

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_kv[0].shape[2] if past_kv is not None else 0

        # Apply RoPE at the absolute token positions, including cached context.
        q = _apply_rotary(q, freqs, offset=past_len)
        k = _apply_rotary(k, freqs, offset=past_len)

        # Append past key/values (for inference caching)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        # Expand KV heads to match Q heads (GQA)
        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        seq_len_full = k.shape[2]

        if self.use_flash:
            try:
                from flash_attn import flash_attn_func  # type: ignore[import]

                out = flash_attn_func(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    causal=True,
                ).transpose(1, 2)
            except ImportError:
                out = self._sdp_attn(q, k, v, mask, seq_len_full, T)
        else:
            out = self._sdp_attn(q, k, v, mask, seq_len_full, T)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out), new_kv

    def _sdp_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_len: int,
        q_len: int,
    ) -> torch.Tensor:
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask
        past_len = kv_len - q_len
        causal = torch.ones(q_len, kv_len, device=q.device, dtype=torch.bool).tril(diagonal=past_len)
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)


# ---------------------------------------------------------------------------
# Feed-forward (SwiGLU)
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block: FFN(x) = SiLU(xW1) * (xW3) @ W2."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv = self.attn(self.norm1(x), freqs, mask, past_kv)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, new_kv


# ---------------------------------------------------------------------------
# Foundation model
# ---------------------------------------------------------------------------


class DarwinLM(nn.Module):
    """Darwin's causal language model.

    Can be used both for next-token prediction (training) and for text
    generation (inference) with optional KV-cache.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        if cfg.tie_embeddings:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Precomputed RoPE frequencies – registered as buffer so they move with .to(device)
        freqs = _precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("freqs_cis", freqs)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[list[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, T] integer token ids.
            labels:    [B, T] integer token ids for teacher-forcing loss.
                       Tokens with id -100 are ignored.
            mask:      Optional attention bias [B, 1, T, T].
            past_kvs:  List of (k, v) tuples from previous forward calls.

        Returns:
            Dict with keys:
              * "logits"  – [B, T, vocab_size]
              * "loss"    – scalar (only when labels are provided)
              * "past_kvs" – updated KV cache
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)

        freqs = self.freqs_cis  # type: ignore[attr-defined]
        new_past_kvs: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, freqs, mask, pkv)
            new_past_kvs.append(new_kv)

        x = self.norm(x)
        logits: torch.Tensor = self.lm_head(x)

        result: dict[str, torch.Tensor] = {"logits": logits, "past_kvs": new_past_kvs}

        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            result["loss"] = loss

        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressive token generation.

        Args:
            input_ids:        [B, T] prompt token ids.
            max_new_tokens:   Maximum number of new tokens to generate.
            temperature:      Sampling temperature (1.0 = unscaled).
            top_k:            Keep only top-k logits (0 = disabled).
            top_p:            Nucleus sampling probability threshold.
            repetition_penalty: Penalise repeated tokens.
            eos_token_id:     Stop generation when this token is emitted.

        Returns:
            [B, T + max_new_tokens] token ids.
        """
        max_new_tokens = min(max_new_tokens, max(0, self.cfg.max_seq_len - input_ids.shape[1]))
        if max_new_tokens == 0:
            return input_ids
        past_kvs = None
        generated = input_ids

        for _ in range(max_new_tokens):
            # Only feed the last token if we have a KV cache
            model_input = generated if past_kvs is None else generated[:, -1:]
            out = self.forward(model_input, past_kvs=past_kvs)
            past_kvs = out["past_kvs"]
            logits = out["logits"][:, -1, :]  # [B, vocab]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(generated.shape[0]):
                    for tok in set(generated[b].tolist()):
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty

            # Temperature
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                remove = cum_probs - probs > top_p
                sorted_logits.masked_fill_(remove, float("-inf"))
                logits.scatter_(-1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only or p.requires_grad)
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "DarwinLM":
        return cls(cfg)
