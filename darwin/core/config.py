"""Configuration dataclasses for Darwin's foundation model and training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Hyper-parameters for the transformer language model."""

    # Vocabulary & sequence
    vocab_size: int = 32_000
    max_seq_len: int = 2_048
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Architecture
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 8          # grouped-query attention when < n_heads
    d_model: int = 512
    d_ff: int = 2_048             # feed-forward hidden size
    dropout: float = 0.1
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    rope_base: float = 10_000.0   # rotary position embedding base

    # Quantization / memory
    use_flash_attention: bool = False  # requires flash-attn package
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    """Training hyper-parameters."""

    # Optimisation
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2_000
    max_steps: int = 100_000
    lr_decay_steps: Optional[int] = None      # defaults to max_steps
    min_lr_ratio: float = 0.1

    # Batching
    batch_size: int = 32
    grad_accum_steps: int = 1
    micro_batch_size: Optional[int] = None    # defaults to batch_size

    # Checkpointing
    save_every: int = 1_000
    eval_every: int = 500
    keep_last_n_checkpoints: int = 3

    # Logging
    log_every: int = 10
    wandb_project: Optional[str] = None

    # Paths
    output_dir: str = "data/checkpoints"
    resume_from: Optional[str] = None

    def __post_init__(self) -> None:
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.max_steps
        if self.micro_batch_size is None:
            self.micro_batch_size = self.batch_size


@dataclass
class DarwinConfig:
    """Top-level configuration for the whole Darwin system."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Self-improvement
    improvement_interval_seconds: int = 3_600   # run the improvement loop every hour
    max_parallel_agents: int = 4
    proposal_budget: int = 10                    # max proposals per improvement cycle
    validation_timeout_seconds: int = 300

    # Safety
    max_code_execution_time_seconds: int = 30
    allow_network_in_sandbox: bool = False
    require_human_review_above_risk: float = 0.7  # 0–1 risk score

    # Domains enabled
    enabled_domains: list[str] = field(
        default_factory=lambda: ["code", "art", "video", "prompting", "research"]
    )

    # Paths
    data_dir: str = "data"
    log_dir: str = "data/logs"
