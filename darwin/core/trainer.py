"""Training infrastructure for the Darwin foundation model.

Supports:
* Causal language-model pre-training on raw text
* Supervised fine-tuning (SFT)
* Cosine LR schedule with warmup
* Gradient clipping and accumulation
* Automatic checkpoint saving / resumption
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset

from darwin.core.config import TrainingConfig
from darwin.core.model import DarwinLM

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """Simple in-memory token-id dataset for SFT / small corpora."""

    def __init__(self, token_ids: List[int], seq_len: int) -> None:
        self.seq_len = seq_len
        # Build (input, label) pairs by sliding window
        self.examples: List[torch.Tensor] = []
        for start in range(0, len(token_ids) - seq_len, seq_len):
            chunk = torch.tensor(token_ids[start : start + seq_len + 1], dtype=torch.long)
            self.examples.append(chunk)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.examples[idx]
        return {
            "input_ids": chunk[:-1].clone(),
            "labels": chunk[1:].clone(),
        }


class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset from a text file."""

    def __init__(self, file_path: str | Path, tokenizer: Any, seq_len: int) -> None:
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[int] = []
        with self.file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                buffer.extend(self.tokenizer.encode(line.rstrip()))
                while len(buffer) >= self.seq_len + 1:
                    chunk = torch.tensor(buffer[: self.seq_len + 1], dtype=torch.long)
                    yield {
                        "input_ids": chunk[:-1].clone(),
                        "labels": chunk[1:].clone(),
                    }
                    buffer = buffer[self.seq_len:]


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------


def _cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine LR with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Manages the training loop for DarwinLM."""

    def __init__(
        self,
        model: DarwinLM,
        cfg: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.eval_dataset = eval_dataset
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.global_step = 0
        self.best_eval_loss = float("inf")

        # Optimizer
        decay_params = [p for n, p in model.named_parameters() if "norm" not in n and p.requires_grad]
        no_decay_params = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": cfg.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.micro_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        self._output_dir = Path(cfg.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Resume if requested
        if cfg.resume_from:
            self._load_checkpoint(cfg.resume_from)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        self.model.train()
        start_time = time.perf_counter()
        total_loss = 0.0
        accum_steps = 0

        while self.global_step < self.cfg.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.cfg.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Update LR
                lr = _cosine_lr(
                    self.global_step,
                    self.cfg.warmup_steps,
                    self.cfg.lr_decay_steps,
                    self.cfg.learning_rate,
                    self.cfg.learning_rate * self.cfg.min_lr_ratio,
                )
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                out = self.model(input_ids, labels=labels)
                loss: torch.Tensor = out["loss"] / self.cfg.grad_accum_steps
                loss.backward()
                total_loss += loss.item() * self.cfg.grad_accum_steps
                accum_steps += 1

                if accum_steps % self.cfg.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    accum_steps = 0

                    if self.global_step % self.cfg.log_every == 0:
                        elapsed = time.perf_counter() - start_time
                        avg_loss = total_loss / self.cfg.log_every
                        total_loss = 0.0
                        logger.info(
                            "train_step",
                            step=self.global_step,
                            loss=round(avg_loss, 4),
                            lr=round(lr, 7),
                            elapsed_s=round(elapsed, 1),
                        )

                    if self.global_step % self.cfg.eval_every == 0 and self.eval_dataset:
                        eval_loss = self.evaluate()
                        self.model.train()
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self._save_checkpoint("best")

                    if self.global_step % self.cfg.save_every == 0:
                        self._save_checkpoint(f"step_{self.global_step:07d}")
                        self._cleanup_old_checkpoints()

        self._save_checkpoint("final")
        logger.info("training_complete", total_steps=self.global_step)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        loader = DataLoader(
            self.eval_dataset,
            batch_size=self.cfg.micro_batch_size,
            shuffle=False,
            num_workers=0,
        )
        total = 0.0
        n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            out = self.model(input_ids, labels=labels)
            total += out["loss"].item()
            n += 1
        avg = total / max(n, 1)
        logger.info("eval", loss=round(avg, 4), perplexity=round(math.exp(min(avg, 20)), 2))
        return avg

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, tag: str) -> Path:
        ckpt_dir = self._output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "best_eval_loss": self.best_eval_loss,
            },
            ckpt_dir / "checkpoint.pt",
        )
        logger.info("checkpoint_saved", tag=tag, step=self.global_step)
        return ckpt_dir

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        logger.info("checkpoint_loaded", path=path, step=self.global_step)

    def _cleanup_old_checkpoints(self) -> None:
        keep = self.cfg.keep_last_n_checkpoints
        all_ckpts = sorted(
            [d for d in self._output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[-1]),
        )
        for old in all_ckpts[:-keep]:
            import shutil

            shutil.rmtree(old)
