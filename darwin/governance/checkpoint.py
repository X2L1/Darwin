"""Versioned checkpoint manager for the Darwin self-improvement system.

Checkpoints are stored locally as JSON + PyTorch state-dict files.
No cloud storage, no paid service.

Layout::

    data/checkpoints/
        registry.json        ← metadata for all checkpoints
        <tag>/
            model.pt         ← model state dict
            meta.json        ← version metadata
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from darwin.core.model import DarwinLM

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"


@dataclass
class CheckpointMeta:
    tag: str
    step: int = 0
    cycle: int = 0
    eval_loss: Optional[float] = None
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    extra: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Saves, loads, and manages versioned model checkpoints locally.

    Usage::

        mgr = CheckpointManager()
        mgr.save(model, tag="after_cycle_3", step=3000, eval_loss=2.4)
        mgr.restore(model, tag="after_cycle_3")
        mgr.rollback(model)   # restore the previous checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "data/checkpoints",
        keep_last_n: int = 5,
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._dir / _REGISTRY_FILE
        self._registry: List[Dict[str, Any]] = self._load_registry()
        self.keep_last_n = keep_last_n

    # ------------------------------------------------------------------
    # Save / restore
    # ------------------------------------------------------------------

    def save(
        self,
        model: DarwinLM,
        tag: str,
        step: int = 0,
        cycle: int = 0,
        eval_loss: Optional[float] = None,
        description: str = "",
    ) -> Path:
        """Save *model* weights under *tag* and update the registry."""
        ckpt_dir = self._dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_dir / "model.pt")

        meta = CheckpointMeta(
            tag=tag, step=step, cycle=cycle, eval_loss=eval_loss, description=description
        )
        (ckpt_dir / "meta.json").write_text(
            json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Update registry (newest last)
        self._registry = [r for r in self._registry if r["tag"] != tag]
        self._registry.append(asdict(meta))
        self._save_registry()

        logger.info("Checkpoint saved: %s (step=%d, loss=%s)", tag, step, eval_loss)
        self._prune_old()
        return ckpt_dir

    def restore(self, model: DarwinLM, tag: str) -> bool:
        """Load weights from checkpoint *tag* into *model*.  Returns True on success."""
        path = self._dir / tag / "model.pt"
        if not path.exists():
            logger.warning("Checkpoint not found: %s", tag)
            return False
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info("Checkpoint restored: %s", tag)
        return True

    def rollback(self, model: DarwinLM) -> bool:
        """Restore the second-most-recent checkpoint (undo last save)."""
        if len(self._registry) < 2:
            logger.warning("No previous checkpoint to roll back to.")
            return False
        previous = self._registry[-2]
        return self.restore(model, previous["tag"])

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return list(reversed(self._registry))

    def latest_tag(self) -> Optional[str]:
        if not self._registry:
            return None
        return self._registry[-1]["tag"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prune_old(self) -> None:
        """Remove oldest checkpoints beyond keep_last_n (excluding 'best')."""
        prunable = [r for r in self._registry if r["tag"] not in ("best", "final")]
        to_remove = prunable[: max(0, len(prunable) - self.keep_last_n)]
        for meta in to_remove:
            ckpt_dir = self._dir / meta["tag"]
            if ckpt_dir.exists():
                shutil.rmtree(ckpt_dir)
                logger.debug("Pruned old checkpoint: %s", meta["tag"])
        self._registry = [
            r for r in self._registry if r["tag"] not in {m["tag"] for m in to_remove}
        ]
        self._save_registry()

    def _load_registry(self) -> List[Dict[str, Any]]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save_registry(self) -> None:
        self._registry_path.write_text(
            json.dumps(self._registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
