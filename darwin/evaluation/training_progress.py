"""Training-progress estimates for the Darwin UI.

These estimates are intentionally conservative. They combine local signals
such as checkpoints, evaluation loss, knowledge entries, and improvement
cycles. They are not a replacement for formal benchmark results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_training_progress(
    cfg: Any,
    model: Any,
    tokenizer: Any,
    knowledge_base: Any,
    metrics: Any,
) -> dict[str, Any]:
    params = model.num_parameters() if hasattr(model, "num_parameters") else 0
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    kb_count = knowledge_base.count()
    summary = metrics.summary() if metrics is not None else {}
    total_cycles = int(summary.get("total_cycles", 0) or 0)
    best_eval_loss = summary.get("best_eval_loss")
    latest_eval_loss = summary.get("latest_eval_loss")

    checkpoint_dir = Path(getattr(cfg.training, "output_dir", "data/checkpoints"))
    checkpoint_count = _checkpoint_count(checkpoint_dir)
    has_language_checkpoint = _has_language_training_checkpoint(checkpoint_dir)
    has_eval_loss = isinstance(best_eval_loss, (int, float))

    signals = {
        "parameters": params,
        "vocab_size": vocab_size,
        "knowledge_entries": kb_count,
        "improvement_cycles": total_cycles,
        "checkpoints": checkpoint_count,
        "has_language_training_checkpoint": has_language_checkpoint,
        "latest_eval_loss": latest_eval_loss,
        "best_eval_loss": best_eval_loss,
    }

    milestones = [
        _milestone(
            "foundation",
            "Foundation model boots",
            "Tokenizer and custom transformer can load and run.",
            1.0 if params > 0 and vocab_size > 0 else 0.0,
        ),
        _milestone(
            "sentence_completion",
            "Can complete a sentence",
            "Needs real language-training checkpoints, not only random starter weights.",
            _sentence_score(has_language_checkpoint, has_eval_loss, best_eval_loss),
        ),
        _milestone(
            "factual_answers",
            "Can provide factual data",
            "Improves when references are added and eval loss is tracked.",
            _facts_score(kb_count, has_eval_loss, best_eval_loss),
        ),
        _milestone(
            "conversation",
            "Can have a conversation",
            "The chat shell works; the raw model still needs assistant-style training.",
            _conversation_score(has_language_checkpoint, has_eval_loss, total_cycles),
        ),
        _milestone(
            "functional_code",
            "Can create functional code",
            "Code agents can inspect and propose changes; generation needs stronger training.",
            _code_score(total_cycles, has_language_checkpoint),
        ),
        _milestone(
            "self_improvement",
            "Can improve itself",
            "Tracks cycles, proposals, reviews, and checkpoints.",
            _self_improvement_score(total_cycles, checkpoint_count),
        ),
        _milestone(
            "image_generation",
            "Can generate images",
            "Requires an image model or tool path; not trained into this language model yet.",
            0.0,
        ),
        _milestone(
            "multimodal_video",
            "Can reason over images/video",
            "Requires multimodal ingestion and evaluation beyond the text transformer.",
            0.0,
        ),
    ]

    overall = round(
        sum(item["score"] * item["weight"] for item in milestones)
        / sum(item["weight"] for item in milestones)
    )
    return {
        "overall_percent": overall,
        "level": _level(overall),
        "summary": _summary(overall, has_language_checkpoint, kb_count),
        "signals": signals,
        "milestones": milestones,
    }


def _milestone(
    milestone_id: str,
    label: str,
    description: str,
    score: float,
    weight: float = 1.0,
) -> dict[str, Any]:
    percent = int(round(max(0.0, min(score, 1.0)) * 100))
    if percent >= 80:
        status = "ready"
    elif percent >= 35:
        status = "learning"
    elif percent > 0:
        status = "started"
    else:
        status = "locked"
    return {
        "id": milestone_id,
        "label": label,
        "description": description,
        "score": percent,
        "status": status,
        "weight": weight,
    }


def _sentence_score(
    has_language_checkpoint: bool,
    has_eval_loss: bool,
    best_eval_loss: float | None,
) -> float:
    if has_eval_loss and best_eval_loss is not None:
        return _loss_score(best_eval_loss, good=2.5, poor=7.5)
    return 0.45 if has_language_checkpoint else 0.12


def _facts_score(kb_count: int, has_eval_loss: bool, best_eval_loss: float | None) -> float:
    reference_score = min(kb_count / 10, 1.0) * 0.55
    model_score = _loss_score(best_eval_loss, good=2.0, poor=7.0) * 0.35 if has_eval_loss else 0.05
    return min(1.0, reference_score + model_score)


def _conversation_score(
    has_language_checkpoint: bool,
    has_eval_loss: bool,
    total_cycles: int,
) -> float:
    shell_score = 0.30
    cycle_score = min(total_cycles / 10, 1.0) * 0.15
    training_score = 0.35 if has_language_checkpoint else 0.0
    eval_score = 0.20 if has_eval_loss else 0.0
    return min(1.0, shell_score + cycle_score + training_score + eval_score)


def _code_score(total_cycles: int, has_language_checkpoint: bool) -> float:
    agent_score = min(total_cycles / 8, 1.0) * 0.45
    model_score = 0.25 if has_language_checkpoint else 0.0
    return min(1.0, agent_score + model_score)


def _self_improvement_score(total_cycles: int, checkpoint_count: int) -> float:
    cycle_score = min(total_cycles / 10, 1.0) * 0.55
    checkpoint_score = min(checkpoint_count / 5, 1.0) * 0.35
    return min(1.0, cycle_score + checkpoint_score)


def _loss_score(loss: float | None, good: float, poor: float) -> float:
    if loss is None:
        return 0.0
    if loss <= good:
        return 1.0
    if loss >= poor:
        return 0.1
    return 1.0 - ((loss - good) / (poor - good)) * 0.9


def _checkpoint_count(checkpoint_dir: Path) -> int:
    registry = checkpoint_dir / "registry.json"
    if registry.exists():
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
        except (json.JSONDecodeError, OSError):
            pass
    if not checkpoint_dir.exists():
        return 0
    return sum(1 for child in checkpoint_dir.iterdir() if child.is_dir())


def _has_language_training_checkpoint(checkpoint_dir: Path) -> bool:
    trainer_markers = [
        checkpoint_dir / "final" / "checkpoint.pt",
        checkpoint_dir / "best" / "checkpoint.pt",
    ]
    if any(path.exists() for path in trainer_markers):
        return True
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("step_*/checkpoint.pt")):
        return True
    return False


def _level(percent: int) -> str:
    if percent >= 80:
        return "Advanced"
    if percent >= 55:
        return "Capable"
    if percent >= 30:
        return "Early learning"
    return "Bootstrapping"


def _summary(percent: int, has_language_checkpoint: bool, kb_count: int) -> str:
    if not has_language_checkpoint:
        return (
            f"{percent}% trained: Darwin's app shell is working, but the raw language model "
            "still needs real training/fine-tuning checkpoints."
        )
    if kb_count == 0:
        return f"{percent}% trained: language checkpoints exist; add references for grounded facts."
    return f"{percent}% trained: language checkpoints and local references are available."
