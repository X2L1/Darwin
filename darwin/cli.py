"""Darwin command-line interface.

Usage
-----
::

    darwin train           – pre-train or fine-tune the foundation model
    darwin run             – start the self-improvement loop
    darwin serve           – launch the REST API server
    darwin generate        – generate text from the local model
    darwin kb add <path>   – add a file/directory/video as primary reference
    darwin kb list         – list knowledge base entries
    darwin kb search <q>   – search the knowledge base
    darwin kb remove <id>  – remove a knowledge base entry
    darwin review list     – list proposals awaiting human review
    darwin review approve  – approve a pending proposal
    darwin review reject   – reject a pending proposal
    darwin status          – print system status
    darwin benchmark       – run all benchmarks

Everything runs locally.  No paid API, no cloud service, no key required.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_BOOTSTRAP_TOKENIZER_TEXT = (
    "Darwin is a self-improving AI assistant. It can chat with the user, "
    "search its local knowledge base, inspect its source code, run internal "
    "agent review cycles, and improve its own Python implementation safely. "
    "The custom language model is a PyTorch transformer defined in Darwin source code."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_system(data_dir: str = "data", small: bool = False):
    """Initialise and return the core Darwin objects."""
    import torch

    from darwin.core.config import DarwinConfig
    from darwin.core.model import DarwinLM
    from darwin.core.tokenizer import BPETokenizer
    from darwin.evaluation.metrics import MetricsCollector
    from darwin.improvement.loop import ImprovementLoop
    from darwin.knowledge.base import KnowledgeBase

    cfg = DarwinConfig()
    cfg.data_dir = data_dir
    cfg.log_dir = str(Path(data_dir) / "logs")
    cfg.training.output_dir = str(Path(data_dir) / "checkpoints")

    if small:
        # Tiny model for quick testing / demo
        cfg.model.n_layers = 2
        cfg.model.n_heads = 2
        cfg.model.n_kv_heads = 2
        cfg.model.d_model = 64
        cfg.model.d_ff = 256
        cfg.model.max_seq_len = 512

    tok_path = Path(data_dir) / "tokenizer.json"
    tokenizer = _load_or_bootstrap_tokenizer(BPETokenizer, tok_path)

    cfg.model.vocab_size = tokenizer.vocab_size
    model = DarwinLM.from_config(cfg.model)
    _load_latest_checkpoint(model, Path(cfg.training.output_dir), torch)

    kb = KnowledgeBase(store_dir=f"{data_dir}/knowledge")
    metrics = MetricsCollector(log_dir=cfg.log_dir)
    loop = ImprovementLoop(cfg, model, tokenizer, kb)
    return cfg, model, tokenizer, kb, metrics, loop


def _load_or_bootstrap_tokenizer(tokenizer_cls, tok_path: Path):
    if tok_path.exists() and tok_path.stat().st_size > 0:
        try:
            return tokenizer_cls.load(tok_path)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Rebuilding unreadable tokenizer %s: %s", tok_path, exc)

    tokenizer = tokenizer_cls()
    tokenizer.train(_BOOTSTRAP_TOKENIZER_TEXT, vocab_size=512)
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok_path)
    return tokenizer


def _load_latest_checkpoint(model, checkpoint_dir: Path, torch_module) -> Optional[Path]:
    """Load the newest compatible trainer/checkpoint-manager weights if present."""
    candidates: list[Path] = []
    for name in ("final", "best"):
        candidates.append(checkpoint_dir / name / "checkpoint.pt")
        candidates.append(checkpoint_dir / name / "model.pt")
    if checkpoint_dir.exists():
        candidates.extend(
            sorted(
                checkpoint_dir.glob("step_*/checkpoint.pt"),
                key=lambda p: p.parent.name,
                reverse=True,
            )
        )

    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = torch_module.load(path, map_location="cpu")
            state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
            model.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Skipping incompatible checkpoint %s: %s", path, exc)
            continue
        logging.getLogger(__name__).info("Loaded model checkpoint: %s", path)
        return path
    return None


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------


@click.group()
@click.version_option("0.1.0", prog_name="darwin")
def main() -> None:
    """Darwin – free, local, self-improving multi-agent AI."""


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@main.command()
@click.option("--data-file", required=True, type=click.Path(exists=True), help="Path to training text file.")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--steps", default=1000, show_default=True, type=int)
@click.option("--lr", default=3e-4, show_default=True, type=float)
@click.option("--batch-size", default=8, show_default=True, type=int)
@click.option("--small", is_flag=True, help="Use a tiny model for quick testing.")
def train(data_file: str, data_dir: str, steps: int, lr: float, batch_size: int, small: bool) -> None:
    """Pre-train or fine-tune the Darwin foundation model on a local text file."""
    import torch

    from darwin.core.config import DarwinConfig
    from darwin.core.model import DarwinLM
    from darwin.core.tokenizer import BPETokenizer
    from darwin.core.trainer import TextDataset, Trainer

    cfg = DarwinConfig()
    cfg.data_dir = data_dir
    cfg.log_dir = str(Path(data_dir) / "logs")
    cfg.training.output_dir = str(Path(data_dir) / "checkpoints")
    cfg.training.max_steps = steps
    cfg.training.learning_rate = lr
    cfg.training.batch_size = batch_size
    cfg.training.micro_batch_size = batch_size

    if small:
        cfg.model.n_layers = 2
        cfg.model.n_heads = 2
        cfg.model.n_kv_heads = 2
        cfg.model.d_model = 64
        cfg.model.d_ff = 256
        cfg.model.max_seq_len = 512

    click.echo(f"Loading training data from {data_file}…")
    text = Path(data_file).read_text(encoding="utf-8")

    click.echo("Training tokenizer…")
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=min(cfg.model.vocab_size, 8_000))
    tok_path = Path(data_dir) / "tokenizer.json"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok_path)
    click.echo(f"Tokenizer saved to {tok_path}")
    cfg.model.vocab_size = tokenizer.vocab_size

    click.echo(f"Tokenizing corpus ({len(text):,} chars)…")
    ids = tokenizer.encode(text)
    seq_len = cfg.model.max_seq_len

    if len(ids) < seq_len + 1:
        click.echo("Corpus too small.  Provide a larger text file.", err=True)
        sys.exit(1)

    dataset = TextDataset(ids, seq_len)
    split = max(1, int(len(dataset) * 0.9))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    eval_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    model = DarwinLM.from_config(cfg.model)
    click.echo(f"Model parameters: {model.num_parameters():,}")

    trainer = Trainer(model, cfg.training, train_ds, eval_ds)
    click.echo(f"Training for {steps} steps…")
    trainer.train()
    click.echo("Training complete.")


# ---------------------------------------------------------------------------
# run  (self-improvement loop)
# ---------------------------------------------------------------------------


@main.command("run")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--interval", default=3600, show_default=True, type=int,
              help="Seconds between improvement cycles.")
@click.option("--small", is_flag=True)
def run_loop(data_dir: str, interval: int, small: bool) -> None:
    """Start the continuous self-improvement loop (runs until Ctrl-C)."""
    cfg, model, tokenizer, kb, metrics, loop = _load_system(data_dir, small=small)
    cfg.improvement_interval_seconds = interval
    click.echo("Darwin self-improvement loop running.  Press Ctrl-C to stop.")
    loop.run_forever()


# ---------------------------------------------------------------------------
# serve  (REST API)
# ---------------------------------------------------------------------------


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--small", is_flag=True)
def serve(host: str, port: int, data_dir: str, small: bool) -> None:
    """Launch the Darwin REST API server (free, local, no API key)."""
    import uvicorn

    from darwin.api import create_app

    cfg, model, tokenizer, kb, metrics, loop = _load_system(data_dir, small=small)
    application = create_app(cfg, model, tokenizer, kb, loop, metrics)
    click.echo(f"Darwin API running at http://{host}:{port}")
    click.echo(f"Docs: http://{host}:{port}/docs")
    uvicorn.run(application, host=host, port=port, log_level="info")


# ---------------------------------------------------------------------------
# chat / generate
# ---------------------------------------------------------------------------


@main.command()
@click.argument("message", nargs=-1, required=True)
@click.option("--max-tokens", default=160, show_default=True, type=int)
@click.option("--temperature", default=0.8, show_default=True, type=float)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--small", is_flag=True)
@click.option("--json-output", is_flag=True, help="Print the full structured chat response.")
@click.option("--no-improve", is_flag=True, help="Do not run self-improvement cycles from chat.")
def chat(
    message: tuple[str, ...],
    max_tokens: int,
    temperature: float,
    data_dir: str,
    small: bool,
    json_output: bool,
    no_improve: bool,
) -> None:
    """Talk to Darwin as one unified assistant."""
    from darwin.orchestrator.brain import UnifiedDarwinBrain

    cfg, model, tokenizer, kb, metrics, loop = _load_system(data_dir, small=small)
    brain = UnifiedDarwinBrain(cfg, model, tokenizer, kb, loop, metrics)
    response = brain.chat(
        " ".join(message),
        max_new_tokens=max_tokens,
        temperature=temperature,
        run_improvements=not no_improve,
    )
    if json_output:
        click.echo(json.dumps(response.to_dict(), indent=2))
    else:
        click.echo(response.message)


@main.command()
@click.argument("prompt")
@click.option("--max-tokens", default=128, show_default=True, type=int)
@click.option("--temperature", default=0.8, show_default=True, type=float)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--small", is_flag=True)
def generate(prompt: str, max_tokens: int, temperature: float, data_dir: str, small: bool) -> None:
    """Generate text using the local Darwin model (no API key needed)."""
    import torch

    _, model, tokenizer, *_ = _load_system(data_dir, small=small)
    ids = tokenizer.encode(prompt, add_bos=True, max_length=max(1, model.cfg.max_seq_len - max_tokens))
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_id,
        )
    new_text = tokenizer.decode(out[0, len(ids):].tolist())
    click.echo(new_text)


# ---------------------------------------------------------------------------
# kb  (knowledge base sub-commands)
# ---------------------------------------------------------------------------


@main.group("kb")
def kb_group() -> None:
    """Manage primary reference files, videos, and texts."""


@kb_group.command("add")
@click.argument("path", type=click.Path())
@click.option("--data-dir", default="data", show_default=True)
@click.option("--tag", multiple=True, help="One or more tags (can be repeated).")
@click.option("--language", default="en", show_default=True)
@click.option("--not-primary", is_flag=True, help="Do not mark as primary reference.")
def kb_add(path: str, data_dir: str, tag: tuple, language: str, not_primary: bool) -> None:
    """Ingest a file, directory, or video as a primary reference.

    Supported formats: .txt .md .pdf .html .srt .vtt .json .mp4 .mkv
    .avi .mov .mp3 .wav .flac and more.  All processing is local and free.

    For video/audio files, install openai-whisper for free local transcription::

        pip install openai-whisper
    """
    from darwin.knowledge.base import KnowledgeBase
    from darwin.knowledge.ingestor import Ingestor

    kb = KnowledgeBase(store_dir=f"{data_dir}/knowledge")
    ingestor = Ingestor()
    tags = list(tag)
    is_primary = not not_primary

    try:
        entries = ingestor.ingest(path, tags=tags, language=language, is_primary_reference=is_primary)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    for entry in entries:
        kb.add_entry(entry)
        click.echo(f"  ✓ {entry.title} [{entry.media_type}] ({entry.word_count:,} words) → {entry.entry_id}")

    click.echo(f"\nAdded {len(entries)} entry/entries to the knowledge base.")


@kb_group.command("list")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--media-type", default=None)
@click.option("--primary-only", is_flag=True)
def kb_list(data_dir: str, media_type: Optional[str], primary_only: bool) -> None:
    """List all entries in the knowledge base."""
    from darwin.knowledge.base import KnowledgeBase

    kb = KnowledgeBase(store_dir=f"{data_dir}/knowledge")
    entries = kb.list_entries(media_type=media_type, primary_only=primary_only)
    if not entries:
        click.echo("Knowledge base is empty.")
        return
    for e in entries:
        primary_flag = "★" if e.get("is_primary_reference") else " "
        click.echo(
            f"  {primary_flag} [{e['media_type']:8s}] {e['title'][:50]:<50s}"
            f"  {e['word_count']:>8,}w  {e['entry_id'][:8]}"
        )
    click.echo(f"\nTotal: {len(entries)} entries")


@kb_group.command("search")
@click.argument("query")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--top-k", default=5, show_default=True, type=int)
def kb_search(query: str, data_dir: str, top_k: int) -> None:
    """Search the knowledge base using local TF-IDF (no external service)."""
    from darwin.knowledge.base import KnowledgeBase

    kb = KnowledgeBase(store_dir=f"{data_dir}/knowledge")
    results = kb.search(query, top_k=top_k)
    if not results:
        click.echo("No results found.")
        return
    for i, r in enumerate(results, 1):
        click.echo(f"\n[{i}] {r['title']}  (score: {r['score']:.3f})")
        click.echo(f"    {r['snippet'][:200]}")


@kb_group.command("remove")
@click.argument("entry_id")
@click.option("--data-dir", default="data", show_default=True)
def kb_remove(entry_id: str, data_dir: str) -> None:
    """Remove a knowledge base entry by its ID."""
    from darwin.knowledge.base import KnowledgeBase

    kb = KnowledgeBase(store_dir=f"{data_dir}/knowledge")
    removed = kb.remove_entry(entry_id)
    if removed:
        click.echo(f"Removed entry {entry_id}.")
    else:
        click.echo(f"Entry {entry_id!r} not found.", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# review  (human review workflow)
# ---------------------------------------------------------------------------


@main.group("review")
def review_group() -> None:
    """Human review workflow for high-risk proposals."""


@review_group.command("list")
@click.option("--data-dir", default="data", show_default=True)
def review_list(data_dir: str) -> None:
    """List proposals awaiting human review."""
    from darwin.governance.review_gate import ReviewGate

    gate = ReviewGate(review_store=f"{data_dir}/reviews")
    pending = gate.list_pending()
    if not pending:
        click.echo("No proposals pending review.")
        return
    for p in pending:
        click.echo(
            f"\n  ID:     {p['proposal_id']}\n"
            f"  Domain: {p['domain']}\n"
            f"  Title:  {p['title']}\n"
            f"  Risk:   {p['risk_score']:.2f}\n"
            f"  Desc:   {p['description'][:100]}"
        )


@review_group.command("approve")
@click.argument("proposal_id")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--notes", default="")
def review_approve(proposal_id: str, data_dir: str, notes: str) -> None:
    """Approve a pending proposal."""
    from darwin.governance.review_gate import ReviewGate

    gate = ReviewGate(review_store=f"{data_dir}/reviews")
    ok = gate.resolve(proposal_id, approved=True, reviewer="human-cli", notes=notes)
    click.echo("Approved." if ok else f"Proposal {proposal_id!r} not found.")


@review_group.command("reject")
@click.argument("proposal_id")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--notes", default="")
def review_reject(proposal_id: str, data_dir: str, notes: str) -> None:
    """Reject a pending proposal."""
    from darwin.governance.review_gate import ReviewGate

    gate = ReviewGate(review_store=f"{data_dir}/reviews")
    ok = gate.resolve(proposal_id, approved=False, reviewer="human-cli", notes=notes)
    click.echo("Rejected." if ok else f"Proposal {proposal_id!r} not found.")


# ---------------------------------------------------------------------------
# status / benchmark
# ---------------------------------------------------------------------------


@main.command()
@click.option("--data-dir", default="data", show_default=True)
@click.option("--small", is_flag=True)
def status(data_dir: str, small: bool) -> None:
    """Print system status and metrics summary."""
    cfg, model, tokenizer, kb, metrics, _ = _load_system(data_dir, small=small)
    click.echo(f"Model parameters : {model.num_parameters():,}")
    click.echo(f"KB entries       : {kb.count()}")
    click.echo(f"Metrics summary  : {json.dumps(metrics.summary(), indent=2)}")


@main.command()
@click.option("--data-dir", default="data", show_default=True)
@click.option("--small", is_flag=True)
def benchmark(data_dir: str, small: bool) -> None:
    """Run all local benchmarks and print results."""
    from darwin.evaluation.benchmarks import BenchmarkSuite

    _, model, tokenizer, *_ = _load_system(data_dir, small=small)
    suite = BenchmarkSuite.default(model, tokenizer)
    results = suite.run_all()
    click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
