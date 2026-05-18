#!/usr/bin/env python3
"""Bootstrap script – train a minimal Darwin model on a sample corpus.

Run once after cloning to verify the installation works::

    python scripts/bootstrap.py

Everything runs locally.  No API key, no paid service, no internet needed.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_TEXT = """\
The quick brown fox jumps over the lazy dog.
Darwin is a self-improving multi-agent AI system.
It runs entirely on your local machine with no external API or paid service.
The system uses a transformer language model trained from scratch.
Domain agents analyse code, art, video, prompting, and research continuously.
The Orchestrator coordinates agents and collects improvement proposals.
The Fusion agent merges and de-duplicates proposals from all domains.
High-risk proposals are queued for human review before being applied.
All knowledge is stored locally in the data/ directory.
Users can add their own reference files, videos, and texts at any time.
The Knowledge Base uses TF-IDF search, which requires no external database.
Video files can be transcribed locally using openai-whisper (free, MIT licence).
PDF text extraction uses pdfminer.six (free, MIT licence).
The system checkpoints itself after every improvement cycle.
Rollback is always possible via the CheckpointManager.
""" * 50   # repeat to give the tokenizer enough data


def main() -> None:
    from darwin.core.config import ModelConfig, TrainingConfig, DarwinConfig
    from darwin.core.model import DarwinLM
    from darwin.core.tokenizer import BPETokenizer
    from darwin.core.trainer import TextDataset, Trainer
    import torch.utils.data

    print("=== Darwin Bootstrap ===")
    print("Training tokenizer…")
    tok = BPETokenizer()
    tok.train(SAMPLE_TEXT, vocab_size=1_000)
    Path("data").mkdir(exist_ok=True)
    tok.save("data/tokenizer.json")
    print(f"  Vocabulary size: {tok.vocab_size}")

    print("Tokenising corpus…")
    ids = tok.encode(SAMPLE_TEXT)
    seq_len = 64
    dataset = TextDataset(ids, seq_len)
    split = max(1, int(len(dataset) * 0.9))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    eval_ds  = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    print(f"  Train batches: {len(train_ds)}, Eval batches: {len(eval_ds)}")

    print("Building model…")
    cfg = DarwinConfig()
    cfg.model.vocab_size = tok.vocab_size
    cfg.model.max_seq_len = seq_len
    cfg.model.n_layers = 2
    cfg.model.n_heads = 2
    cfg.model.n_kv_heads = 2
    cfg.model.d_model = 64
    cfg.model.d_ff = 256
    cfg.training.max_steps = 200
    cfg.training.batch_size = 4
    cfg.training.micro_batch_size = 4
    cfg.training.save_every = 200
    cfg.training.eval_every = 100
    cfg.training.log_every = 50

    model = DarwinLM.from_config(cfg.model)
    print(f"  Parameters: {model.num_parameters():,}")

    print("Training (200 steps)…")
    trainer = Trainer(model, cfg.training, train_ds, eval_ds)
    trainer.train()

    print("\nBootstrap complete!  Try:")
    print("  darwin generate 'The quick brown' --small --data-dir data")
    print("  darwin kb add <your-file.txt>")
    print("  darwin serve --small")


if __name__ == "__main__":
    main()
