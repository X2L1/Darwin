# Darwin 🧬

> **A continuously self-improving, multi-agent AI system that runs entirely on your local machine.**
> No API keys. No paid services. No cloud subscriptions. 100% free and open-source.

---

## Table of Contents

1. [What is Darwin?](#what-is-darwin)
2. [Architecture Overview](#architecture-overview)
3. [Quick Start](#quick-start)
4. [Using Your Own Files, Videos & Texts as Reference](#using-your-own-files-videos--texts-as-reference)
5. [Self-Improvement Loop](#self-improvement-loop)
6. [Domain Agents](#domain-agents)
7. [REST API](#rest-api)
8. [CLI Reference](#cli-reference)
9. [Configuration](#configuration)
10. [Governance & Safety](#governance--safety)
11. [Development & Testing](#development--testing)
12. [Everything is Free](#everything-is-free)

---

## What is Darwin?

Darwin is a self-contained AI system built on a **GPT-style transformer
language model** trained and run entirely on your own hardware.  It
continuously analyses its own code and outputs, proposes improvements,
validates them, and merges the best ones – forming a closed loop that
makes the system progressively better over time.

Key properties:

| Property | Value |
|---|---|
| Inference cost | **$0** – runs on CPU or GPU locally |
| External APIs | **None** required |
| Paid services | **None** |
| Licence | MIT |
| Languages | Python 3.10+ |

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────┐
│                     Darwin System                      │
│                                                        │
│  ┌──────────────┐    ┌──────────────────────────────┐ │
│  │  Foundation  │    │        Domain Agents          │ │
│  │    Model     │    │  Code · Art · Video ·         │ │
│  │  (local LLM) │    │  Prompting · Research         │ │
│  └──────┬───────┘    └──────────────┬───────────────┘ │
│         │                           │                  │
│         │        ┌──────────────────▼───────────────┐ │
│         │        │       Orchestrator               │ │
│         │        │  (schedules & coordinates agents)│ │
│         │        └──────────────────┬───────────────┘ │
│         │                           │                  │
│         │        ┌──────────────────▼───────────────┐ │
│         │        │     Fusion / Integrator Agent    │ │
│         │        │  (de-dup, conflict resolution,   │ │
│         │        │   KB-boosting, ranking)          │ │
│         │        └──────────────────┬───────────────┘ │
│         │                           │                  │
│         │   ┌───────────────────────▼──────────────┐  │
│         │   │         Self-Improvement Loop        │  │
│         │   │  Validate → ReviewGate → Merge →     │  │
│         │   │  Checkpoint → Metrics → Sleep        │  │
│         │   └───────────────────────┬──────────────┘  │
│         │                           │                  │
│         └───────────────────────────┘                  │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              Knowledge Base (local)              │  │
│  │  User-supplied files · videos · texts · PDFs    │  │
│  │  TF-IDF search index (no external database)     │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Bootstrap (train a tiny model to verify everything works)

```bash
python scripts/bootstrap.py
```

### 3. Add your own reference material

```bash
darwin kb add my_notes.txt
darwin kb add lecture.mp4          # transcribed locally with Whisper (free)
darwin kb add research_paper.pdf   # extracted with pdfminer.six (free)
darwin kb add docs/                # whole directory
```

### 4. Run the self-improvement loop

```bash
darwin run --small                 # use a small model for fast iteration
```

### 5. Generate text

```bash
darwin generate "Once upon a time" --small
```

### 6. Launch the REST API

```bash
darwin serve --small
# open http://127.0.0.1:8000/docs
```

---

## Using Your Own Files, Videos & Texts as Reference

Darwin's **Knowledge Base** lets you supply your own reference materials.
The system prioritises these when generating responses and when agents
propose improvements.

### Supported formats (all processed locally, zero cost)

| Format | How it's handled |
|---|---|
| `.txt` `.md` `.rst` `.csv` | Read as UTF-8 text |
| `.pdf` | Text extracted with **pdfminer.six** (free, MIT) |
| `.html` `.htm` | Tags stripped with stdlib |
| `.srt` `.vtt` | Subtitle tracks → clean transcript |
| `.json` `.jsonl` | Serialised as readable text |
| `.mp4` `.mkv` `.avi` `.mov` `.webm` … | Sidecar subtitles + **ffprobe** metadata + optional **Whisper** transcript |
| `.mp3` `.wav` `.flac` `.ogg` … | Transcribed with local **Whisper** |
| Any other text-like file | Attempted UTF-8 read |

For video/audio transcription install [openai-whisper](https://github.com/openai/whisper)
(free, MIT licence, runs on CPU or GPU):

```bash
pip install openai-whisper
```

### CLI commands

```bash
# Add a single file
darwin kb add my_book.txt

# Add a video (transcript extracted locally with Whisper)
darwin kb add lecture.mp4 --tag lecture --tag ai

# Add a whole directory recursively
darwin kb add ~/my-research-papers/

# List all entries
darwin kb list

# List only primary references
darwin kb list --primary-only

# Search the knowledge base (local TF-IDF, no external service)
darwin kb search "attention mechanism transformer"

# Remove an entry
darwin kb remove <entry-id>
```

### REST API

```bash
# Add a file via API
curl -X POST http://localhost:8000/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/notes.txt", "tags": ["notes"]}'

# Search
curl "http://localhost:8000/knowledge/search?q=transformer+attention&top_k=3"

# List all entries
curl http://localhost:8000/knowledge
```

---

## Self-Improvement Loop

Each cycle:

1. **Orchestrator** dispatches context to all domain agents
2. Agents return **Proposals** (code improvements, prompt rewrites, dataset curation tasks, …)
3. **Fusion Agent** de-duplicates, resolves conflicts, boosts proposals aligned with your reference materials, and ranks by utility
4. **Validator** runs code proposals in a **sandboxed subprocess** (time-limited, network-disabled)
5. **ReviewGate** auto-approves low-risk proposals; queues high-risk ones for human review
6. **Merger** applies approved proposals and logs advisory ones
7. **CheckpointManager** saves a versioned model snapshot
8. **Metrics** are recorded locally as JSON lines
9. Loop sleeps until the next interval

```bash
# Start the loop (cycles every hour by default)
darwin run

# Run with a faster cycle for development
darwin run --interval 60 --small

# Review pending high-risk proposals
darwin review list
darwin review approve <proposal-id>
darwin review reject  <proposal-id>
```

---

## Domain Agents

| Agent | What it analyses | Example proposals |
|---|---|---|
| **Code** | AST complexity, duplication, coverage, security patterns | "Reduce cyclomatic complexity of `_parse` (score 14)", "Add tests for `trainer.py`" |
| **Art** | Prompt quality, style consistency, dataset size | "Add style modifiers to 12 plain prompts", "Fine-tune on 500 curated images" |
| **Video** | Temporal coherence, frame quality, FPS stability, latency | "Apply optical-flow smoothing (3/10 clips below threshold)" |
| **Prompting** | Vague language, missing CoT/few-shot, prompt length | "Add chain-of-thought to 5 low-scoring prompts" |
| **Research** | Corpus size, knowledge gaps, synthetic data | "Generate 4 000 synthetic Q&A pairs from local corpus" |

---

## REST API

Start the server:

```bash
darwin serve --port 8000 --small
```

Interactive docs: `http://127.0.0.1:8000/docs`

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/status` | Model params, KB size, metrics |
| POST | `/cycle/run` | Trigger one improvement cycle |
| GET | `/proposals` | Recent proposals from all agents |
| GET | `/knowledge` | List KB entries |
| POST | `/knowledge/add` | Add a reference file/video/text |
| DELETE | `/knowledge/{id}` | Remove a KB entry |
| GET | `/knowledge/search?q=…` | TF-IDF search |
| GET | `/reviews/pending` | Proposals awaiting human review |
| POST | `/reviews/{id}/resolve` | Approve or reject a proposal |
| POST | `/generate` | Generate text (local model, free) |

---

## CLI Reference

```
darwin --help

Commands:
  train      Pre-train or fine-tune the local model on a text file
  run        Start the continuous self-improvement loop
  serve      Launch the REST API server
  generate   Generate text using the local model
  kb         Manage primary reference files, videos, and texts
    add      Ingest a file/directory/video
    list     List all KB entries
    search   Search the KB
    remove   Remove an entry
  review     Human review workflow for high-risk proposals
    list     List pending proposals
    approve  Approve a proposal
    reject   Reject a proposal
  status     Print system status and metrics summary
  benchmark  Run all local benchmarks
```

---

## Configuration

Edit `configs/default.yaml` or pass flags to the CLI:

```yaml
model:
  n_layers: 12        # increase for a larger model
  d_model: 512

improvement_interval_seconds: 3600
require_human_review_above_risk: 0.7  # 0–1 risk threshold

enabled_domains:
  - code
  - art
  - video
  - prompting
  - research
```

---

## Governance & Safety

* **Sandboxed execution** – code proposals run in an isolated subprocess with a CPU time limit and network access disabled
* **ReviewGate** – proposals above a configurable risk score are held for human approval
* **Versioned checkpoints** – the model is saved before and after every cycle; rollback is one command away
* **Advisory proposals** – non-code proposals (art, video, etc.) are logged as advisory notes, never applied automatically

---

## Development & Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest -v

# Run a specific test file
pytest tests/test_knowledge.py -v

# Lint
ruff check darwin tests
```

---

## Everything is Free

| Capability | Tool / Library | Licence | Cost |
|---|---|---|---|
| Foundation model | PyTorch (custom transformer) | BSD | Free |
| Tokenizer | Pure Python BPE | MIT | Free |
| PDF extraction | pdfminer.six (optional) | MIT | Free |
| Video metadata | ffprobe / FFmpeg (optional) | LGPL | Free |
| Audio/video transcription | openai-whisper (optional) | MIT | Free |
| Search index | Pure Python TF-IDF | MIT | Free |
| REST API | FastAPI + Uvicorn | MIT | Free |
| CLI | Click | BSD | Free |
| Sandboxing | Python subprocess | PSF | Free |
| Storage | Local JSON files | — | Free |

**No OpenAI API. No Anthropic API. No Hugging Face Inference API. No cloud database. No vector store subscription. No tokens to buy.**
