"""Darwin REST API – serves the system over HTTP using FastAPI.

100% free: FastAPI + Uvicorn (both MIT-licensed, no paid tiers).
No external service or API key required.

Endpoints
---------
GET  /health                     – liveness probe
GET  /status                     – system status and metrics summary
POST /cycle/run                  – trigger a single improvement cycle
GET  /proposals                  – list recent proposals
GET  /knowledge                  – list knowledge base entries
POST /knowledge/add              – ingest a new reference file/text/video
DELETE /knowledge/{entry_id}     – remove a knowledge base entry
GET  /knowledge/search           – search the knowledge base
GET  /reviews/pending            – list proposals awaiting human review
POST /reviews/{proposal_id}/resolve – approve or reject a proposal
POST /generate                   – generate text from the local model
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Darwin Self-Improving AI",
    description="Locally running, free, self-improving multi-agent AI system.",
    version="0.1.0",
)

# These are set by create_app() or startup event
_system: Optional[Any] = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=0, le=1000)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    total_tokens: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    max_new_tokens: int = Field(160, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    run_improvements: bool = True


class ChatAPIResponse(BaseModel):
    message: str
    intent: str
    actions: List[Dict[str, Any]] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeAddRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file or directory to ingest")
    tags: List[str] = Field(default_factory=list)
    language: str = "en"
    is_primary_reference: bool = True


class ReviewResolveRequest(BaseModel):
    approved: bool
    reviewer: str = "human"
    notes: str = ""


class CycleRunRequest(BaseModel):
    context_overrides: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def web_ui() -> str:
    from darwin.api.ui import render_chat_ui

    return render_chat_ui()


@app.get("/ui", response_class=HTMLResponse)
def web_ui_alias() -> str:
    return web_ui()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> Dict[str, Any]:
    sys = _get_system()
    return {
        "model_parameters": sys["model"].num_parameters(),
        "kb_entries": sys["kb"].count(),
        "metrics_summary": sys["metrics"].summary(),
        "enabled_domains": sys["cfg"].enabled_domains,
    }


@app.get("/training/progress")
def training_progress() -> Dict[str, Any]:
    sys = _get_system()
    from darwin.evaluation.training_progress import build_training_progress

    return build_training_progress(
        sys["cfg"],
        sys["model"],
        sys["tokenizer"],
        sys["kb"],
        sys["metrics"],
    )


@app.post("/cycle/run")
def run_cycle(req: CycleRunRequest) -> Dict[str, Any]:
    sys = _get_system()
    try:
        return sys["loop"].run_once(context_overrides=req.context_overrides or None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/proposals")
def list_proposals(n: int = Query(50, ge=1, le=500)) -> List[Dict[str, Any]]:
    sys = _get_system()
    history = sys["loop"].orchestrator.get_history()
    proposals: List[Dict[str, Any]] = []
    for cycle in history[-10:]:
        for domain_props in cycle.get("proposals_by_domain", {}).values():
            proposals.extend(domain_props)
    return proposals[-n:]


# -- Knowledge Base ----------------------------------------------------------


@app.get("/knowledge")
def list_knowledge(
    media_type: Optional[str] = None,
    primary_only: bool = False,
    tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    sys = _get_system()
    return sys["kb"].list_entries(media_type=media_type, primary_only=primary_only, tag=tag)


@app.post("/knowledge/add")
def add_knowledge(req: KnowledgeAddRequest) -> Dict[str, Any]:
    """Ingest a file, directory, or video as primary reference material.

    This is the main way for users to supply their own reference files,
    texts, and videos.  All processing is local and free.
    """
    sys = _get_system()
    from darwin.knowledge.ingestor import Ingestor

    ingestor = Ingestor()
    try:
        entries = ingestor.ingest(
            req.path,
            tags=req.tags,
            language=req.language,
            is_primary_reference=req.is_primary_reference,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    added_ids = []
    for entry in entries:
        sys["kb"].add_entry(entry)
        added_ids.append(entry.entry_id)

    return {"ingested": len(entries), "entry_ids": added_ids}


@app.delete("/knowledge/{entry_id}")
def remove_knowledge(entry_id: str) -> Dict[str, Any]:
    sys = _get_system()
    removed = sys["kb"].remove_entry(entry_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Entry {entry_id!r} not found.")
    return {"removed": entry_id}


@app.get("/knowledge/search")
def search_knowledge(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    primary_only: bool = False,
) -> List[Dict[str, Any]]:
    sys = _get_system()
    return sys["kb"].search(q, top_k=top_k, primary_only=primary_only)


# -- Reviews -----------------------------------------------------------------


@app.get("/reviews/pending")
def pending_reviews() -> List[Dict[str, Any]]:
    sys = _get_system()
    return sys["loop"].review_gate.list_pending()


@app.post("/reviews/{proposal_id}/resolve")
def resolve_review(proposal_id: str, req: ReviewResolveRequest) -> Dict[str, Any]:
    sys = _get_system()
    found = sys["loop"].review_gate.resolve(
        proposal_id, approved=req.approved, reviewer=req.reviewer, notes=req.notes
    )
    if not found:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id!r} not in pending queue.")
    return {"proposal_id": proposal_id, "approved": req.approved}


# -- Unified chat ------------------------------------------------------------


@app.post("/chat", response_model=ChatAPIResponse)
def chat(req: ChatRequest) -> ChatAPIResponse:
    """Talk to Darwin as one unified assistant."""
    sys = _get_system()
    from darwin.orchestrator.brain import UnifiedDarwinBrain

    brain = UnifiedDarwinBrain(
        sys["cfg"],
        sys["model"],
        sys["tokenizer"],
        sys["kb"],
        sys["loop"],
        sys["metrics"],
    )
    result = brain.chat(
        req.message,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        run_improvements=req.run_improvements,
    )
    return ChatAPIResponse(**result.to_dict())


# -- Text generation ---------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate text using Darwin's local foundation model.

    Runs entirely on your machine – no paid API, no network call.
    """
    import torch

    sys = _get_system()
    model = sys["model"]
    tokenizer = sys["tokenizer"]

    max_prompt_len = max(1, model.cfg.max_seq_len - req.max_new_tokens)
    ids = tokenizer.encode(req.prompt, add_bos=True, max_length=max_prompt_len)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            eos_token_id=tokenizer.eos_id,
        )

    new_tokens = out_ids[0, len(ids):].tolist()
    generated = tokenizer.decode(new_tokens)
    return GenerateResponse(
        prompt=req.prompt,
        generated_text=generated,
        total_tokens=len(ids) + len(new_tokens),
    )


# ---------------------------------------------------------------------------
# System accessor
# ---------------------------------------------------------------------------


def _get_system() -> Dict[str, Any]:
    if _system is None:
        raise HTTPException(status_code=503, detail="System not initialised.  Call create_app() first.")
    return _system


def create_app(
    cfg: Any,
    model: Any,
    tokenizer: Any,
    knowledge_base: Any,
    improvement_loop: Any,
    metrics: Any,
) -> FastAPI:
    """Wire the Darwin system into the FastAPI application and return it."""
    global _system
    _system = {
        "cfg": cfg,
        "model": model,
        "tokenizer": tokenizer,
        "kb": knowledge_base,
        "loop": improvement_loop,
        "metrics": metrics,
    }
    return app
