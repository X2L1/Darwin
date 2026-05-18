"""Tests for the unified Darwin chat brain."""

from __future__ import annotations

from pathlib import Path

from darwin.core.config import DarwinConfig
from darwin.core.model import DarwinLM
from darwin.core.tokenizer import BPETokenizer
from darwin.evaluation.metrics import MetricsCollector
from darwin.improvement.loop import ImprovementLoop
from darwin.knowledge.base import KnowledgeBase, KnowledgeEntry
from darwin.orchestrator.brain import UnifiedDarwinBrain
from darwin.orchestrator.orchestrator import Orchestrator


def _tiny_config(tmp_path: Path) -> DarwinConfig:
    cfg = DarwinConfig()
    cfg.data_dir = str(tmp_path)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.training.output_dir = str(tmp_path / "checkpoints")
    cfg.model.vocab_size = 512
    cfg.model.max_seq_len = 64
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.model.n_kv_heads = 2
    cfg.model.d_model = 32
    cfg.model.d_ff = 64
    cfg.proposal_budget = 3
    cfg.validation_timeout_seconds = 1
    cfg.enabled_domains = ["code"]
    return cfg


def _system(tmp_path: Path):
    tok = BPETokenizer()
    tok.train("Darwin can chat and improve itself safely. " * 10, vocab_size=512)
    cfg = _tiny_config(tmp_path)
    cfg.model.vocab_size = tok.vocab_size
    model = DarwinLM.from_config(cfg.model)
    kb = KnowledgeBase(store_dir=tmp_path / "knowledge")
    metrics = MetricsCollector(log_dir=cfg.log_dir)
    loop = ImprovementLoop(cfg, model, tok, kb)
    brain = UnifiedDarwinBrain(cfg, model, tok, kb, loop, metrics)
    return cfg, model, tok, kb, metrics, loop, brain


def test_status_is_unified(tmp_path: Path) -> None:
    *_, brain = _system(tmp_path)
    response = brain.chat("status")
    assert response.intent == "status"
    assert "unified assistant" in response.message
    assert response.data["model"]["provider"] == "local-custom"


def test_help_response_is_actionable(tmp_path: Path) -> None:
    *_, brain = _system(tmp_path)
    response = brain.chat("what can you do?")
    assert response.intent == "help"
    assert "status" in response.message
    assert "improve yourself" in response.message


def test_repetitive_fallback_question_explains_training_state(tmp_path: Path) -> None:
    *_, brain = _system(tmp_path)
    response = brain.chat("why is chat giving the same response?")
    assert response.intent == "training"
    assert "weights" in response.message


def test_untrained_open_chat_gets_grounded_reply(tmp_path: Path) -> None:
    *_, brain = _system(tmp_path)

    class AlwaysFallback:
        def describe(self):
            return {"provider": "local-custom", "parameters": 123, "source_files": []}

        def complete(self, *args, **kwargs):
            from darwin.core.llm import LLMGeneration

            return LLMGeneration(text="static fallback", used_fallback=True)

    brain.llm = AlwaysFallback()
    response = brain.chat("Can you answer normal questions yet?")
    assert response.intent == "chat"
    assert response.message != "static fallback"
    assert "open-ended language model still needs training" in response.message
    assert any(action["type"] == "grounded_system_reply" for action in response.actions)


def test_source_response_lists_model_source(tmp_path: Path) -> None:
    *_, brain = _system(tmp_path)
    response = brain.chat("show me the model source")
    assert response.intent == "source"
    assert any(path.endswith("model.py") for path in response.data["model"]["source_files"])


def test_improvement_intent_can_be_disabled(tmp_path: Path) -> None:
    *_, brain = _system(tmp_path)
    response = brain.chat("improve yourself", run_improvements=False)
    assert response.intent == "improve"
    assert response.actions == []


def test_knowledge_search_is_hidden_behind_chat(tmp_path: Path) -> None:
    *_, kb, _, _, brain = _system(tmp_path)
    kb.add_entry(KnowledgeEntry(title="Clean Code", chunks=["small functions and clear tests"]))
    response = brain.chat("search knowledge clean functions")
    assert response.intent == "knowledge"
    assert response.data["results"][0]["title"] == "Clean Code"


def test_orchestrator_one_shot_cycle_does_not_need_started_scheduler(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    cfg.enabled_domains = ["code"]
    source_root = tmp_path / "darwin"
    source_root.mkdir()
    (source_root / "unsafe.py").write_text("eval(input())\n", encoding="utf-8")

    orchestrator = Orchestrator(cfg, knowledge_base=KnowledgeBase(store_dir=tmp_path / "knowledge"))
    summary = orchestrator.run_improvement_cycle({"code": {"source_root": str(source_root)}})

    assert summary["total_proposals"] >= 1
    assert summary["agent_errors"] == {}


def test_apply_unified_patch(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    diff = """--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""
    ImprovementLoop._apply_patch("sample.py", diff, project_root=tmp_path)
    assert target.read_text(encoding="utf-8") == "def f():\n    return 2\n"
    assert (tmp_path / "sample.py.bak").exists()
