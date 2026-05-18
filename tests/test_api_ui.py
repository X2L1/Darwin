"""Tests for the built-in Darwin web UI."""

from __future__ import annotations

from fastapi.testclient import TestClient

from darwin.api import create_app
from darwin.cli import _load_system


def test_root_serves_chat_ui(tmp_path) -> None:
    cfg, model, tokenizer, kb, metrics, loop = _load_system(str(tmp_path), small=True)
    client = TestClient(create_app(cfg, model, tokenizer, kb, loop, metrics))

    response = client.get("/")

    assert response.status_code == 200
    assert "Darwin" in response.text
    assert 'id="chat-form"' in response.text
    assert 'fetch("/chat"' in response.text
    assert 'id="training-fill"' in response.text
    assert 'fetch("/training/progress"' in response.text


def test_chat_endpoint_from_ui_system(tmp_path) -> None:
    cfg, model, tokenizer, kb, metrics, loop = _load_system(str(tmp_path), small=True)
    client = TestClient(create_app(cfg, model, tokenizer, kb, loop, metrics))

    response = client.post("/chat", json={"message": "status"})

    assert response.status_code == 200
    assert response.json()["intent"] == "status"


def test_training_progress_endpoint(tmp_path) -> None:
    cfg, model, tokenizer, kb, metrics, loop = _load_system(str(tmp_path), small=True)
    client = TestClient(create_app(cfg, model, tokenizer, kb, loop, metrics))

    response = client.get("/training/progress")
    data = response.json()

    assert response.status_code == 200
    assert 0 <= data["overall_percent"] <= 100
    assert data["level"]
    assert any(item["id"] == "sentence_completion" for item in data["milestones"])
    assert any(item["id"] == "image_generation" for item in data["milestones"])
