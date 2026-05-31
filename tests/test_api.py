"""Tests for the FastAPI layer (app/api/main.py).

The LangGraph workflow is replaced with a fake so no LLM, vector store, or
network is touched — we're testing the request/response wiring only.
"""

from fastapi.testclient import TestClient

import app.api.main as main
from app.agents.evaluator import ChunkEval
from app.agents.retriever import RetrievedChunk
from app.agents.verifier import Verification


def _canned_state(grounded: bool, notes: str) -> dict:
    # Two chunks from the same source -> sources should dedup to one entry.
    c1 = RetrievedChunk(text="a", source="SQL_injection.txt", chunk_index=0, score=0.9)
    c2 = RetrievedChunk(text="b", source="SQL_injection.txt", chunk_index=1, score=0.8)
    evals = [
        ChunkEval(chunk=c1, relevance=60, completeness=60, confidence=60),
        ChunkEval(chunk=c2, relevance=60, completeness=60, confidence=60),
    ]
    return {
        "question": "what is sql injection?",
        "answer": "SQL injection is an attack against a database query.",
        "chunks": [c1, c2],
        "eval_result": evals,
        "search_type": "vector",
        "iteration": 0,
        "verification": Verification(grounded=grounded, confidence=80, notes=notes),
    }


class _FakeGraph:
    def __init__(self, state: dict) -> None:
        self._state = state

    async def ainvoke(self, _initial_state: dict) -> dict:
        return self._state


def test_health():
    client = TestClient(main.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"status": "running"}


def test_query_grounded(monkeypatch):
    monkeypatch.setattr(main, "graph", _FakeGraph(_canned_state(grounded=True, notes="ok")))
    client = TestClient(main.app)

    resp = client.post("/query", json={"question": "what is sql injection?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"].startswith("SQL injection")
    assert data["sources"] == ["SQL_injection.txt"]  # deduped
    assert data["confidence"] == 60                   # mean of the per-chunk averages
    assert data["search_type"] == "vector"
    assert data["verified"] is True
    assert data["warning"] == ""                      # grounded -> no warning


def test_query_ungrounded_sets_warning(monkeypatch):
    notes = "claim about X is unsupported"
    monkeypatch.setattr(main, "graph", _FakeGraph(_canned_state(grounded=False, notes=notes)))
    client = TestClient(main.app)

    resp = client.post("/query", json={"question": "what is sql injection?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["verified"] is False
    assert data["warning"]                # populated
    assert notes in data["warning"]       # includes the verifier's reason
