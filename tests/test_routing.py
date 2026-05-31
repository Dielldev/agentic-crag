"""Tests for the CRAG routing decision (app/graph/routing.py).

Pure logic — no LLM, vector store, or network involved.
"""

from app.agents.evaluator import ChunkEval
from app.agents.retriever import RetrievedChunk
from app.graph.routing import route_after_evaluation


def _eval(score: int) -> ChunkEval:
    """A ChunkEval whose three sub-scores all equal `score`, so its average is `score`."""
    chunk = RetrievedChunk(text="t", source="src.txt", chunk_index=0, score=0.9)
    return ChunkEval(chunk=chunk, relevance=score, completeness=score, confidence=score)


def test_high_score_generates():
    state = {"eval_result": [_eval(80)], "iteration": 0}
    assert route_after_evaluation(state) == "generate"


def test_mixed_score_with_retries_rewrites():
    state = {"eval_result": [_eval(50)], "iteration": 0}
    assert route_after_evaluation(state) == "rewrite"


def test_mixed_score_retries_exhausted_falls_back_to_web():
    state = {"eval_result": [_eval(50)], "iteration": 2}
    assert route_after_evaluation(state) == "web_search"


def test_low_score_falls_back_to_web():
    state = {"eval_result": [_eval(20)], "iteration": 0}
    assert route_after_evaluation(state) == "web_search"


def test_no_chunks_falls_back_to_web():
    state = {"eval_result": [], "iteration": 0}
    assert route_after_evaluation(state) == "web_search"


def test_post_web_search_always_generates():
    """The loop guard: once web search has run, never route back into it."""
    # Even with zero/low evals, web_search_done short-circuits to generate.
    state = {"eval_result": [_eval(10)], "iteration": 2, "web_search_done": True}
    assert route_after_evaluation(state) == "generate"

    state_empty = {"eval_result": [], "iteration": 0, "web_search_done": True}
    assert route_after_evaluation(state_empty) == "generate"


def test_threshold_boundaries():
    # exactly 70 -> generate; just under -> rewrite (retries available)
    assert route_after_evaluation({"eval_result": [_eval(70)], "iteration": 0}) == "generate"
    assert route_after_evaluation({"eval_result": [_eval(69)], "iteration": 0}) == "rewrite"
    # exactly 40 with retries -> rewrite; just under -> web_search
    assert route_after_evaluation({"eval_result": [_eval(40)], "iteration": 0}) == "rewrite"
    assert route_after_evaluation({"eval_result": [_eval(39)], "iteration": 0}) == "web_search"
