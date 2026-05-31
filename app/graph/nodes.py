"""LangGraph node functions for the CRAG pipeline.

Each node receives GraphState and returns a partial dict of fields to update.
"""

from app.agents.evaluator import evaluate
from app.agents.generator import generate
from app.agents.retriever import retrieve
from app.agents.rewriter import rewrite
from app.agents.verifier import verify
from app.agents.web_search import web_search
from app.graph.state import GraphState


def retrieval_node(state: GraphState) -> dict:
    chunks = retrieve(state["question"])
    return {"chunks": chunks, "search_type": "vector"}


def evaluation_node(state: GraphState) -> dict:
    eval_result = evaluate(state["question"], state["chunks"])
    return {"eval_result": eval_result}


def rewrite_node(state: GraphState) -> dict:
    new_question = rewrite(state["question"])
    return {
        "question": new_question,
        "iteration": state.get("iteration", 0) + 1,
    }


def web_search_node(state: GraphState) -> dict:
    chunks = web_search(state["question"])
    # web search is the terminal recovery step — flag it so routing never loops back.
    return {"chunks": chunks, "search_type": "web", "web_search_done": True}


def generation_node(state: GraphState) -> dict:
    result = generate(state["question"], state["chunks"])
    return {"answer": result.answer}


def verification_node(state: GraphState) -> dict:
    result = verify(state["question"], state["answer"], state["chunks"])
    return {"verification": result}
