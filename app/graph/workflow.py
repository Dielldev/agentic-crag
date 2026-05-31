"""Compiled LangGraph workflow for the CRAG pipeline."""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    evaluation_node,
    generation_node,
    retrieval_node,
    rewrite_node,
    verification_node,
    web_search_node,
)
from app.graph.routing import route_after_evaluation
from app.graph.state import GraphState

_builder = StateGraph(GraphState)

_builder.add_node("retrieve", retrieval_node)
_builder.add_node("evaluate", evaluation_node)
_builder.add_node("rewrite", rewrite_node)
_builder.add_node("web_search", web_search_node)
_builder.add_node("generate", generation_node)
_builder.add_node("verify", verification_node)

_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "evaluate")
_builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "web_search": "web_search",
    },
)
_builder.add_edge("rewrite", "retrieve")
_builder.add_edge("web_search", "evaluate")
_builder.add_edge("generate", "verify")
_builder.add_edge("verify", END)

app = _builder.compile()
