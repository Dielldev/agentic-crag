"""LangGraph state definition for the CRAG pipeline."""

from typing_extensions import TypedDict

from app.agents.evaluator import ChunkEval
from app.agents.retriever import RetrievedChunk
from app.agents.verifier import Verification

MAX_RETRIEVAL_LOOPS = 2

# Type alias used throughout the graph
EvalResult = list[ChunkEval]


class GraphState(TypedDict, total=False):
    question: str
    chunks: list[RetrievedChunk]
    eval_result: EvalResult
    answer: str
    iteration: int                # tracks how many retrieval loops, max 2
    search_type: str              # "vector" or "web"
    web_search_done: bool         # web search is terminal — set once it runs
    verification: Verification    # post-generation fact check
