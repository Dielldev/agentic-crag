"""Routing logic for the CRAG pipeline."""

from statistics import mean

from app.graph.state import GraphState
from app.logger import get_logger

logger = get_logger("routing")


def route_after_evaluation(state: GraphState) -> str:
    evals = state["eval_result"]
    iteration = state.get("iteration", 0)

    # Web search is the terminal recovery step. Once it has run, generate with
    # whatever it returned — never route back into web_search (infinite loop) or
    # rewrite (which would discard the web results and retry vector retrieval).
    if state.get("web_search_done", False):
        logger.info(
            "post-web-search fallback | iteration: %d → routing to GENERATE",
            iteration,
        )
        return "generate"

    if not evals:
        logger.info(
            "Avg score: n/a | no chunks evaluated | iteration: %d → routing to WEB SEARCH",
            iteration,
        )
        return "web_search"

    avg_score = mean(
        (e.relevance + e.completeness + e.confidence) / 3 for e in evals
    )

    if avg_score >= 70:
        logger.info(
            "Avg score: %.0f | threshold met (>= 70) | iteration: %d → routing to GENERATE",
            avg_score,
            iteration,
        )
        return "generate"
    if avg_score >= 40 and iteration < 2:
        logger.info(
            "Avg score: %.0f | mixed (40-69), retries left | iteration: %d → routing to REWRITE",
            avg_score,
            iteration,
        )
        return "rewrite"
    # avg_score >= 40 and iteration >= 2, or avg_score < 40
    reason = "max iterations reached" if avg_score >= 40 else "threshold not met"
    logger.info(
        "Avg score: %.0f | %s | iteration: %d → routing to WEB SEARCH",
        avg_score,
        reason,
        iteration,
    )
    return "web_search"
