"""Rewriter agent — reformulates a query to improve retrieval on the next attempt."""

from app.llm_client import chat
from app.logger import get_logger

logger = get_logger("rewriter")

_SYSTEM_PROMPT = """\
You are a search query optimizer. A retrieval step failed to return useful results \
for the user's original query. Reformulate the query to be more specific and \
retrieval-friendly — use precise terminology, remove ambiguity, and focus on the \
core information need. Return only the rewritten query, no explanation.
"""


def rewrite(query: str) -> str:
    """Return a reformulated version of query better suited for vector retrieval."""
    logger.info("Original query: %s", query)
    rewritten = chat(_SYSTEM_PROMPT, query)
    logger.info("Rewritten query: %s", rewritten)
    return rewritten
