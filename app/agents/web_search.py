"""Web search fallback agent — queries Tavily when vector retrieval scores are too low."""

from tavily import TavilyClient

from app.agents.retriever import RetrievedChunk
from app.config import settings
from app.logger import get_logger

logger = get_logger("web_search")

_client = TavilyClient(api_key=settings.tavily_api_key)


def web_search(query: str) -> list[RetrievedChunk]:
    """Search the web via Tavily and return results as RetrievedChunks."""
    logger.info("Vector retrieval insufficient — falling back to web search")
    logger.info("Querying Tavily: %s", query)
    response = _client.search(query, max_results=5)
    chunks = [
        RetrievedChunk(
            text=result["content"],
            source=result["url"],
            chunk_index=0,
            # Tavily returns a 0–1 relevance score per result, matching the
            # cosine-similarity scale used for vector hits.
            score=float(result.get("score", 1.0)),
        )
        for result in response["results"]
    ]
    urls = ", ".join(c.source for c in chunks)
    logger.info("Tavily returned %d results | URLs: %s", len(chunks), urls or "—")
    return chunks
