"""Vector store retrieval agent."""

from pydantic import BaseModel

from app.config import settings
from app.logger import get_logger
from app.vectorstore.database import client
from app.vectorstore.embeddings import embed_texts

logger = get_logger("retriever")


class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Search the Qdrant collection and return the top_k most relevant chunks."""
    logger.info('Searching for: "%s"', query)
    vector = embed_texts([query])[0]
    results = client.query_points(
        collection_name=settings.collection_name,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    chunks = [
        RetrievedChunk(
            text=hit.payload["text"],
            source=hit.payload["source"],
            chunk_index=hit.payload["chunk_index"],
            score=hit.score,
        )
        for hit in results.points
    ]
    scores = ", ".join(f"{c.score:.2f}" for c in chunks)
    logger.info("Retrieved %d chunks | scores: %s", len(chunks), scores or "—")
    return chunks
