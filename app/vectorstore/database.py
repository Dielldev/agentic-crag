"""Qdrant client wrapper and collection management."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings

# Shared client, connected to the URL from config.
client = QdrantClient(url=settings.qdrant_url)


def get_or_create_collection(name: str, vector_size: int = 384) -> str:
    """Ensure a collection exists, creating it with cosine distance if needed.

    vector_size defaults to 384 to match FastEmbed BAAI/bge-small-en-v1.5.
    Returns the collection name.
    """
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    return name
