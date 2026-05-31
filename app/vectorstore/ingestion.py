"""Ingest data/raw/*.txt into the Qdrant vector store.

Pipeline: read raw text files -> token-chunk with tiktoken -> embed in
batches -> upsert into the collection with source/chunk metadata.

Run with: python -m app.vectorstore.ingestion
"""

import uuid
from pathlib import Path

import tiktoken
from qdrant_client.models import PointStruct

from app.config import settings
from app.vectorstore.database import client, get_or_create_collection
from app.vectorstore.embeddings import VECTOR_SIZE, embed_texts

# scripts run from anywhere -> resolve project root -> data/raw
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Generic tiktoken encoder used purely to size chunks by token count.
_encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> list[str]:
    """Split text into ~CHUNK_SIZE-token chunks with CHUNK_OVERLAP overlap."""
    tokens = _encoding.encode(text)
    chunks: list[str] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_SIZE]
        if not window:
            break
        chunks.append(_encoding.decode(window))
        if start + CHUNK_SIZE >= len(tokens):
            break
    return chunks


def ingest() -> None:
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {RAW_DIR}")
        return

    collection = get_or_create_collection(settings.collection_name, VECTOR_SIZE)
    print(f"Collection ready: {collection}")

    # Build chunks across all files, tracking metadata per chunk.
    texts: list[str] = []
    payloads: list[dict] = []
    for path in files:
        content = path.read_text(encoding="utf-8")
        chunks = chunk_text(content)
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            payloads.append(
                {"source": path.name, "chunk_index": idx, "text": chunk}
            )
        print(f"  {path.name}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(texts)} chunks...")
    vectors = embed_texts(texts)

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
        for vector, payload in zip(vectors, payloads)
    ]

    print(f"Upserting {len(points)} points into '{collection}'...")
    client.upsert(collection_name=collection, points=points)

    print(f"\nDone. Ingested {len(points)} chunks from {len(files)} files.")


if __name__ == "__main__":
    ingest()
