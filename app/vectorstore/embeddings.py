"""Text embedding via FastEmbed (local, no API key).

Uses BAAI/bge-small-en-v1.5 (384-dim) — runs locally on CPU, integrates
cleanly with Qdrant. FastEmbed batches internally.
"""

from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384

# Loaded lazily on first use so merely importing this module (e.g. via the
# retriever during tests or API startup) doesn't pull the ONNX model into memory.
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Return the embedding model, constructing it once on first call."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, returning one vector per input (order preserved)."""
    return [vector.tolist() for vector in _get_model().embed(texts)]
