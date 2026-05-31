"""Tests for chunk_text (app/vectorstore/ingestion.py) — token-window chunking."""

from app.vectorstore.ingestion import CHUNK_SIZE, _encoding, chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []


def test_short_text_is_single_chunk_roundtrip():
    text = "Buffer overflow is a memory-safety vulnerability."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    # A single sub-CHUNK_SIZE window should decode back to the original ASCII text.
    assert chunks[0] == text


def test_long_text_splits_into_multiple_bounded_chunks():
    # ~1500 tokens, comfortably past CHUNK_SIZE (400) -> several chunks.
    text = "security " * 1500
    chunks = chunk_text(text)
    assert len(chunks) > 1
    # No chunk exceeds the configured window size.
    for chunk in chunks:
        assert chunk.strip()  # non-empty
        assert len(_encoding.encode(chunk)) <= CHUNK_SIZE
