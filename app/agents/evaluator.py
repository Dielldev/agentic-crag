"""Evaluator agent — grades each retrieved chunk for relevance to the query.

Scores are 0–100 integers. Routing thresholds (from CLAUDE.md):
  all >= 70  → generate
  mixed      → rewrite and retry
  all < 40   → web search fallback
"""

from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from pydantic import BaseModel

from app.agents.retriever import RetrievedChunk
from app.llm_client import chat, extract_json
from app.logger import get_logger

logger = get_logger("evaluator")

# Cap on concurrent grading calls. Equal to the default top_k, so a single query
# never bursts more than a handful of requests at Groq's free-tier rate limits.
_MAX_GRADERS = 5

_SYSTEM_PROMPT = """\
You are a retrieval quality grader. Given a user query and a text chunk, \
score how useful the chunk is for answering the query.

Respond with valid JSON only — no prose, no markdown fences:
{"relevance": <0-100>, "completeness": <0-100>, "confidence": <0-100>}

relevance:    does the chunk address the query topic?
completeness: how much of the query could be answered from this chunk alone?
confidence:   how certain are you of the above scores?
"""


class ChunkEval(BaseModel):
    chunk: RetrievedChunk
    relevance: int
    completeness: int
    confidence: int


def _grade_chunk(query: str, chunk: RetrievedChunk) -> ChunkEval:
    """Grade a single chunk. Never raises — on any LLM/parse error, returns a
    zeroed eval so a single bad call can't crash the whole evaluation batch.
    A zeroed score fails toward RECOVER (rewrite/web search) rather than falsely
    proceeding to generate from a chunk we couldn't actually grade.
    """
    user_message = f"Query: {query}\n\nChunk:\n{chunk.text}"
    try:
        raw = chat(_SYSTEM_PROMPT, user_message, json_mode=True)
        scores = extract_json(raw)
        return ChunkEval(
            chunk=chunk,
            relevance=int(scores["relevance"]),
            completeness=int(scores["completeness"]),
            confidence=int(scores["confidence"]),
        )
    except Exception as exc:  # transport error, bad JSON, missing key, etc.
        logger.warning("Grading failed for chunk %d (%s) — defaulting to 0: %s",
                       chunk.chunk_index, chunk.source, exc)
        return ChunkEval(chunk=chunk, relevance=0, completeness=0, confidence=0)


def evaluate(query: str, chunks: list[RetrievedChunk]) -> list[ChunkEval]:
    """Grade each chunk and return a ChunkEval per input chunk (order preserved).

    Chunks are graded concurrently (bounded by _MAX_GRADERS) since each grade is
    an independent, blocking Groq call.
    """
    logger.info("Evaluating %d chunks for question: %s", len(chunks), query)
    if chunks:
        with ThreadPoolExecutor(max_workers=min(len(chunks), _MAX_GRADERS)) as pool:
            evals = list(pool.map(lambda c: _grade_chunk(query, c), chunks))
    else:
        evals = []

    if evals:
        relevance = round(mean(e.relevance for e in evals))
        completeness = round(mean(e.completeness for e in evals))
        confidence = round(mean(e.confidence for e in evals))
        avg = (relevance + completeness + confidence) / 3
        decision = "PROCEED" if avg >= 70 else "RECOVER"
        logger.info(
            "Scores → relevance: %d | completeness: %d | confidence: %d | decision: %s",
            relevance,
            completeness,
            confidence,
            decision,
        )
    else:
        logger.info("Scores → no chunks to evaluate | decision: RECOVER")
    return evals
