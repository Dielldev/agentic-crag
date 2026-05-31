"""Verifier agent — post-generation fact check.

Grades whether the generated answer is actually grounded in the retrieved
context, catching hallucinations or unsupported claims before the answer is
returned to the user.
"""

from pydantic import BaseModel

from app.agents.retriever import RetrievedChunk
from app.llm_client import chat, extract_json
from app.logger import get_logger

logger = get_logger("verifier")

_SYSTEM_PROMPT = """\
You are a fact-checking grader. You are given a user question, an answer that \
was generated, and the context chunks the answer was supposed to be based on. \
Judge whether every claim in the answer is supported by the context.

Respond with valid JSON only — no prose, no markdown fences:
{"grounded": <true|false>, "confidence": <0-100>, "notes": "<short reason>"}

grounded:   true only if all claims in the answer are supported by the context.
confidence: how certain you are of that judgement.
notes:      brief explanation; name any unsupported claim if grounded is false.
"""


class Verification(BaseModel):
    grounded: bool
    confidence: int
    notes: str


def verify(query: str, answer: str, chunks: list[RetrievedChunk]) -> Verification:
    """Check the answer against the context chunks and report grounding."""
    logger.info("Verifying answer against %d context chunks", len(chunks))

    context_parts = [
        f"[{i + 1}] (source: {chunk.source})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    ]
    context_block = "\n\n".join(context_parts)
    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer to verify:\n{answer}"
    )

    try:
        raw = chat(_SYSTEM_PROMPT, user_message, json_mode=True)
        data = extract_json(raw)
        result = Verification(
            grounded=bool(data["grounded"]),
            confidence=int(data["confidence"]),
            notes=str(data.get("notes", "")),
        )
    except Exception as exc:  # transport error, bad JSON, missing key, etc.
        # Fail toward flagging: if we can't verify, don't claim the answer is grounded.
        logger.warning("Verification failed — marking ungrounded: %s", exc)
        result = Verification(
            grounded=False, confidence=0, notes="verification parse failed"
        )
    logger.info(
        "Verification → grounded: %s | confidence: %d | %s",
        result.grounded,
        result.confidence,
        result.notes,
    )
    return result
