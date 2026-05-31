"""Generator agent — produces a grounded answer from retrieved context."""

from pydantic import BaseModel

from app.agents.retriever import RetrievedChunk
from app.llm_client import chat
from app.logger import get_logger

logger = get_logger("generator")

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the user's query using \
only the provided context chunks. Do not add information that is not present in \
the context. If the context is insufficient to answer fully, say so explicitly \
rather than speculating. Write the answer as clean prose only — do not mention, \
cite, or reference the source filenames, URLs, or chunk numbers in your answer. \
The sources are tracked and displayed separately, so the answer itself must \
never name where the information came from.
"""


class GeneratedAnswer(BaseModel):
    answer: str
    sources: list[str]


def generate(query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    """Generate a grounded answer from the query and retrieved chunks."""
    logger.info("Generating answer from %d chunks", len(chunks))
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))
    logger.info("Using sources: %s", ", ".join(sources) or "—")

    context_parts = [
        f"[{i + 1}] (source: {chunk.source})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    ]
    context_block = "\n\n".join(context_parts)
    user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

    answer = chat(_SYSTEM_PROMPT, user_message)
    logger.info("Answer preview: %s", answer[:100])
    return GeneratedAnswer(answer=answer, sources=sources)
