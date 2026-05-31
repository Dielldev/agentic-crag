"""FastAPI entrypoint for the CRAG pipeline.

Run with: uvicorn app.api.main:app --reload
"""

import time
from statistics import mean

from fastapi import FastAPI
from pydantic import BaseModel

from app.graph.workflow import app as graph
from app.logger import get_logger

logger = get_logger("api")

app = FastAPI(title="Agentic CRAG")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: int
    search_type: str
    verified: bool
    verification_notes: str
    warning: str = ""  # non-empty when the answer was judged ungrounded


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger.info("New query received: %s", request.question)
    start = time.perf_counter()

    initial_state = {
        "question": request.question,
        "chunks": [],
        "eval_result": [],
        "answer": "",
        "iteration": 0,
        "search_type": "vector",
        "web_search_done": False,
    }

    result = await graph.ainvoke(initial_state)

    elapsed = time.perf_counter() - start

    if result.get("search_type") == "web":
        route = "vector→web"
    elif result.get("iteration", 0) > 0:
        route = "vector→rewrite"
    else:
        route = "vector only"
    logger.info("Pipeline complete in %.2fs | route taken: %s", elapsed, route)

    sources = list(dict.fromkeys(chunk.source for chunk in result["chunks"]))

    evals = result.get("eval_result", [])
    if evals:
        confidence = round(
            mean((e.relevance + e.completeness + e.confidence) / 3 for e in evals)
        )
    else:
        confidence = 0

    verification = result.get("verification")

    warning = ""
    if verification and not verification.grounded:
        warning = "⚠ This answer may not be fully grounded in the sources"
        if verification.notes:
            warning += f": {verification.notes}"

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        confidence=confidence,
        search_type=result.get("search_type", "vector"),
        verified=verification.grounded if verification else False,
        verification_notes=verification.notes if verification else "",
        warning=warning,
    )
