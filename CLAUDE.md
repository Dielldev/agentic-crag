# Agentic CRAG System

## What This Is
A Corrective RAG (CRAG) pipeline that self-evaluates retrieval quality and recovers 
from poor results before generating answers. Built with LangGraph as a stateful graph.

## Tech Stack
- **Orchestration:** LangGraph (Python)
- **LLM:** Groq (llama-3.3-70b-versatile) for all chat/grading/generation
- **Embeddings:** FastEmbed BAAI/bge-small-en-v1.5 (384-dim, runs locally on CPU, no API key)
- **Vector DB:** Qdrant (local via docker-compose, port 6333)
- **Web Search Fallback:** Tavily
- **Knowledge Source:** Wikipedia (security articles, ingested to data/raw/)
- **Backend:** FastAPI
- **Observability:** LangSmith

## Project Structure
app/graph/      → workflow.py, nodes.py, routing.py, state.py
app/agents/     → retriever, evaluator, rewriter, web_search, generator
app/vectorstore/→ embeddings, ingestion, database
app/api/        → main.py (FastAPI entrypoint)
app/            → llm_client.py (shared Groq client), config.py, logger.py
scripts/        → fetch_wikipedia.py (data ingestion)
data/raw/       → fetched source .txt articles (the ingestion source)

## Agents & Responsibilities
1. retriever.py    — Qdrant vector search (client.query_points), returns top-k RetrievedChunk
2. evaluator.py    — LLM grades chunks (relevance/completeness/confidence), returns ChunkEval
3. rewriter.py     — reformulates query when eval scores are mixed
4. web_search.py   — Tavily fallback when vector retrieval is too weak
5. generator.py    — grounded answer generation, no hallucination
6. verifier.py     — post-generation fact check; grades whether the answer is grounded
                     in the retrieved context (runs after generate, before END)

## Routing Logic
Routing is driven by the **average** of per-chunk (relevance + completeness + confidence)/3:
- avg ≥ 70 → generate
- 40 ≤ avg < 70 and iteration < 2 → rewrite query → retry retrieval
- avg < 40, OR (avg ≥ 40 with iterations exhausted), OR no chunks → web search fallback

## Conventions
- Agents return typed Pydantic models (RetrievedChunk, ChunkEval)
- LangGraph state (GraphState TypedDict): question, chunks, eval_result, answer, iteration, search_type
- Max retrieval loops: 2 (MAX_RETRIEVAL_LOOPS in state.py; prevents infinite cycles)
- Embeddings run locally (FastEmbed); only chat/grading/generation hit an external API (Groq)
- Every LLM call goes through the shared chat() wrapper in app/llm_client.py
- Ingestion chunks raw text with tiktoken (CHUNK_SIZE=400, CHUNK_OVERLAP=50) in-memory, then upserts to Qdrant
- Env vars via .env + pydantic-settings, never hardcoded keys (see app/config.py)
