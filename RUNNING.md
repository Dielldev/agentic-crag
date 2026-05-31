# Running the Project

## Prerequisites

- Python 3.14 (the `.venv` was created with `python3.14`)
- Docker + Docker Compose (for Qdrant)
- API keys: Groq (required), Tavily (required for web search fallback), LangSmith (optional)

---

## 1. Environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here

# Optional — remove if you don't want LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=agentic-crag

# Leave these as-is for local dev
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=documents
```

---

## 2. Virtual environment

The `.venv` is already created at project root. Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

Install dependencies (if not already installed):

```bash
pip install -r requirements.txt
```

---

## 3. Start Qdrant (Docker)

```bash
docker compose up -d
```

This starts the Qdrant vector database on:
- REST API: `http://localhost:6333`
- gRPC: `http://localhost:6334`

Data persists in the `qdrant_storage` Docker volume.

To stop:

```bash
docker compose down
```

To wipe the vector store and start fresh:

```bash
docker compose down -v
```

---

## 4. Ingest data

### 4a. Fetch Wikipedia articles (skip if `data/raw/` is already populated)

```bash
python scripts/fetch_wikipedia.py
```

Downloads 15 security-focused articles into `data/raw/*.txt`.

### 4b. Chunk and embed into Qdrant

```bash
python -m app.vectorstore.ingestion
```

Reads all `.txt` files from `data/raw/`, chunks them, embeds with FastEmbed (`BAAI/bge-small-en-v1.5`), and upserts into the `documents` collection. No API key needed for embeddings.

---

## 5. Run the API

```bash
uvicorn app.api:app --reload
```

The FastAPI server starts at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## Full startup sequence (fresh clone)

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt

# 3. Set up env
cp .env.example .env
# → edit .env with your API keys

# 4. Start Qdrant
docker compose up -d

# 5. Fetch + ingest knowledge base
python scripts/fetch_wikipedia.py
python -m app.vectorstore.ingestion

# 6. Start API
uvicorn app.api:app --reload
```

---

## Docker commands reference

| Command | Description |
|---|---|
| `docker compose up -d` | Start Qdrant in the background |
| `docker compose down` | Stop Qdrant (data preserved) |
| `docker compose down -v` | Stop Qdrant and delete all stored vectors |
| `docker compose logs -f qdrant` | Tail Qdrant logs |
| `docker compose ps` | Check container status |

---

## Troubleshooting

**`Connection refused` on port 6333** — Qdrant isn't running. Run `docker compose up -d`.

**`GROQ_API_KEY not set`** — Make sure `.env` exists and is populated, and that you activated the venv before starting the server.

**Ingestion finds no files** — Run `python scripts/fetch_wikipedia.py` first to populate `data/raw/`.

**Port 8000 already in use** — Run on a different port: `uvicorn app.api:app --reload --port 8001`.
