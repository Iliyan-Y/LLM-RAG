# Basic proof of concept for RAG system

System that uses an LLM and a vector DB (FAISS) to answer questions from a custom dataset of PDFs.

- Retrieval and indexing are biased toward financial filings but can be adapted by changing keywords and prompts.

## Prerequisites

- Python 3.10+
- One of:
  - Ollama (local models, free to run): https://ollama.com
  - OpenAI account + API key (for OpenAI-hosted models)
- Create a `.env` file in the project root (see below)

## Installation

Create and activate a virtual environment, then install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (.env)

Minimum (local Ollama by default):

```
# Vector store and embedding model
VECTORSTORE_DIR=./vectorstore
SEMANTIC_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# PDF source directory for indexing
PDF_DIR=./pdfs

# LLM provider and models
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:latest
```

To use OpenAI instead of Ollama:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
# Optional custom base if needed:
# OPENAI_API_BASE=https://api.openai.com/v1
```

Example of full .env

```
VECTORSTORE_DIR=./vectorstore
OLLAMA_MODEL=gemma3:latest
SEMANTIC_MODEL_NAME=intfloat/e5-base-v2
PDF_DIR=./pdfs
OPENAI_API_KEY=sk-proj-blqblq
OPENAI_MODEL=gpt-5-mini-2025-08-07
LLM_PROVIDER=openai
MEMORY_BUF_SIZE=4
LOGGING_ENABLED=False
```

Notes:

- With `LLM_PROVIDER=ollama`, the app uses your local Ollama server and `OLLAMA_MODEL`.
- With `LLM_PROVIDER=openai`, the app uses OpenAI via `OPENAI_API_KEY` and `OPENAI_MODEL`.

## How to use

- Follow the steps 1, 2 to create your own vector store dataset (preferred option)
- or just create vectorstore dir and unzip the test data from vectorstore.zip inside then proceed to step 3

1. Prepare PDFs

- Create a `pdfs/` folder and add your PDFs.

2. Index PDFs

```
python index_pdfs.py
```

3. Run RAG query loop

```
python query_rag.py
```

## Tips

- For Ollama, make sure the model is pulled, e.g.:
  `ollama pull gemma3:latest`

- For OpenAI, ensure your `OPENAI_API_KEY` is set in `.env`.

- Deactivate the venv with `deactivate` when done.

## WebSocket server (FastAPI) over the RAG pipeline

This project includes a production-ready WebSocket/HTTP API server that wraps the existing RAG pipeline:

- Server: FastAPI + Uvicorn at `/ws` (WebSocket) and `/query` (HTTP POST)
- Containerized: `docker/Dockerfile`
- Orchestrated: `docker-compose.yml`
- Vectorstore bundle baked from `vectorstore.zip` into `/app/vectorstore`
- Embedding model pre-downloaded at build time to avoid cold starts

### Quick start (Docker)

1. Build

```
docker compose build
```

2a) Run with local Ollama (default)

```
# This starts both the RAG WS server and an Ollama container
docker compose --profile ollama up -d
```

2b) Run with OpenAI instead of Ollama

- Set in your `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

- Then start only the API (no Ollama profile needed):

```
docker compose up -d
```

The API will listen on:

- HTTP: `http://localhost:8000`
- WS: `ws://localhost:8000/ws`

Notes:

- With `--profile ollama`, the API uses `OLLAMA_HOST=http://ollama:11434` (internal service).
- The embedding model defined by `SEMANTIC_MODEL_NAME` is pre-fetched during image build.
- The FAISS index is auto-unzipped from `vectorstore.zip` into `/app/vectorstore` during build.

### Endpoints

- HTTP health check

```
GET /health
```

Response example:

```
{"status":"ok","provider":"ollama","model":"gemma3:latest"}
```

- HTTP query

```
POST /query
Content-Type: application/json
{
  "query": "What is in the Q2 filing?"
}
```

Response example:

```
{
  "answer": "...",
  "sources": [
    {
      "source": "path/to.pdf",
      "filing_type": "...",
      "period_end_date": "...",
      "page_label": "..."
    }
  ]
}
```

- WebSocket chat

```
WS /ws
```

Messages are JSON. Server may send:

- `{"type":"welcome","message":"..."}`
- `{"type":"status","message":"processing"}`
- `{"type":"answer","answer":"...","sources":[ ... ]}`
- `{"type":"error","error":"..."}`
  Client should send:
- `{"query":"your question"}`

### Minimal WebSocket client examples

- JavaScript (browser)

```html
<script>
	const ws = new WebSocket("ws://localhost:8000/ws");
	ws.onopen = () => {
		ws.send(JSON.stringify({ query: "What does the latest filing say?" }));
	};
	ws.onmessage = (e) => {
		const msg = JSON.parse(e.data);
		console.log("WS message:", msg);
	};
	ws.onerror = (e) => console.error("WS error:", e);
</script>
```

- Node.js

```js
import WebSocket from "ws";
const ws = new WebSocket("ws://localhost:8000/ws");
ws.on("open", () => ws.send(JSON.stringify({ query: "Hello RAG" })));
ws.on("message", (data) => console.log("WS message:", data.toString()));
```

- curl (HTTP)

```bash
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Summarize the latest 10-K"}' | jq
```

### Configuration

These can be defined in `.env` (and are read inside the container):

- `VECTORSTORE_DIR=/app/vectorstore` (default inside container)
- `SEMANTIC_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`
- `LLM_PROVIDER=ollama` (or `openai`)
- `OLLAMA_MODEL=gemma3:latest`
- `OPENAI_API_KEY=...` (required if `LLM_PROVIDER=openai`)
- `OPENAI_MODEL=gpt-4o-mini`
- `TOTAL_CHUNK_CONTEXT=6`
- `LOGGING_ENABLED=false`

When running with the Ollama profile, the compose file sets:

- `OLLAMA_HOST=http://ollama:11434`

### Using a different vectorstore

By default the image includes `vectorstore.zip` which is extracted to `/app/vectorstore`.

To use your own:

- Replace `vectorstore.zip` before building, or
- Mount a host directory:

```
docker compose run --rm \
  -v "$(pwd)/my-vectorstore:/app/vectorstore" \
  rag-ws
```

Or edit `VECTORSTORE_DIR` in `.env` and mount accordingly.

### Local development (without Docker)

- Install Python deps:

```
pip install -r requirements.txt
```

- Ensure your `.env` is set (matching the above variables).
- Start server:

```
python ws_server.py
```

Server runs at `http://localhost:8000` and `ws://localhost:8000/ws`.

### Security

- Adjust CORS via `CORS_ALLOW_ORIGINS` env (comma-separated list). Default is `*` for development.
- For production, put this service behind TLS termination (reverse proxy) and restrict origins.
