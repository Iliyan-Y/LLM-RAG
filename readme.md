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

Notes:

- With `LLM_PROVIDER=ollama`, the app uses your local Ollama server and `OLLAMA_MODEL`.
- With `LLM_PROVIDER=openai`, the app uses OpenAI via `OPENAI_API_KEY` and `OPENAI_MODEL`.

## How to use

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
