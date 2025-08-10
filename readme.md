# Basic proof of concept for RAG system

System that will use LLM and vectorDB to generate responses based on custom dataset of PDFs

- The retrieval and indexing are oriented to parse financial documents but can be modified by changing the keywords and
  llm actor scripts

### Pre req

- python & ollama
- in queary_rag add your model name e.g. OLLAMA_MODEL = "gemma3:latest"
- create .env file in the root dir

```
VECTORSTORE_DIR=./vectorstore
OLLAMA_MODEL=gemma3:latest
SEMANTIC_MODEL_NAME=intfloat/e5-base-v2
PDF_DIR=./pdfs
```

## Hot to

- In pdfs folder (create one) add all the pdf files you want to store in the vector storage
- create virtual env and install all the dependencies
- run the index_pdfs script
- run the queary_rag to start the chat

```
source .venv/bin/activate
pip install langchain faiss-cpu sentence-transformers pypdf ollama python-dotenv
python index_pdfs.py
python query_rag.py
```

- to disconnect from the venv just type `deactivate`
