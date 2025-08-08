"""
RAG Pipeline Example with Ollama, LangChain, FAISS, and SentenceTransformers

Requirements:
    pip install langchain faiss-cpu sentence-transformers pypdf ollama

- Ollama must be running locally with a supported model pulled (e.g., llama3, gemma, etc.)
- Place your PDF as 'example.pdf' in the same directory or change the filename below.
"""

import os
import sys

# 1. PDF Loading
try:
    from langchain.document_loaders import PyPDFLoader
except ImportError:
    print("Missing langchain. Install with: pip install langchain")
    sys.exit(1)

PDF_PATH = "example.pdf"
if not os.path.exists(PDF_PATH):
    print(f"PDF file '{PDF_PATH}' not found. Please add your PDF to the project directory.")
    sys.exit(1)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2. Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embeddings (local, using SentenceTransformers)
try:
    from langchain.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("Missing langchain or sentence-transformers. Install with: pip install langchain sentence-transformers")
    sys.exit(1)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Vector DB (FAISS)
try:
    from langchain.vectorstores import FAISS
except ImportError:
    print("Missing faiss-cpu. Install with: pip install faiss-cpu")
    sys.exit(1)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

# 5. LLM (Ollama)
try:
    from langchain.chat_models import ChatOllama
except ImportError:
    print("Missing langchain or ollama. Install with: pip install langchain ollama")
    sys.exit(1)

# Make sure Ollama is running and the model is pulled, e.g.:
# ollama pull llama3
# ollama serve

OLLAMA_MODEL = "llama3.2:3b"  # Change to your preferred model, e.g., "llama3", "gemma", etc.

llm = ChatOllama(model=OLLAMA_MODEL)

# 6. RetrievalQA Chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 7. Query
query = "What are the main risk factors mentioned in the document?"
response = qa_chain.run(query)
print("Q:", query)
print("A:", response)
