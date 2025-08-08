"""
Index PDFs for RAG: Chunk, embed, and store in FAISS vector DB.

Usage:
    python3 index_pdfs.py

Requirements:
    pip install langchain faiss-cpu sentence-transformers pypdf

- Place all your PDFs in the ./pdfs/ directory.
- The vector DB will be saved to ./vectorstore/
"""

import os
import sys
from glob import glob
from langchain.document_loaders import PyPDFLoader

PDF_DIR = "./pdfs"
VECTORSTORE_DIR = "./vectorstore"


# 1. Load all PDFs
# try:
#     from langchain.document_loaders import PyPDFLoader
# except ImportError:
#     print("Missing langchain. Install with: pip install langchain")
#     sys.exit(1)

if not os.path.exists(PDF_DIR):
    print(f"PDF directory '{PDF_DIR}' not found. Please create it and add your PDFs.")
    sys.exit(1)

pdf_files = glob(os.path.join(PDF_DIR, "*.pdf"))
if not pdf_files:
    print(f"No PDF files found in '{PDF_DIR}'. Please add PDFs to index.")
    sys.exit(1)

all_docs = []
for pdf_path in pdf_files:
    print(f"Loading: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    all_docs.extend(docs)

# 2. Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

# 3. Embeddings
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

# 5. Save vector DB
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
db.save_local(VECTORSTORE_DIR)
print(f"Vector DB saved to '{VECTORSTORE_DIR}'")