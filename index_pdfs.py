import os
import sys

from glob import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from util.indexing import detect_section_from_text, extract_sec_metadata

load_dotenv()


PDF_DIR = os.getenv('PDF_DIR')
VECTORSTORE_DIR = os.getenv('VECTORSTORE_DIR')
SEMANTIC_MODEL_NAME = os.getenv('SEMANTIC_MODEL_NAME')#all-mpnet-base-v2" || intfloat/e5-base-v2

CHUNK_SIZE = 450 * 4 # 350 characters * 4 Approx for English finance text  
CHUNK_OVERLAP = 80 * 4


# Check PDF dir
if not os.path.exists(PDF_DIR):
    print(f"PDF directory '{PDF_DIR}' not found. Please create it and add your PDFs.")
    sys.exit(1)

pdf_files = glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True)
if not pdf_files:
    print(f"No PDF files found in '{PDF_DIR}'. Please add PDFs to index.")
    sys.exit(1)

all_chunks = []

# Process PDFs
for pdf_path in pdf_files:
    print(f"Loading: {pdf_path}")

    dir_name = os.path.basename(os.path.dirname(pdf_path))
    file_name = os.path.basename(pdf_path)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Extract sample text from the first few pages for metadata extraction
    head_pages = min(5, len(docs))
    sample_text = "\n".join(d.page_content for d in docs[:head_pages])

    meta_common = extract_sec_metadata(sample_text, file_name, dir_name)

    # Chunking tuned for SEC structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Attach metadata to each chunk
    for i, chunk in enumerate(chunks):
        # Preserve any existing metadata from loader
        base_meta = chunk.metadata.copy() if isinstance(chunk.metadata, dict) else {}
        base_meta.update({
            "creator": "",
            "producer": "",
            "source": file_name,
            "author": meta_common.get("company"),
            "title": meta_common.get("filing_type"),
            "category": dir_name,
            "company": meta_common.get("company"),
            "filing_type": meta_common.get("filing_type"),
            "period_end_date": meta_common.get("period_end_date"),
            "year": meta_common.get("year"),
            "quarter": meta_common.get("quarter"),
            "section": detect_section_from_text(chunk.page_content),
            # Todo use multi model for table detection: "doc_type": "table" if is_table(chunk.page_content) else "text",
            # todo add dedup: "content_hash": hashlib.md5(chunk.page_content.encode()).hexdigest()
            "chunk_index": i,
        })
        chunk.metadata = base_meta
        all_chunks.append(chunk)

print(f"Total chunks: {len(all_chunks)}")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=SEMANTIC_MODEL_NAME)

# Vector DB (FAISS)
print("Building vector DB...")
db = FAISS.from_documents(all_chunks, embeddings)

# Save vector DB
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
db.save_local(VECTORSTORE_DIR)

print(f"Vector DB saved to '{VECTORSTORE_DIR}'")
os.system('afplay /System/Library/Sounds/Hero.aiff')