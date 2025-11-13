# ==========================================
# 📘 FAISS-based File Querying Example
# ==========================================

import faiss
from os import environ
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from PyPDF2 import PdfReader
load_dotenv()

# ------------------------------------------
# 1️⃣ Load API key (make sure OPENAI_API_KEY is set in .env)
# ------------------------------------------
load_dotenv()

# ------------------------------------------
# 2️⃣ Initialize Embeddings + FAISS index
# ------------------------------------------
embeddings = OpenAIEmbeddings(model = environ.get("EMBEDDING_MODEL_NAME", ""), 
                              api_key = environ.get("OPENAI_API_KEY", ""), 
                              dimensions = 64, 
                              timeout = 300)

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))  
vector_store = FAISS(embedding_function = embeddings, 
                     index = index, 
                     docstore = InMemoryDocstore(), 
                     index_to_docstore_id = {})

# ------------------------------------------
# 3️⃣ Helper: Extract and chunk text from a file
# ------------------------------------------
def extract_text_from_file(file_path):
    """Supports .txt and .pdf files"""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        pdf = PdfReader(file_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text
    else:
        raise ValueError("Unsupported file type. Please upload .txt or .pdf")

def chunk_text(text, chunk_size=800, overlap=100):
    """Split text into manageable Document chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n", ".", "?", "!"]
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks]

# ------------------------------------------
# 4️⃣ Add documents to FAISS
# ------------------------------------------
def index_file_in_faiss(file_path):
    """Read file → chunk → add to FAISS"""
    text = extract_text_from_file(file_path)
    docs = chunk_text(text)

    # Create unique IDs for each document
    ids = [str(i) for i in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=ids)
    print(f"✅ Indexed {len(docs)} chunks from {file_path}")

# ------------------------------------------
# 5️⃣ Query FAISS
# ------------------------------------------
def query_faiss(query, k=3):
    """Perform semantic search over stored docs"""
    results = vector_store.similarity_search(query=query, k=k)
    print("\n🔍 Top Matches:")
    for i, doc in enumerate(results, start=1):
        print(f"{i}. {doc.page_content[:200]}...\n")

# ==========================================
# ✅ Example Run
# ==========================================

# Suppose user uploaded a file (for demo, replace with your file path)
sample_file = "/home/siddharth/Downloads/test_file.pdf"   # or "sample.txt"

index_file_in_faiss(sample_file)

# Now user asks a query
query = "Summaries the uploaded file"
query_faiss(query)
