# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY

# 1) Load documents from ./data
def load_documents(data_dir="data"):
    docs = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            path = os.path.join(root, f)
            if f.lower().endswith(".txt"):
                docs.extend(TextLoader(path, encoding="utf-8").load())
            elif f.lower().endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
    return docs

# 2) Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# 3) Build embeddings and persist to Chroma
def build_vectorstore(chunks, persist_dir="chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)
    vs.persist()
    return vs

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents("data")
    if not docs:
        raise RuntimeError("No documents found in ./data. Add .txt or .pdf files first.")
    print(f"Loaded {len(docs)} documents. Splitting...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks. Building vectorstore...")
    vs = build_vectorstore(chunks)
    print("Vectorstore persisted to ./chroma_db")
