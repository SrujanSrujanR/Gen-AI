import os
import re
import sys
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def offline_answer(query: str, split_docs, top_k: int = 3) -> str:
    """Return a simple extractive answer without calling any external API."""
    query_tokens = Counter(_tokenize(query))
    scored = []
    for doc in split_docs:
        content = doc.page_content or ""
        if not content.strip():
            continue
        doc_tokens = Counter(_tokenize(content))
        score = sum((query_tokens & doc_tokens).values())
        scored.append((score, content.strip()))

    top_chunks = [chunk for score, chunk in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k] if score > 0]
    if not top_chunks and scored:
        top_chunks = [sorted(scored, key=lambda x: len(x[1]), reverse=True)[0][1]]

    if not top_chunks:
        return "Could not extract content from the PDF."

    joined = "\n\n".join(top_chunks)
    return joined[:900]

# Resolve runtime inputs (PDF path + API key) before building the chain.
pdf_path = Path("sample.pdf")
if not pdf_path.exists():
    available_pdfs = sorted(Path(".").glob("*.pdf"))
    if available_pdfs:
        pdf_path = available_pdfs[0]
    else:
        print("No PDF found. Add a PDF file (e.g., sample.pdf) in this folder and run again.")
        sys.exit(1)

api_key = os.getenv("GOOGLE_API_KEY", "").strip()

# Step 1: Load PDF
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Step 8: Query
query = "What is the main topic of the document?"

if not api_key:
    print("GOOGLE_API_KEY not found. Running in offline mode (no API calls).")
    print("Answer:", offline_answer(query, docs))
    sys.exit(0)

# Step 3: Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)

# Step 4: Store in FAISS
vector_store = FAISS.from_documents(docs, embeddings)

# Step 5: Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 6: LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0,
)

# Step 7: RAG Pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
)

response = qa_chain.invoke({"query": query})
print("Answer:", response["result"])