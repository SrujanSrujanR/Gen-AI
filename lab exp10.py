import os
import sys
import requests
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# Download the Indian Penal Code document
IPC_URL = "https://www.mha.gov.in/sites/default/files/IPAct_1860.pdf"
IPC_FILE = "Indian_Penal_Code.pdf"

def download_pdf(url, filename):
    """Downloads the PDF file and verifies its integrity."""
    print("Downloading Indian Penal Code document...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an error if the request fails
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")
    except requests.RequestException as e:
        print(f"Error downloading PDF: {e}")
        sys.exit(1)  # ✅ Proper exit using sys.exit()

if not os.path.exists(IPC_FILE):
    download_pdf(IPC_URL, IPC_FILE)

# Extract text using pdfplumber
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        sys.exit(1)  # ✅ Proper exit if extraction fails

print("Extracting text from the IPC document...")
ipc_text = extract_text_from_pdf(IPC_FILE)

# Ensure text is extracted
if not ipc_text:
    print("Error: Failed to extract text from the PDF. The file may be corrupted.")
    sys.exit(1)  # ✅ Proper script exit

# Split text into searchable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
ipc_chunks = text_splitter.split_text(ipc_text)

# Create FAISS vector store
vector_store = FAISS.from_texts(ipc_chunks, embedding=OpenAIEmbeddings())

# Load Chat Model
chat_model = ChatOpenAI(model_name="gpt-4")

# Create retrieval-based QA system
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# Chatbot Interface
def ipc_chatbot():
    print("\nIndian Penal Code Chatbot (Type 'exit' to stop)")
    while True:
        query = input("\nAsk about the Indian Penal Code: ").strip()
        if query.lower() == "exit":
            print("Exiting chatbot. Have a great day!")
            sys.exit(0)  # ✅ Proper exit when the user wants to quit
        response = qa_chain.run(query)
        print("\nIPC Chatbot: ", response)

if __name__ == "__main__":
    ipc_chatbot()
