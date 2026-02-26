import os
import glob
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import docx

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "data"
DB_DIR = "db"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_docx_files(data_dir: str) -> List[Document]:
    """
    Scans the data directory for .docx files and loads their content.
    """
    documents = []
    docx_files = glob.glob(os.path.join(data_dir, "*.docx"))

    if not docx_files:
        print(f"No .docx files found in {data_dir}")
        return []

    print(f"Found {len(docx_files)} documents: {docx_files}")

    for file_path in docx_files:
        try:
            print(f"Loading {file_path}...")
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            text = "\n".join(full_text)

            if text.strip():
                # Store filename in metadata
                filename = os.path.basename(file_path)
                documents.append(Document(page_content=text, metadata={"source": filename}))
                print(f"Loaded {len(text)} characters from {filename}")
            else:
                print(f"Warning: {file_path} is empty.")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a configured text splitter for legal documents.
    """
    # Smart chunking for legal texts
    # separators order matters: try to split by article/chapter first, then paragraphs, then sentences
    separators = ["\nСтатья ", "\nГлава ", "\n\n", "\n", " ", ""]

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        is_separator_regex=False
    )

def create_vector_store(documents: List[Document]):
    """
    Splits documents and creates a Chroma vector store.
    """
    if not documents:
        print("No documents to index.")
        return

    text_splitter = get_text_splitter()

    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")

    # Initialize Embeddings
    print(f"Initializing embeddings model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create Chroma DB
    print(f"Creating Chroma vector store in {DB_DIR}...")
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print("Vector store created and persisted.")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Please put .docx files there.")
        return

    documents = load_docx_files(DATA_DIR)
    create_vector_store(documents)

if __name__ == "__main__":
    main()
