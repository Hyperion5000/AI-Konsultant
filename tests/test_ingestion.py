import pytest
import os
import sys
from unittest.mock import MagicMock, patch, mock_open
from langchain_core.documents import Document

# Add project root to path so we can import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.ingest_data import load_documents, get_text_splitter, create_vector_store, main

def test_load_documents():
    """
    Test that load_documents correctly reads text and adds metadata.
    """
    # Mock docx.Document
    mock_doc_instance = MagicMock()
    mock_para1 = MagicMock()
    mock_para1.text = "Paragraph 1"
    mock_para2 = MagicMock()
    mock_para2.text = "Paragraph 2"
    mock_doc_instance.paragraphs = [mock_para1, mock_para2]

    # Mock fitz (PyMuPDF)
    mock_pdf_doc = MagicMock()
    # Iterate over pages
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "PDF Page 1 Text"
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "PDF Page 2 Text"
    mock_pdf_doc.__iter__.return_value = [mock_page1, mock_page2]

    mock_fitz_module = MagicMock()
    mock_fitz_module.open.return_value = mock_pdf_doc

    # We need to patch where fitz is imported, which is inside load_documents (lazy import)
    # But sys.modules works too.
    with patch.dict("sys.modules", {"fitz": mock_fitz_module}):
        with patch("docx.Document", return_value=mock_doc_instance) as mock_docx_cls:
            # Mock glob to return specific file paths
            with patch("glob.glob", side_effect=[
                ["storage/documents/test.docx"],
                ["storage/documents/test.txt"],
                [], # md
                ["storage/documents/test.pdf"]
            ]):
                # Remove os.path.basename patch.
                # Use mock_open for file reading.
                with patch("builtins.open", mock_open(read_data="Text file content")) as mock_file:

                    documents = load_documents("storage/documents")

                    # Check we got 3 documents
                    assert len(documents) == 3

                    # Docx
                    assert documents[0].page_content == "Paragraph 1\nParagraph 2"
                    # os.path.basename works on the string path
                    assert documents[0].metadata["source"] == "test.docx"
                    assert documents[0].metadata["type"] == "docx"

                    # Txt
                    assert documents[1].page_content == "Text file content"
                    assert documents[1].metadata["source"] == "test.txt"
                    assert documents[1].metadata["type"] == "txt"

                    # PDF
                    assert documents[2].page_content == "PDF Page 1 Text\nPDF Page 2 Text"
                    assert documents[2].metadata["source"] == "test.pdf"
                    assert documents[2].metadata["type"] == "pdf"

def test_text_splitter_configuration():
    """
    Test that the text splitter is configured with the correct separators.
    """
    splitter = get_text_splitter(1000, 200)
    expected_separators = ["\nСтатья ", "\nГлава ", "\n\n", "\n", " ", ""]
    assert splitter._separators == expected_separators

def test_create_vector_store_logic():
    """
    Test the create_vector_store function logic (mocking Chroma and Embeddings).
    Also verify BM25 creation.
    """
    documents = [Document(page_content="Test")]

    # Patch dependencies in scripts.ingest_data
    with patch("scripts.ingest_data.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("scripts.ingest_data.Chroma") as mock_chroma, \
         patch("scripts.ingest_data.BM25Retriever") as mock_bm25, \
         patch("scripts.ingest_data.pickle.dump") as mock_pickle_dump, \
         patch("builtins.open", mock_open()) as mock_file:

        # Mock BM25 instance
        mock_bm25_instance = MagicMock()
        mock_bm25.from_documents.return_value = mock_bm25_instance

        create_vector_store(
            documents=documents,
            chroma_dir="test_chroma",
            bm25_dir="test_bm25",
            embedding_model_name="model_name",
            chunk_size=100,
            chunk_overlap=20
        )

        # Verify embeddings init
        mock_embeddings.assert_called_once_with(model_name="model_name")

        # Verify Chroma creation
        mock_chroma.from_documents.assert_called_once()

        # Verify BM25 creation
        mock_bm25.from_documents.assert_called_once()

        # Verify pickle dump was called
        mock_file.assert_called()
        mock_pickle_dump.assert_called_once()

def test_main_script():
    with patch("scripts.ingest_data.load_documents") as mock_load, \
         patch("scripts.ingest_data.create_vector_store") as mock_create, \
         patch("sys.argv", ["script", "--input", "test_input"]), \
         patch("os.path.exists", return_value=True): # Input dir exists

        mock_load.return_value = [Document(page_content="test")]

        main()

        mock_load.assert_called_with("test_input")
        mock_create.assert_called()

def test_main_script_no_docs():
    with patch("scripts.ingest_data.load_documents") as mock_load, \
         patch("scripts.ingest_data.create_vector_store") as mock_create, \
         patch("sys.argv", ["script"]), \
         patch("os.path.exists", return_value=True):

        mock_load.return_value = []

        main()

        mock_load.assert_called()
        mock_create.assert_not_called()
