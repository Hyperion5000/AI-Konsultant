import pytest
import os
from unittest.mock import MagicMock, patch, mock_open
from langchain_core.documents import Document
from create_index import load_documents, get_text_splitter, create_vector_store

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

    with patch.dict("sys.modules", {"fitz": mock_fitz_module}):
        with patch("docx.Document", return_value=mock_doc_instance) as mock_docx_cls:
            # Mock glob to return a specific file path
            # Sequence:
            # 1. docx
            # 2. txt
            # 3. md
            # 4. pdf

            with patch("glob.glob", side_effect=[
                ["data/test.docx"],
                ["data/test.txt"],
                [], # md
                ["data/test.pdf"]
            ]):
                # Mock basename. Called for each file found.
                # Files found: test.docx, test.txt, test.pdf.
                # Order of processing: docx, txt, pdf.
                with patch("os.path.basename", side_effect=["test.docx", "test.txt", "test.pdf"]):
                    with patch("builtins.open", mock_open(read_data="Text file content")) as mock_file:

                        documents = load_documents("data")

                        # Check we got 3 documents (docx, txt, pdf)
                        assert len(documents) == 3

                        # Docx
                        assert documents[0].page_content == "Paragraph 1\nParagraph 2"
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
    # The separators attribute name might be different or private
    # RecursiveCharacterTextSplitter stores separators in _separators
    expected_separators = ["\nСтатья ", "\nГлава ", "\n\n", "\n", " ", ""]
    assert splitter._separators == expected_separators

def test_create_vector_store_logic():
    """
    Test the create_vector_store function logic (mocking Chroma and Embeddings).
    Also verify BM25 creation.
    """
    documents = [Document(page_content="Test")]

    with patch("create_index.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("create_index.Chroma") as mock_chroma, \
         patch("create_index.BM25Retriever") as mock_bm25, \
         patch("create_index.pickle.dump") as mock_pickle_dump, \
         patch("builtins.open", mock_open()) as mock_file:

        create_vector_store(documents, "test_db", "model_name", 100, 20)

        # Verify embeddings init
        mock_embeddings.assert_called_once_with(model_name="model_name")

        # Verify Chroma creation
        mock_chroma.from_documents.assert_called_once()

        # Verify BM25 creation
        mock_bm25.from_documents.assert_called_once()

        # Verify pickle dump was called (meaning we tried to save)
        # We need to ensure we opened a file for writing
        mock_file.assert_called()
        mock_pickle_dump.assert_called_once()
