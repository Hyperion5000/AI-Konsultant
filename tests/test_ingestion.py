import pytest
import os
import glob
from unittest.mock import MagicMock, patch
from create_index import load_docx_files, get_text_splitter, create_vector_store, main as create_index_main
from langchain_core.documents import Document

# Mock docx library since it might not be installed in CI or we want to avoid file I/O
class MockDocument:
    def __init__(self, file_path):
        self.paragraphs = [MagicMock(text="Paragraph 1"), MagicMock(text="Paragraph 2")]

def test_load_docx_files():
    """
    Test that load_docx_files correctly reads text and adds metadata.
    """
    # Create a mock docx.Document that returns our MockDocument structure
    mock_doc_instance = MagicMock()
    mock_para1 = MagicMock()
    mock_para1.text = "Paragraph 1"
    mock_para2 = MagicMock()
    mock_para2.text = "Paragraph 2"
    mock_doc_instance.paragraphs = [mock_para1, mock_para2]

    with patch("docx.Document", return_value=mock_doc_instance) as mock_docx_cls:
        # Mock glob to return a specific file path
        with patch("glob.glob", return_value=["data/test.docx"]):
            # Mock os.path.basename to return the filename
            with patch("os.path.basename", return_value="test.docx"):

                documents = load_docx_files("data")

                # Check that we got 1 document back
                assert len(documents) == 1

                # Check the content: paragraphs joined by newline
                assert documents[0].page_content == "Paragraph 1\nParagraph 2"

                # Check metadata source
                assert documents[0].metadata["source"] == "test.docx"

                # Verify docx.Document was called with the correct path
                mock_docx_cls.assert_called_with("data/test.docx")

def test_text_splitter_configuration():
    """
    Test that the text splitter is configured with the correct separators for legal documents.
    """
    splitter = get_text_splitter()

    # Expected separators from the requirements
    expected_separators = ["\nСтатья ", "\nГлава ", "\n\n", "\n", " ", ""]

    assert splitter._separators == expected_separators

def test_text_splitter_chunking():
    """
    Critical Test: Ensure text is split effectively.
    """
    splitter = get_text_splitter()
    # Force a smaller chunk size to trigger splitting easily for this test
    splitter._chunk_size = 100
    splitter._chunk_overlap = 0

    # Construct a text where:
    # Chunk 1: "Глава 1. ..." (~60 chars)
    # Chunk 2: "Статья 1. ..." (~60 chars)

    part1 = "Глава 1. Введение в закон о тестировании." # ~40 chars
    part2 = "Статья 1. Основные положения по модульным тестам." # ~50 chars

    # Pad them to force split
    part1 += " " * (80 - len(part1)) # Pad to 80 chars
    part2 += " " * (80 - len(part2)) # Pad to 80 chars

    full_text = f"{part1}\n{part2}"

    chunks = splitter.split_text(full_text)

    # Should be at least 2 chunks
    assert len(chunks) >= 2

    # The first chunk should contain the Chapter
    assert "Глава 1" in chunks[0]

    # The second chunk should start with "Статья 1" (stripped) or contain it cleanly
    assert "Статья 1" in chunks[1]

def test_create_vector_store_logic():
    """
    Test the create_vector_store function logic (mocking Chroma and Embeddings).
    """
    documents = [Document(page_content="Test")]

    with patch("create_index.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("create_index.Chroma") as mock_chroma, \
         patch("create_index.RecursiveCharacterTextSplitter") as mock_splitter_cls:

        # Setup mock splitter instance
        mock_splitter = mock_splitter_cls.return_value
        mock_splitter.split_documents.return_value = documents # Return same docs for simplicity

        # We need to ensure get_text_splitter is used, but if we mock RecursiveCharacterTextSplitter class,
        # get_text_splitter will return the mock instance.

        create_vector_store(documents)

        # Verify embeddings init
        mock_embeddings.assert_called_once()

        # Verify Chroma creation
        mock_chroma.from_documents.assert_called_once()

def test_main_execution():
    """
    Test the main function flow.
    """
    with patch("create_index.os.path.exists", return_value=True), \
         patch("create_index.load_docx_files", return_value=[]) as mock_load, \
         patch("create_index.create_vector_store") as mock_create:

        create_index_main()

        mock_load.assert_called_once()
        mock_create.assert_called_once()

    # Test directory creation if not exists
    with patch("create_index.os.path.exists", return_value=False), \
         patch("create_index.os.makedirs") as mock_makedirs:

        create_index_main()
        mock_makedirs.assert_called_once()
