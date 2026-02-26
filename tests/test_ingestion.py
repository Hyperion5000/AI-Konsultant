import pytest
import os
from unittest.mock import MagicMock, patch
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

    with patch("docx.Document", return_value=mock_doc_instance) as mock_docx_cls:
        # Mock glob to return a specific file path
        # create_index calls glob 3 times: docx, txt, md
        with patch("glob.glob", side_effect=[["data/test.docx"], [], []]):
            # Mock os.path.basename to return the filename
            with patch("os.path.basename", return_value="test.docx"):

                documents = load_documents("data")

                # Check that we got 1 document back
                assert len(documents) == 1
                assert documents[0].page_content == "Paragraph 1\nParagraph 2"
                assert documents[0].metadata["source"] == "test.docx"
                assert documents[0].metadata["type"] == "docx"

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
    """
    documents = [Document(page_content="Test")]

    with patch("create_index.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("create_index.Chroma") as mock_chroma:

        # We need to mock RecursiveCharacterTextSplitter which is used inside create_vector_store
        # Actually create_vector_store calls get_text_splitter which returns an instance.
        # We can just let it run or mock it.
        # Let's let it run since it's fast.

        create_vector_store(documents, "test_db", "model_name", 100, 20)

        # Verify embeddings init
        mock_embeddings.assert_called_once_with(model_name="model_name")

        # Verify Chroma creation
        mock_chroma.from_documents.assert_called_once()
