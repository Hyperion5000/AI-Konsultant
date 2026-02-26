import pytest
import os
from unittest.mock import MagicMock, patch
from bot.core.resources import initialize_bot_resources
from bot import config

@pytest.fixture
def mock_dependencies():
    with patch("bot.core.resources.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("bot.core.resources.Chroma") as mock_chroma, \
         patch("bot.core.resources.ChatOllama") as mock_llm, \
         patch("bot.core.resources.create_agent_graph") as mock_create_graph, \
         patch("bot.core.resources.set_retriever") as mock_set_retriever, \
         patch("os.path.exists") as mock_exists, \
         patch("bot.core.resources.pickle.load") as mock_pickle_load, \
         patch("builtins.open") as mock_open_func:

        yield {
            "embeddings": mock_embeddings,
            "chroma": mock_chroma,
            "llm": mock_llm,
            "create_graph": mock_create_graph,
            "set_retriever": mock_set_retriever,
            "exists": mock_exists,
            "pickle_load": mock_pickle_load,
            "open": mock_open_func
        }

def test_initialize_resources_success(mock_dependencies):
    # Setup mocks
    mock_dependencies["exists"].return_value = True # DB exists
    mock_dependencies["pickle_load"].return_value = MagicMock() # BM25 loaded

    # Run
    graph = initialize_bot_resources()

    # Verify
    assert graph is not None
    mock_dependencies["create_graph"].assert_called_once()
    mock_dependencies["set_retriever"].assert_called_once()

def test_initialize_resources_no_db(mock_dependencies):
    # Setup mocks
    # config.CHROMA_DIR check returns False
    # We need to distinguish between calls to exists.
    # 1. config.CHROMA_DIR
    # 2. bm25_path

    def side_effect(path):
        if path == config.CHROMA_DIR:
            return False
        return True

    mock_dependencies["exists"].side_effect = side_effect

    # Run
    with pytest.raises(FileNotFoundError):
        initialize_bot_resources()

def test_initialize_resources_bm25_fail(mock_dependencies):
    # Setup mocks
    mock_dependencies["exists"].return_value = True
    # pickle load fails
    mock_dependencies["pickle_load"].side_effect = Exception("BM25 Error")

    # Run
    graph = initialize_bot_resources()

    # Verify we continued
    assert graph is not None
    # set_retriever should still be called (fallback to chroma)
    mock_dependencies["set_retriever"].assert_called_once()

def test_initialize_resources_reranker_fail(mock_dependencies):
    # Setup mocks
    mock_dependencies["exists"].return_value = True

    # Mock Reranker fail
    with patch("bot.core.resources.HuggingFaceCrossEncoder", side_effect=Exception("Reranker Error")):
        graph = initialize_bot_resources()

        assert graph is not None
        mock_dependencies["set_retriever"].assert_called_once()
