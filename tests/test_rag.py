import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from langchain_core.documents import Document
from bot.rag_service import RAGService

@pytest.fixture
def mock_dependencies():
    with patch("bot.rag_service.Chroma") as MockChroma, \
         patch("bot.rag_service.HuggingFaceEmbeddings") as MockEmbeddings, \
         patch("bot.rag_service.ChatOllama") as MockOllama, \
         patch("bot.rag_service.EnsembleRetriever") as MockEnsemble, \
         patch("bot.rag_service.create_agent_graph") as MockCreateGraph, \
         patch("bot.rag_service.set_retriever") as MockSetRetriever, \
         patch("bot.rag_service.pickle.load") as MockPickle, \
         patch("bot.rag_service.HuggingFaceCrossEncoder") as MockCrossEncoder, \
         patch("bot.rag_service.CrossEncoderReranker") as MockReranker, \
         patch("bot.rag_service.ContextualCompressionRetriever") as MockCompression, \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", new_callable=MagicMock):

        mock_retriever = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        MockChroma.return_value = mock_vector_store

        mock_llm = MagicMock()
        MockOllama.return_value = mock_llm

        mock_graph = MagicMock()
        MockCreateGraph.return_value = mock_graph

        yield mock_llm, mock_graph, MockCreateGraph

@pytest.mark.asyncio
async def test_initialization(mock_dependencies):
    mock_llm, mock_graph, MockCreateGraph = mock_dependencies
    service = RAGService()

    assert MockCreateGraph.called
    assert service.graph == mock_graph
    assert hasattr(service, "llm_semaphore")

@pytest.mark.asyncio
async def test_initialization_no_db():
    with patch("os.path.exists", return_value=False), \
         patch("bot.rag_service.HuggingFaceEmbeddings"):
        with pytest.raises(FileNotFoundError):
            RAGService()

@pytest.mark.asyncio
async def test_initialization_bm25_fail(mock_dependencies):
    # Simulate BM25 file exists but pickle load fails
    with patch("os.path.exists", side_effect=lambda x: True), \
         patch("bot.rag_service.pickle.load", side_effect=Exception("Pickle Error")), \
         patch("bot.rag_service.HuggingFaceEmbeddings"), \
         patch("bot.rag_service.Chroma") as MockChroma:

        MockChroma.return_value.as_retriever.return_value = MagicMock()

        # Should catch error and proceed with base retriever
        service = RAGService()
        assert service.retriever is not None

@pytest.mark.asyncio
async def test_initialization_reranker_fail(mock_dependencies):
    with patch("bot.rag_service.HuggingFaceCrossEncoder", side_effect=Exception("Model Error")):
         service = RAGService()
         assert service.retriever is not None # fallback to base

@pytest.mark.asyncio
async def test_reset_history(mock_dependencies):
    mock_llm, mock_graph, MockCreateGraph = mock_dependencies
    service = RAGService()
    service.history[123] = ["msg"]
    service.reset_history(123)
    assert 123 not in service.history

@pytest.mark.asyncio
async def test_get_answer_flow(mock_dependencies):
    mock_llm, mock_graph, MockCreateGraph = mock_dependencies
    service = RAGService()

    # Mock graph.astream_events to yield some events
    async def mock_astream_events(input_state, version):
        # 1. Tool start
        yield {
            "event": "on_tool_start",
            "name": "search_laws",
            "data": {}
        }
        # 2. LLM streaming tokens
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content="Hello")}
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": AIMessageChunk(content=" World")}
        }

    mock_graph.astream_events = mock_astream_events

    responses = []
    async for chunk in service.get_answer(123, "question"):
        responses.append(chunk)

    full_response = "".join(responses)

    # Check if tool feedback was yielded
    assert "⏳ Выполняю запрос к инструментам..." in full_response
    # Check if LLM content was yielded
    assert "Hello World" in full_response

    # Check history update
    assert len(service.history[123]) == 2
    assert service.history[123][1].content == "Hello World"

@pytest.mark.asyncio
async def test_get_answer_error_handling(mock_dependencies):
    mock_llm, mock_graph, MockCreateGraph = mock_dependencies
    service = RAGService()

    # Mock graph to raise exception
    mock_graph.astream_events.side_effect = Exception("Graph Error")

    responses = []
    async for chunk in service.get_answer(123, "question"):
        responses.append(chunk)

    assert "Произошла ошибка" in "".join(responses)
