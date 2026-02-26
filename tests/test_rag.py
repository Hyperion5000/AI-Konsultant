import pytest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document
from bot.rag_service import RAGService
from bot import config

@pytest.fixture
def mock_rag_dependencies():
    # We patch classes and os.path.exists
    with patch("bot.rag_service.Chroma") as MockChroma, \
         patch("bot.rag_service.HuggingFaceEmbeddings") as MockEmbeddings, \
         patch("bot.rag_service.ChatOllama") as MockOllama, \
         patch("bot.rag_service.EnsembleRetriever") as MockEnsemble, \
         patch("bot.rag_service.pickle.load") as MockPickleLoad, \
         patch("bot.rag_service.HuggingFaceCrossEncoder") as MockHFCrossEncoder, \
         patch("bot.rag_service.CrossEncoderReranker") as MockCrossEncoderReranker, \
         patch("bot.rag_service.ContextualCompressionRetriever") as MockCompressionRetriever, \
         patch("os.path.exists", return_value=True) as MockExists, \
         patch("builtins.open", new_callable=MagicMock): # Mock open for pickle loading

        mock_vector_store = MagicMock()
        MockChroma.return_value = mock_vector_store

        # Mock as_retriever
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever

        mock_llm = MagicMock()
        MockOllama.return_value = mock_llm

        mock_ensemble_instance = MagicMock()
        MockEnsemble.return_value = mock_ensemble_instance

        MockPickleLoad.return_value = MagicMock() # BM25 retriever mock

        mock_compression_retriever_instance = MagicMock()
        MockCompressionRetriever.return_value = mock_compression_retriever_instance

        yield mock_vector_store, mock_llm, MockEnsemble, mock_ensemble_instance, \
              MockHFCrossEncoder, MockCrossEncoderReranker, MockCompressionRetriever, \
              mock_compression_retriever_instance, MockExists

@pytest.mark.asyncio
async def test_reset_history(mock_rag_dependencies):
    service = RAGService()
    user_id = 123
    service.history[user_id] = ["msg1", "msg2"]

    service.reset_history(user_id)
    assert user_id not in service.history

@pytest.mark.asyncio
async def test_get_answer_sources_prompt_compliance(mock_rag_dependencies):
    mock_vector_store, mock_llm, MockEnsemble, mock_ensemble_instance, \
    MockHFCrossEncoder, MockCrossEncoderReranker, MockCompressionRetriever, \
    mock_compression_retriever_instance, _ = mock_rag_dependencies

    # Setup mock return value for retrieve
    doc1 = Document(page_content="content1", metadata={"source": "doc1.txt", "chunk_id": 1})
    doc2 = Document(page_content="content2", metadata={"source": "doc2.txt", "chunk_id": 2, "title": "Title 2"})

    # Mock compression retriever ainvoke (since it wraps others)
    mock_compression_retriever_instance.ainvoke = AsyncMock(return_value=[doc1, doc2])

    # Mock LLM stream to simulate structured response with sources
    async def mock_stream(messages):
        yield MagicMock(content="1. **Краткий вывод**: Да.\n")
        yield MagicMock(content="2. **Обоснование**: Потому что.\n")
        yield MagicMock(content="3. **Источники**: doc1.txt, doc2.txt")
    mock_llm.astream = mock_stream

    service = RAGService()

    # Run
    responses = []
    async for chunk in service.get_answer(123, "question"):
        responses.append(chunk)

    full_response = "".join(responses)

    # Check sources presence (from LLM)
    assert "**Источники**" in full_response
    assert "doc1.txt" in full_response

    # Check that manual legacy appending is GONE
    # Previous manual format was "**Основания:**"
    assert "**Основания:**" not in full_response

@pytest.mark.asyncio
async def test_initialization_full_pipeline(mock_rag_dependencies):
    mock_vector_store, mock_llm, MockEnsemble, mock_ensemble_instance, \
    MockHFCrossEncoder, MockCrossEncoderReranker, MockCompressionRetriever, \
    mock_compression_retriever_instance, _ = mock_rag_dependencies

    # RAGService should init embeddings, Chroma, BM25(Ensemble), CrossEncoder, ContextualCompression
    service = RAGService()

    # Check Ensemble
    assert MockEnsemble.called

    # Check Reranker
    assert MockHFCrossEncoder.called
    assert MockCrossEncoderReranker.called
    assert MockCompressionRetriever.called

    assert service.retriever == mock_compression_retriever_instance

@pytest.mark.asyncio
async def test_initialization_reranker_failure(mock_rag_dependencies):
    mock_vector_store, mock_llm, MockEnsemble, mock_ensemble_instance, \
    MockHFCrossEncoder, MockCrossEncoderReranker, MockCompressionRetriever, \
    mock_compression_retriever_instance, _ = mock_rag_dependencies

    # Simulate Reranker failure
    MockHFCrossEncoder.side_effect = Exception("Model not found")

    service = RAGService()

    # Should fallback to base retriever (Ensemble or Chroma)
    # Since BM25 exists (mocked), it should be Ensemble
    assert service.retriever == mock_ensemble_instance
