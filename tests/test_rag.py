import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document
from bot.rag_service import RAGService
from bot import config

@pytest.fixture
def mock_rag_dependencies():
    with patch("bot.rag_service.Chroma") as MockChroma, \
         patch("bot.rag_service.HuggingFaceEmbeddings") as MockEmbeddings, \
         patch("bot.rag_service.ChatOllama") as MockOllama, \
         patch("os.path.exists", return_value=True):

        mock_vector_store = MagicMock()
        MockChroma.return_value = mock_vector_store

        mock_llm = MagicMock()
        MockOllama.return_value = mock_llm

        yield mock_vector_store, mock_llm

@pytest.mark.asyncio
async def test_reset_history(mock_rag_dependencies):
    service = RAGService()
    user_id = 123
    service.history[user_id] = ["msg1", "msg2"]

    service.reset_history(user_id)
    assert user_id not in service.history

@pytest.mark.asyncio
async def test_get_answer_sources_format(mock_rag_dependencies):
    mock_vector_store, mock_llm = mock_rag_dependencies
    service = RAGService()

    # Mock retrieval results
    doc1 = Document(page_content="content1", metadata={"source": "doc1.txt", "chunk_id": 1})
    doc2 = Document(page_content="content2", metadata={"source": "doc2.txt", "chunk_id": 2, "title": "Title 2"})

    # asimilarity_search_with_score returns List[Tuple[Document, float]]
    mock_vector_store.asimilarity_search_with_score = AsyncMock(return_value=[(doc1, 0.1), (doc2, 0.2)])

    # Mock LLM stream
    async def mock_stream(messages):
        yield MagicMock(content="Answer part 1")
        yield MagicMock(content=" Answer part 2")

    mock_llm.astream = mock_stream

    # Run
    responses = []
    async for chunk in service.get_answer(123, "question"):
        responses.append(chunk)

    full_response = "".join(responses)

    # Check sources formatting
    assert "**Основания:**" in full_response
    assert "- doc1.txt (chunk 1)" in full_response
    assert "- Title 2 — doc2.txt (chunk 2)" in full_response

@pytest.mark.asyncio
async def test_get_answer_low_relevance(mock_rag_dependencies, monkeypatch):
    # Enable MAX_DISTANCE check
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.5)

    mock_vector_store, _ = mock_rag_dependencies
    service = RAGService()

    # Mock retrieval with high distance (bad match)
    doc = Document(page_content="irrelevant", metadata={"source": "doc.txt"})
    mock_vector_store.asimilarity_search_with_score = AsyncMock(return_value=[(doc, 0.8)]) # 0.8 > 0.5

    responses = []
    async for chunk in service.get_answer(123, "question"):
        responses.append(chunk)

    full_response = "".join(responses)
    assert "В базе не найдено достаточно релевантных оснований" in full_response
