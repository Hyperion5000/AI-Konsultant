import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bot.rag_service import RAGService
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

@pytest.fixture
def mock_embeddings():
    with patch("bot.rag_service.HuggingFaceEmbeddings") as mock:
        yield mock

@pytest.fixture
def mock_chroma():
    with patch("bot.rag_service.Chroma") as mock:
        # Mock retriever
        retriever = AsyncMock()
        retriever.ainvoke.return_value = [
            MagicMock(page_content="Article 1: Law text"),
            MagicMock(page_content="Article 2: Another law text")
        ]
        mock_instance = mock.return_value
        mock_instance.as_retriever.return_value = retriever
        yield mock_instance

@pytest.fixture
def mock_chat_ollama():
    with patch("bot.rag_service.ChatOllama") as mock:
        mock_instance = mock.return_value
        # Mock astream to yield chunks
        async def mock_astream(messages):
            yield MagicMock(content="Hello")
            yield MagicMock(content=" World")
        mock_instance.astream = mock_astream
        yield mock_instance

@pytest.mark.asyncio
async def test_rag_service_initialization(mock_embeddings, mock_chroma, mock_chat_ollama):
    rag = RAGService()
    assert rag.llm is not None
    assert rag.retriever is not None
    assert rag.history == {}

@pytest.mark.asyncio
async def test_get_answer_flow(mock_embeddings, mock_chroma, mock_chat_ollama):
    rag = RAGService()

    # Run get_answer
    responses = []
    async for chunk in rag.get_answer(user_id=123, question="What is the law?"):
        responses.append(chunk)

    assert "".join(responses) == "Hello World"

    # Check retriever called
    rag.retriever.ainvoke.assert_called_once_with("What is the law?")

    # Check history updated
    assert 123 in rag.history
    assert len(rag.history[123]) == 2 # Question + Answer
    assert isinstance(rag.history[123][0], HumanMessage)
    assert rag.history[123][0].content == "What is the law?"
    assert isinstance(rag.history[123][1], AIMessage)
    assert rag.history[123][1].content == "Hello World"

@pytest.mark.asyncio
async def test_history_trimming(mock_embeddings, mock_chroma, mock_chat_ollama):
    rag = RAGService()
    user_id = 456

    # Fill history with 3 pairs (6 messages)
    rag.history[user_id] = [HumanMessage(content=f"Q{i}") for i in range(6)]

    # Ask another question
    async for _ in rag.get_answer(user_id=user_id, question="New Question"):
        pass

    # Check history length (should be 6: old ones removed, new added)
    # Wait, my logic:
    # 1. Start with 6
    # 2. Add Q (7)
    # 3. Add A (8)
    # 4. Trim to last 6
    assert len(rag.history[user_id]) == 6
    assert rag.history[user_id][-2].content == "New Question"
    assert rag.history[user_id][-1].content == "Hello World"
