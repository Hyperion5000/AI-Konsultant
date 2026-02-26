import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bot.rag_service import RAGService
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

# --- Fixtures ---

@pytest.fixture
def mock_dependencies():
    with patch("bot.rag_service.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("bot.rag_service.Chroma") as mock_chroma, \
         patch("bot.rag_service.ChatOllama") as mock_ollama, \
         patch("os.path.exists", return_value=True):

        # Setup Chroma retriever mock
        mock_retriever = AsyncMock()
        mock_retriever.ainvoke.return_value = [
            Document(page_content="Статья 1. Тестовый закон."),
            Document(page_content="Статья 2. Дополнительная информация.")
        ]

        mock_vector_store = mock_chroma.return_value
        mock_vector_store.as_retriever.return_value = mock_retriever

        # Setup LLM instance mock
        mock_llm_instance = MagicMock()
        mock_ollama.return_value = mock_llm_instance

        # Create a default async generator for astream
        async def default_astream_generator(messages):
            yield MagicMock(content="Ответ")
            yield MagicMock(content=" на")
            yield MagicMock(content=" вопрос.")

        mock_llm_instance.astream.side_effect = default_astream_generator

        yield {
            "embeddings": mock_embeddings,
            "chroma": mock_chroma,
            "retriever": mock_retriever,
            "ollama": mock_ollama,
            "llm": mock_llm_instance
        }

@pytest.fixture
def rag_service(mock_dependencies):
    return RAGService()

# --- Tests ---

@pytest.mark.asyncio
async def test_initialization_failure(mock_dependencies):
    """Test that RAGService raises FileNotFoundError if DB_DIR is missing."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            RAGService()

@pytest.mark.asyncio
async def test_get_answer_context_retrieval(rag_service, mock_dependencies):
    """Test that get_answer retrieves documents and constructs context correctly."""
    user_id = 123
    question = "Как работает закон?"

    # Run the generator
    full_response = ""
    async for chunk in rag_service.get_answer(user_id, question):
        full_response += chunk

    # Check retriever called
    mock_dependencies["retriever"].ainvoke.assert_called_once_with(question)

    # Check context in LLM call
    mock_llm = mock_dependencies["llm"]
    args, _ = mock_llm.astream.call_args
    messages = args[0]

    # The last message is the HumanMessage with context + question
    last_message = messages[-1]
    assert isinstance(last_message, HumanMessage)
    assert "Текст закона:\nСтатья 1. Тестовый закон.\n\nСтатья 2. Дополнительная информация." in last_message.content
    assert f"Вопрос пользователя: {question}" in last_message.content

@pytest.mark.asyncio
async def test_history_management(rag_service, mock_dependencies):
    """Test that history is trimmed to strictly 6 messages (3 pairs)."""
    user_id = 456

    # Simulate 4 interactions
    for i in range(4):
        question = f"Question {i}"

        # Mock LLM response to vary slightly if needed, or use default
        async for _ in rag_service.get_answer(user_id, question):
            pass

    # History should contain only the last 3 pairs (6 messages)
    assert len(rag_service.history[user_id]) == 6

    # Check order: Q1, A1, Q2, A2, Q3, A3. Since we did 0, 1, 2, 3...
    # The history should store: Q1, A1, Q2, A2, Q3, A3 (where 3 is the last one)
    # Wait, history stores the original question and the full answer.

    # 1st interaction (i=0): Hist: [Q0, A0]
    # 2nd interaction (i=1): Hist: [Q0, A0, Q1, A1]
    # 3rd interaction (i=2): Hist: [Q0, A0, Q1, A1, Q2, A2]
    # 4th interaction (i=3): Hist: [Q1, A1, Q2, A2, Q3, A3] (trimmed)

    assert rag_service.history[user_id][0].content == "Question 1"
    assert rag_service.history[user_id][-2].content == "Question 3"
    assert isinstance(rag_service.history[user_id][0], HumanMessage)
    assert isinstance(rag_service.history[user_id][1], AIMessage)

@pytest.mark.asyncio
async def test_system_prompt_inclusion(rag_service, mock_dependencies):
    """Test that the SystemMessage is always added as the first message."""
    user_id = 789
    question = "Test system prompt"

    async for _ in rag_service.get_answer(user_id, question):
        pass

    mock_llm = mock_dependencies["llm"]
    args, _ = mock_llm.astream.call_args
    messages = args[0]

    assert isinstance(messages[0], SystemMessage)
    assert "Ты — опытный юрист-консультант" in messages[0].content

@pytest.mark.asyncio
async def test_streaming_output(rag_service, mock_dependencies):
    """Test that get_answer yields chunks correctly from the LLM."""
    user_id = 101
    question = "Stream check"

    chunks = []
    async for chunk in rag_service.get_answer(user_id, question):
        chunks.append(chunk)

    assert "".join(chunks) == "Ответ на вопрос."
