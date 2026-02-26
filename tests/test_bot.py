import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from bot.handlers.base import command_reset_handler, command_start_handler
from bot.handlers.chat import process_question, handle_message
from aiogram.types import Message, User
from langgraph.graph.state import CompiledStateGraph

@pytest.fixture
def mock_message():
    # Create Message mock
    message = MagicMock(spec=Message)

    # Mock User
    user = MagicMock(spec=User)
    user.id = 123
    user.full_name = "Test User"
    message.from_user = user

    message.text = "Question"

    # Mock answer method as AsyncMock
    # We want to capture the returned message mock to inspect edit_text calls
    last_msg_mock = MagicMock(spec=Message)
    last_msg_mock.edit_text = AsyncMock()

    async def answer_side_effect(*args, **kwargs):
        return last_msg_mock

    message.answer = AsyncMock(side_effect=answer_side_effect)
    # Attach last_msg_mock to message for inspection in tests
    message.last_msg_mock = last_msg_mock
    return message

@pytest.fixture
def mock_graph():
    graph = MagicMock(spec=CompiledStateGraph)
    return graph

@pytest.fixture
def llm_lock():
    return asyncio.Lock()

@pytest.mark.asyncio
async def test_command_start(mock_message):
    await command_start_handler(mock_message)
    assert mock_message.answer.called
    args = mock_message.answer.call_args[0]
    assert "Привет" in args[0]

@pytest.mark.asyncio
async def test_command_reset(mock_message):
    # Mock clear_chat_history where it is imported in handlers/base.py
    with patch("bot.handlers.base.clear_chat_history", new_callable=AsyncMock) as mock_clear:
        await command_reset_handler(mock_message)
        mock_clear.assert_called_once_with(123)
        assert mock_message.answer.called
        assert "очищена" in mock_message.answer.call_args[0][0]

@pytest.mark.asyncio
async def test_handle_message_no_graph(mock_message, llm_lock):
    await handle_message(mock_message, None, llm_lock)
    assert mock_message.answer.called
    assert "инициализируется" in mock_message.answer.call_args[0][0]

@pytest.mark.asyncio
async def test_handle_message_calls_process(mock_message, mock_graph, llm_lock):
    with patch("bot.handlers.chat.process_question", new_callable=AsyncMock) as mock_process:
        await handle_message(mock_message, mock_graph, llm_lock)
        mock_process.assert_called_once()

@pytest.mark.asyncio
async def test_process_question(mock_message, mock_graph, llm_lock):
    # Async generator for graph events
    async def event_generator(*args, **kwargs):
        # Yield tool start
        yield {"event": "on_tool_start", "data": {}}

        # Yield LLM chunk
        chunk = MagicMock()
        chunk.content = "Hello"
        yield {"event": "on_chat_model_stream", "data": {"chunk": chunk}}

        # Yield another chunk
        chunk2 = MagicMock()
        chunk2.content = " World"
        yield {"event": "on_chat_model_stream", "data": {"chunk": chunk2}}

    mock_graph.astream_events.side_effect = event_generator

    # Mock get_chat_history and log_chat in handlers/chat.py
    with patch("bot.handlers.chat.get_chat_history", new_callable=AsyncMock) as mock_history, \
         patch("bot.handlers.chat.log_chat", new_callable=AsyncMock) as mock_log:

        mock_history.return_value = [] # Empty history

        await process_question(mock_graph, mock_message, 123, "Test Question", llm_lock)

        # Verify interactions
        mock_history.assert_called_once_with(123)
        mock_graph.astream_events.assert_called()

        # Verify edit_text called on status message
        # last_msg_mock is the status message returned by message.answer
        assert mock_message.last_msg_mock.edit_text.called
        # Check final text
        # calls:
        # 1. "⏳ ..." (tool usage) - accumulated in display_text
        # 2. "⏳ ...Hello" - accumulated
        # 3. "⏳ ...Hello World" - accumulated
        # But logic only calls edit_text periodically or at end.
        # At end: "⏳ ...Hello World"

        # We can inspect call args
        args = mock_message.last_msg_mock.edit_text.call_args[0]
        assert "Hello World" in args[0]
        assert "⏳" in args[0]

@pytest.mark.asyncio
async def test_process_question_error(mock_message, mock_graph, llm_lock):
    # Mock error during streaming
    mock_graph.astream_events.side_effect = Exception("Graph Error")

    with patch("bot.handlers.chat.get_chat_history", new_callable=AsyncMock), \
         patch("bot.handlers.chat.log_chat", new_callable=AsyncMock):

        # Should not raise exception
        await process_question(mock_graph, mock_message, 123, "Error", llm_lock)

        # Verify error message sent
        assert mock_message.last_msg_mock.edit_text.called
        args = mock_message.last_msg_mock.edit_text.call_args[0]
        assert "ошибка" in args[0].lower()
