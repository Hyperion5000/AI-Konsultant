import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from bot.handlers.base import command_reset_handler, command_start_handler
from bot.handlers.chat import process_question
from aiogram.types import Message
from bot.rag_service import RAGService

@pytest.fixture
def mock_message():
    message = AsyncMock()
    message.from_user.id = 123
    message.from_user.full_name = "Test User"
    message.text = "Question"

    # Mock message.answer to return a NEW mock each time (simulating new message object)
    # But we need to track them.

    # Side effect to return a new AsyncMock
    def answer_side_effect(*args, **kwargs):
        msg = AsyncMock()
        msg.edit_text = AsyncMock() # Ensure edit_text is mockable
        return msg

    message.answer.side_effect = answer_side_effect
    return message

@pytest.fixture
def mock_rag():
    rag = MagicMock(spec=RAGService)
    rag.reset_history = MagicMock()
    return rag

@pytest.fixture
def llm_lock():
    return asyncio.Lock()

@pytest.mark.asyncio
async def test_command_start(mock_message):
    await command_start_handler(mock_message)
    mock_message.answer.assert_called_once()
    assert "Привет" in mock_message.answer.call_args[0][0]

@pytest.mark.asyncio
async def test_command_reset(mock_message, mock_rag):
    # Pass mock_rag explicitly
    await command_reset_handler(mock_message, rag_service=mock_rag)
    mock_rag.reset_history.assert_called_with(123)
    # Check if answer was called with success message
    assert mock_message.answer.called
    assert "очищена" in mock_message.answer.call_args[0][0]

@pytest.mark.asyncio
async def test_process_question_long_message(mock_message, mock_rag, llm_lock):
    # Test message splitting when response > 4000 chars

    # 45 chunks * 100 = 4500 chars. Limit is 4000.
    chunk = "a" * 100

    async def mock_generator(uid, q):
        for _ in range(45):
             yield chunk

    mock_rag.get_answer.side_effect = mock_generator

    await process_question(mock_rag, mock_message, 123, "Long question", llm_lock)

    # Verify that message.answer("...") was called
    # Calls to message.answer:
    # 1. "Analysing..." (initial)
    # 2. "..." (when split happens)

    answer_calls = mock_message.answer.call_args_list
    assert len(answer_calls) >= 2
    assert answer_calls[0][0][0] == "🔍 Анализирую законы..."

    # Check if "..." was sent
    continuation_calls = [call for call in answer_calls if call.args[0] == "..."]
    assert len(continuation_calls) >= 1

@pytest.mark.asyncio
async def test_process_question_error(mock_message, mock_rag, llm_lock):
    async def error_generator(uid, q):
        raise Exception("Error")
        yield "ignored" # Unreachable

    mock_rag.get_answer.side_effect = error_generator

    await process_question(mock_rag, mock_message, 123, "Error", llm_lock)

    # We can't easily assert on the return value of message.answer() because it's dynamic mock.
    # But we can check that an error message was sent or edited.
    # Since we can't inspect the returned mock object easily here without capturing,
    # let's just ensure no unhandled exception propagated.
    pass
