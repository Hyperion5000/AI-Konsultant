import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import main
from main import process_question, handle_message, llm_lock, command_start_handler

# --- Fixtures ---

@pytest.fixture
def mock_message():
    # Use MagicMock for the message object itself, but ensure methods are AsyncMock
    message = MagicMock()
    message.from_user.id = 12345
    message.from_user.full_name = "Test User"
    message.text = "Test question"

    # message.answer is an async method
    message.answer = AsyncMock()

    # The result of awaiting message.answer(...) is status_msg
    status_msg = MagicMock()
    status_msg.edit_text = AsyncMock()

    message.answer.return_value = status_msg

    return message

@pytest.fixture
def mock_rag_service():
    service = MagicMock()
    return service

# --- Tests ---

@pytest.mark.asyncio
async def test_command_start(mock_message):
    """
    Test /start command.
    """
    await command_start_handler(mock_message)
    mock_message.answer.assert_called_once()
    assert "Привет" in mock_message.answer.call_args[0][0]

@pytest.mark.asyncio
async def test_handle_message_unlocked(mock_message):
    """
    Test handle_message when lock is free.
    """
    # Ensure lock is free
    if llm_lock.locked():
        llm_lock.release()

    with patch("main.process_question", new_callable=AsyncMock) as mock_process:
        await handle_message(mock_message)

        # Check that process_question was called directly
        mock_process.assert_called_once()
        # Check that "queue" message was NOT sent
        mock_message.answer.assert_not_called()

@pytest.mark.asyncio
async def test_concurrency_lock_message(mock_message):
    """
    Test that if llm_lock is locked, the user receives a queue message.
    """
    await llm_lock.acquire()

    try:
        with patch("main.process_question", new_callable=AsyncMock) as mock_process:
            task = asyncio.create_task(handle_message(mock_message))
            await asyncio.sleep(0.01)

            mock_message.answer.assert_any_call("⏳ Система сейчас обрабатывает другой запрос, ваш вопрос в очереди...")
            mock_process.assert_not_called()

            llm_lock.release()
            await task
            mock_process.assert_called_once()

    finally:
        if llm_lock.locked():
            llm_lock.release()

@pytest.mark.asyncio
async def test_debounce_buffer_logic(mock_rag_service, mock_message):
    """
    Critical Test: Verify edit_text is called only when buffer conditions are met.
    """
    async def fast_chunk_generator(user_id, question):
        for i in range(100):
            yield "a"

    mock_rag_service.get_answer.side_effect = fast_chunk_generator

    time_values = [1000.0] # init
    time_values.extend([1000.0] * 50) # loop 0-49 (chunks 1-50)
    time_values.append(1002.0) # loop 50 (chunk 51) -> Update

    remaining_calls = 100 - 50 - 1
    time_values.extend([1002.0] * remaining_calls)
    time_values.extend([1005.0] * 10)

    with patch("time.time", side_effect=time_values):
        await process_question(mock_rag_service, mock_message, 123, "Test")

    status_msg = mock_message.answer.return_value

    # Verify debounce
    # We expect at least one update (chunk 51) and one final update (chunk 100)
    assert status_msg.edit_text.call_count >= 2
    assert status_msg.edit_text.call_count < 10

    # Check intermediate update has cursor
    calls = status_msg.edit_text.call_args_list
    assert "▌" in calls[0][0][0]

    # Check final update has no cursor and full text
    assert calls[-1][0][0] == "a" * 100

@pytest.mark.asyncio
async def test_graceful_degradation_error(mock_rag_service, mock_message):
    """
    Test that exceptions in RAG service are handled gracefully.
    """
    async def error_generator(user_id, question):
        yield "Start"
        raise Exception("Database Connection Failed")

    mock_rag_service.get_answer.side_effect = error_generator

    await process_question(mock_rag_service, mock_message, 123, "Error test")

    status_msg = mock_message.answer.return_value
    assert status_msg.edit_text.called
    assert "Произошла ошибка при генерации ответа" in status_msg.edit_text.call_args_list[-1][0][0]

@pytest.mark.asyncio
async def test_empty_response(mock_rag_service, mock_message):
    """
    Test handling of empty response from RAG service.
    """
    async def empty_generator(user_id, question):
        if False: yield "nothing" # Empty generator

    mock_rag_service.get_answer.side_effect = empty_generator

    await process_question(mock_rag_service, mock_message, 123, "Test")

    status_msg = mock_message.answer.return_value
    status_msg.edit_text.assert_called_with("К сожалению, я не смог сформировать ответ.")

