import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiogram import types
import asyncio

# Mock main.py dependencies before importing
with patch("main.RAGService") as MockRAG:
    from main import handle_message, process_question, llm_lock

@pytest.fixture
def mock_message():
    message = AsyncMock(spec=types.Message)
    message.text = "Hello Law"
    message.from_user = MagicMock()
    message.from_user.id = 123
    message.from_user.full_name = "User"
    message.answer = AsyncMock()
    return message

@pytest.fixture
def mock_rag_service():
    rag = MagicMock()
    # Mock get_answer to return async generator
    async def mock_get_answer(user_id, question):
        yield "Part 1"
        yield "Part 2"
    rag.get_answer = mock_get_answer
    return rag

@pytest.mark.asyncio
async def test_handle_message_lock(mock_message, mock_rag_service):
    # Patch main.rag_service
    with patch("main.rag_service", mock_rag_service):
        with patch("main.process_question", new_callable=AsyncMock) as mock_process:
            # Case 1: Lock is free
            assert not llm_lock.locked()
            await handle_message(mock_message)
            mock_process.assert_awaited_once()

            # Case 2: Lock is taken
            with patch("main.llm_lock") as mock_lock:
                mock_lock.locked.return_value = True
                mock_lock.__aenter__ = AsyncMock()
                mock_lock.__aexit__ = AsyncMock()

                await handle_message(mock_message)

                # Verify user was notified
                mock_message.answer.assert_called_with("⏳ Система сейчас обрабатывает другой запрос, ваш вопрос в очереди...")

@pytest.mark.asyncio
async def test_process_question_streaming(mock_message, mock_rag_service):
    with patch("main.rag_service", mock_rag_service):
        with patch("main.time.time") as mock_time:
            # Create a generator for time
            time_values = [0, 0.1, 2.0, 4.0, 6.0, 8.0, 10.0]
            mock_time.side_effect = time_values

            # Mock status message returned by answer
            status_msg = AsyncMock()
            mock_message.answer.return_value = status_msg

            # Use large chunks to trigger update
            async def large_chunks(user_id, question):
                yield "A" * 20 # Buffer 20
                yield "B" * 20 # Buffer 40. Total 40.

            mock_rag_service.get_answer = large_chunks

            await process_question(mock_message, 123, "Question")

            # Verify answer called
            mock_message.answer.assert_called_with("🔍 Анализирую законы...")

            # Verify edits
            # 1. Start: time=0
            # 2. Chunk A: time=0.1. Diff=0.1. No edit.
            # 3. Chunk B: time=2.0. Diff=1.9. Buffer=40 > 30. Edit!
            #    Edit with "A"*20 + "B"*20 + " ▌"
            #    Reset buffer to ""? No, process_question resets buffer.
            # 4. Final update with "A"*20 + "B"*20

            # Check calls
            # We expect at least one intermediate edit and one final edit.
            assert status_msg.edit_text.call_count >= 2

            # Check arguments of intermediate edit
            # The exact content might vary depending on loop, but we expect cursor in intermediate
            intermediate_call = status_msg.edit_text.call_args_list[-2]
            assert "▌" in intermediate_call[0][0]

            final_call = status_msg.edit_text.call_args_list[-1]
            assert "▌" not in final_call[0][0]
            assert final_call[0][0] == "A"*20 + "B"*20
