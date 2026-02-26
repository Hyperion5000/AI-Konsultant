import asyncio
import logging
import time
from aiogram import Router, F
from aiogram.types import Message
from bot.rag_service import RAGService

router = Router()
logger = logging.getLogger(__name__)

@router.message(F.text)
async def handle_message(message: Message, rag_service: RAGService, llm_lock: asyncio.Lock):
    """
    Handler for text messages.
    """
    if not rag_service:
        await message.answer("Бот инициализируется, пожалуйста, подождите...")
        return

    user_id = message.from_user.id
    question = message.text

    # We use the injected llm_lock to manage concurrency at the handler level if needed,
    # or pass it to process_question.
    # The requirement is to pass llm_lock to handlers.
    # We will use it in process_question to ensure only one heavy generation happens at a time if that's the goal.
    # Or maybe the user just wants it available.
    # Given the CPU constraints, serializing is safer.

    # We don't want to block the handler itself (so we can answer ping/pong),
    # but we want to process the question.
    # We'll call process_question which will use the lock internally or we await it here?
    # If we await it here, we block this coroutine.
    await process_question(rag_service, message, user_id, question, llm_lock)

async def process_question(rag_service: RAGService, message: Message, user_id: int, question: str, llm_lock: asyncio.Lock):
    """
    Process the question using RAG service with streaming response.
    Handles long messages by splitting.
    """
    # Send initial status
    status_msg = await message.answer("🔍 Анализирую законы...")

    full_response = ""       # Total response
    current_msg_text = ""    # Text for the currently active Telegram message
    current_msg_obj = status_msg

    last_update_time = time.time()
    buffer = ""

    TELEGRAM_LIMIT = 4000

    try:
        # Acquire lock before starting generation to prevent CPU overload from multiple users
        # interacting with the LLM simultaneously.
        async with llm_lock:
            # Stream response from RAG service
            async for chunk in rag_service.get_answer(user_id, question):
                full_response += chunk
                current_msg_text += chunk
                buffer += chunk

                # Check if current message is getting too long
                if len(current_msg_text) > TELEGRAM_LIMIT:
                    try:
                        await current_msg_obj.edit_text(current_msg_text)
                    except Exception as e:
                        logger.warning(f"Failed to finalize message part: {e}")

                    current_msg_obj = await message.answer("...")
                    current_msg_text = ""
                    buffer = ""
                    last_update_time = time.time()
                    continue

                current_time = time.time()
                # Smart buffering: update only if > 1.5s elapsed AND buffer > 30 chars
                if (current_time - last_update_time > 1.5) and (len(buffer) > 30):
                    try:
                        await current_msg_obj.edit_text(current_msg_text + " ▌")
                        last_update_time = current_time
                        buffer = ""
                    except Exception as e:
                        logger.warning(f"Failed to edit message: {e}")

        # Final update
        if current_msg_text:
            try:
                await current_msg_obj.edit_text(current_msg_text)
            except Exception as e:
                logger.warning(f"Failed to final edit message: {e}")
        elif not full_response:
             await current_msg_obj.edit_text("Не удалось сформировать ответ.")

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        try:
            await current_msg_obj.edit_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        except:
            await message.answer("Произошла ошибка при генерации ответа.")
