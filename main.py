import asyncio
import logging
import os
import time
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.utils.markdown import hbold
from dotenv import load_dotenv

from bot.rag_service import RAGService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    logger.error("BOT_TOKEN not found in .env file")
    exit(1)

# Initialize Bot and Dispatcher
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Global Lock for CPU-bound LLM generation
llm_lock = asyncio.Lock()

# Initialize RAG Service (will be done in main async function)
rag_service: RAGService = None

@dp.message(CommandStart())
async def command_start_handler(message: types.Message):
    """
    This handler receives messages with `/start` command
    """
    await message.answer(
        f"Привет, {hbold(message.from_user.full_name)}!\n"
        "Я — твой юридический ИИ-консультант.\n"
        "Задай мне вопрос, и я отвечу на основе законов РФ (ФЗ-214, Закон о защите прав потребителей и др.).\n"
        "Пожалуйста, учти, что я работаю локально, поэтому ответ может занять некоторое время."
    )

@dp.message(F.text)
async def handle_message(message: types.Message):
    """
    Handler for text messages
    """
    user_id = message.from_user.id
    question = message.text

    # Check lock availability without waiting
    if llm_lock.locked():
        await message.answer("⏳ Система сейчас обрабатывает другой запрос, ваш вопрос в очереди...")
        # Wait for lock
        async with llm_lock:
            await process_question(rag_service, message, user_id, question)
    else:
        async with llm_lock:
            await process_question(rag_service, message, user_id, question)

async def process_question(rag_service: RAGService, message: types.Message, user_id: int, question: str):
    """
    Process the question using RAG service with streaming response.
    """
    status_msg = await message.answer("🔍 Анализирую законы...")

    full_response = ""
    buffer = ""
    last_update_time = time.time()

    try:
        # Stream response from RAG service
        async for chunk in rag_service.get_answer(user_id, question):
            full_response += chunk
            buffer += chunk

            current_time = time.time()
            # Smart buffering: update only if > 1.5s elapsed AND buffer > 30 chars
            if (current_time - last_update_time > 1.5) and (len(buffer) > 30):
                try:
                    await status_msg.edit_text(full_response + " ▌") # Add cursor effect
                    last_update_time = current_time
                    buffer = ""
                except Exception as e:
                    logger.warning(f"Failed to edit message: {e}")

        # Final update
        if full_response:
            await status_msg.edit_text(full_response)
        else:
            await status_msg.edit_text("К сожалению, я не смог сформировать ответ.")

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        await status_msg.edit_text("Произошла ошибка при генерации ответа. Попробуйте позже.")

async def main():
    global rag_service
    # Initialize RAG Service
    try:
        rag_service = RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        return

    # Start polling
    logger.info("Starting bot polling...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped!")
