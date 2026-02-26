import asyncio
import logging
import time
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from bot import config
from bot.rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Bot and Dispatcher
# Use defaults or fail if token is invalid (though config.py handles required check)
try:
    bot = Bot(token=config.BOT_TOKEN)
except Exception as e:
    logger.error(f"Invalid BOT_TOKEN: {e}")
    exit(1)

dp = Dispatcher()

# Global RAG Service instance
rag_service: RAGService = None

@dp.message(CommandStart())
async def command_start_handler(message: Message):
    """
    Handler for `/start` command.
    """
    await message.answer(
        f"Привет, {message.from_user.full_name}!\n"
        "Я — твой юридический ИИ-консультант.\n"
        "Задай мне вопрос, и я отвечу на основе законов РФ.\n\n"
        "Команды:\n"
        "/reset - начать новый диалог (очистить контекст)\n"
        "/help - справка"
    )

@dp.message(Command("help"))
async def command_help_handler(message: Message):
    """
    Handler for `/help` command.
    """
    await message.answer(
        "Я использую базу знаний (индексированные документы) для поиска ответов.\n"
        "Также я помню последние несколько сообщений нашего диалога.\n"
        "Если вы хотите сменить тему или чтобы я забыл предыдущий контекст, используйте команду /reset.\n"
        "Ответ может занимать время, так как я работаю на локальном оборудовании."
    )

@dp.message(Command("reset"))
async def command_reset_handler(message: Message):
    """
    Handler for `/reset` command.
    """
    if rag_service:
        rag_service.reset_history(message.from_user.id)
        await message.answer("История диалога очищена. Можем начать сначала!")
    else:
        await message.answer("Сервис еще не готов.")

@dp.message(F.text)
async def handle_message(message: Message):
    """
    Handler for text messages.
    """
    if not rag_service:
        await message.answer("Бот инициализируется, пожалуйста, подождите...")
        return

    user_id = message.from_user.id
    question = message.text

    # We rely on RAGService's semaphore for concurrency, so we don't block here.
    # However, to give immediate feedback, we send a status message first.
    await process_question(rag_service, message, user_id, question)

async def process_question(rag_service: RAGService, message: Message, user_id: int, question: str):
    """
    Process the question using RAG service with streaming response.
    Handles long messages by splitting.
    """
    status_msg = await message.answer("🔍 Анализирую законы...")

    full_response = ""       # Total response (for debugging or history if needed, though history is handled in RAG)
    current_msg_text = ""    # Text for the currently active Telegram message
    current_msg_obj = status_msg

    last_update_time = time.time()
    buffer = ""

    # Telegram message length limit is 4096. We use a safe margin.
    TELEGRAM_LIMIT = 4000

    try:
        # Stream response from RAG service
        async for chunk in rag_service.get_answer(user_id, question):
            full_response += chunk
            current_msg_text += chunk
            buffer += chunk

            # Check if current message is getting too long
            if len(current_msg_text) > TELEGRAM_LIMIT:
                # Finalize the current message (remove cursor if any, though we overwrite it)
                try:
                    await current_msg_obj.edit_text(current_msg_text)
                except Exception as e:
                    logger.warning(f"Failed to finalize message part: {e}")

                # Start a new message for continuation
                current_msg_obj = await message.answer("...")
                current_msg_text = ""
                buffer = ""
                last_update_time = time.time()
                continue

            current_time = time.time()
            # Smart buffering: update only if > 1.5s elapsed AND buffer > 30 chars
            if (current_time - last_update_time > 1.5) and (len(buffer) > 30):
                try:
                    # Add cursor effect
                    await current_msg_obj.edit_text(current_msg_text + " ▌")
                    last_update_time = current_time
                    buffer = ""
                except Exception as e:
                    logger.warning(f"Failed to edit message: {e}")

        # Final update for the last active message
        if current_msg_text:
            try:
                await current_msg_obj.edit_text(current_msg_text)
            except Exception as e:
                logger.warning(f"Failed to final edit message: {e}")
        elif not full_response:
             # If completely empty (e.g. refusal yielded empty string? No, refusal yields text)
             # But if error or empty generator
             await current_msg_obj.edit_text("Не удалось сформировать ответ.")

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        try:
            await current_msg_obj.edit_text("Произошла ошибка при генерации ответа. Попробуйте позже.")
        except:
            await message.answer("Произошла ошибка при генерации ответа.")

async def main():
    global rag_service
    # Initialize RAG Service
    try:
        rag_service = RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        # We don't exit, maybe it will work later? Or we should exit.
        # But for now let's keep running so bot can at least say "error".
        # But handler checks for rag_service.
        pass

    # Start polling
    logger.info("Starting bot polling...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped!")
