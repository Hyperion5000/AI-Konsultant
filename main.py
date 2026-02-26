import asyncio
import logging
from aiogram import Bot, Dispatcher
from bot import config
from bot.rag_service import RAGService
from bot.handlers import base, chat
from bot import database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Initialize Analytics DB
    try:
        await database.init_db()
    except Exception as e:
        logger.error(f"Failed to initialize analytics DB: {e}")
        # Continue anyway, bot can work without logs if needed, or exit.
        # But MVP might prefer working bot.

    # Initialize Bot
    try:
        bot = Bot(token=config.BOT_TOKEN)
    except Exception as e:
        logger.error(f"Invalid BOT_TOKEN: {e}")
        exit(1)

    # Initialize RAG Service
    try:
        rag_service = RAGService()
    except Exception as e:
        logger.error(f"Failed to initialize RAG Service: {e}")
        rag_service = None

    # Global lock for LLM
    llm_lock = asyncio.Lock()

    # Initialize Dispatcher with dependencies
    dp = Dispatcher(rag_service=rag_service, llm_lock=llm_lock)

    # Include routers
    dp.include_router(base.router)
    dp.include_router(chat.router)

    # Start polling
    logger.info("Starting bot polling...")
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Polling error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped!")
