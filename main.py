import asyncio
import logging
from aiogram import Bot, Dispatcher
from bot import config
from bot.core.resources import initialize_bot_resources
from bot.handlers import base, chat
from bot import database
from bot.core.logger import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

async def main():
    # Initialize Analytics DB
    try:
        await database.init_db()
    except Exception as e:
        logger.error(f"Failed to initialize analytics DB: {e}")

    # Initialize Bot
    try:
        bot = Bot(token=config.BOT_TOKEN)
    except Exception as e:
        logger.error(f"Invalid BOT_TOKEN: {e}")
        exit(1)

    # Initialize Resources (Graph, Retrievers, LLM)
    graph = None
    try:
        graph = initialize_bot_resources()
    except Exception as e:
        logger.error(f"Failed to initialize bot resources: {e}")
        # We continue, but the bot won't be able to answer questions.
        # Handlers check if graph is None.

    # Global lock for LLM
    llm_lock = asyncio.Lock()

    # Initialize Dispatcher with dependencies
    # Inject graph and llm_lock
    dp = Dispatcher(graph=graph, llm_lock=llm_lock)

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
