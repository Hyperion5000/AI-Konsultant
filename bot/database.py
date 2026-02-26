import aiosqlite
import logging
import os
from datetime import datetime
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from bot import config
from bot.core.logger import get_logger

logger = get_logger(__name__)

# Use new config path
DB_PATH = config.SQLITE_DB_PATH

async def init_db():
    """Initializes the analytics database."""
    # Ensure directory exists
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp DATETIME,
                question TEXT,
                answer TEXT,
                retrieved_context_sources TEXT
            )
        """)
        # Add index for faster retrieval by user_id
        await db.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON chat_logs(user_id)")
        await db.commit()
        logger.info(f"Analytics DB initialized at {DB_PATH}")

async def log_chat(user_id: int, question: str, answer: str, sources: str):
    """Logs a chat interaction to the database."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO chat_logs (user_id, timestamp, question, answer, retrieved_context_sources) VALUES (?, ?, ?, ?, ?)",
                (user_id, datetime.now(), question, answer, sources)
            )
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to log chat: {e}")

async def get_chat_history(user_id: int, limit: int = 6) -> List[BaseMessage]:
    """
    Retrieves the chat history for a user from the database.
    Returns a list of LangChain BaseMessage objects (HumanMessage, AIMessage).
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            # We need to order by id DESC to get latest, then reverse.
            rows_limit = limit // 2 if limit % 2 == 0 else (limit + 1) // 2

            cursor = await db.execute(
                "SELECT question, answer FROM chat_logs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, rows_limit)
            )
            rows = await cursor.fetchall()

            messages = []
            for q, a in reversed(rows):
                if q:
                    messages.append(HumanMessage(content=q))
                if a:
                    messages.append(AIMessage(content=a))

            return messages
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {e}")
        return []

async def clear_chat_history(user_id: int):
    """
    Clears the chat history for a user (deletes logs).
    In a real system, you might want to soft-delete or archive.
    """
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM chat_logs WHERE user_id = ?", (user_id,))
            await db.commit()
            logger.info(f"Chat history cleared for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}")
