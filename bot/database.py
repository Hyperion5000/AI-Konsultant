import aiosqlite
import logging
import os
from datetime import datetime
from bot import config

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(config.CHROMA_DIR, "analytics.db")

async def init_db():
    """Initializes the analytics database."""
    if not os.path.exists(config.CHROMA_DIR):
        os.makedirs(config.CHROMA_DIR, exist_ok=True)

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
