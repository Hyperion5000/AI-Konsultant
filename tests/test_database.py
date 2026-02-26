import pytest
import aiosqlite
import os
from unittest.mock import patch
from bot.database import init_db, log_chat, get_chat_history, clear_chat_history
from bot import config
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
async def temp_db(tmp_path):
    # Use a temporary file for DB
    db_file = tmp_path / "test_analytics.db"
    db_path = str(db_file)

    # We patch DB_PATH inside bot.database module
    with patch("bot.database.DB_PATH", db_path):
        await init_db()
        yield db_path
        # Cleanup if needed (tmp_path is auto cleaned)

@pytest.mark.asyncio
async def test_database_operations(temp_db):
    db_path = temp_db

    # 1. Log chat
    await log_chat(123, "Question 1", "Answer 1", "Source1")
    await log_chat(123, "Question 2", "Answer 2", "Source2")

    # 2. Verify data
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT * FROM chat_logs WHERE user_id = 123") as cursor:
            rows = await cursor.fetchall()
            assert len(rows) == 2
            assert rows[0][3] == "Question 1" # question
            assert rows[1][3] == "Question 2"

@pytest.mark.asyncio
async def test_chat_history(temp_db):
    # Add some history
    await log_chat(123, "Q1", "A1", "")
    await log_chat(123, "Q2", "A2", "")
    await log_chat(123, "Q3", "A3", "")

    # Retrieve history limit 4 (2 pairs)
    history = await get_chat_history(123, limit=4)

    # Expect Q2, A2, Q3, A3
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "Q2"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "A2"
    assert history[3].content == "A3"

@pytest.mark.asyncio
async def test_clear_history(temp_db):
    await log_chat(123, "Q1", "A1", "")
    await log_chat(456, "Q_User2", "A_User2", "")

    # Clear user 123
    await clear_chat_history(123)

    # Verify user 123 is empty
    history = await get_chat_history(123, limit=10)
    assert len(history) == 0

    # Verify user 456 still exists
    history_456 = await get_chat_history(456, limit=10)
    assert len(history_456) == 2

@pytest.mark.asyncio
async def test_init_db_creates_directory(tmp_path):
    # Test directory creation
    new_dir = tmp_path / "subdir"
    db_file = new_dir / "db.sqlite"

    with patch("bot.database.DB_PATH", str(db_file)):
        await init_db()
        assert new_dir.exists()
        assert db_file.exists()
