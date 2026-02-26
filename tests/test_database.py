import pytest
import aiosqlite
import os
from unittest.mock import patch
from bot.database import init_db, log_chat
from bot import config

@pytest.mark.asyncio
async def test_database_operations(tmp_path):
    # Use a temporary file for DB
    db_file = tmp_path / "test_analytics.db"
    db_path = str(db_file)

    # Patch config.CHROMA_DIR to tmp_path so it creates dir there if needed (tmp_path exists)
    with patch("bot.database.config.CHROMA_DIR", str(tmp_path)):
        with patch("bot.database.DB_PATH", db_path):
            # 1. Init DB
            await init_db()
            assert os.path.exists(db_path)

            # 2. Log chat
            await log_chat(123, "Question", "Answer", "Source1")

            # 3. Verify data
            async with aiosqlite.connect(db_path) as db:
                async with db.execute("SELECT * FROM chat_logs") as cursor:
                    rows = await cursor.fetchall()
                    assert len(rows) == 1
                    row = rows[0]
                    # id, user_id, timestamp, question, answer, sources
                    assert row[1] == 123
                    assert row[3] == "Question"
                    assert row[4] == "Answer"
                    assert row[5] == "Source1"

@pytest.mark.asyncio
async def test_init_db_creates_directory(tmp_path):
    # Test directory creation
    new_dir = tmp_path / "subdir"
    db_file = new_dir / "db.sqlite"

    with patch("bot.database.config.CHROMA_DIR", str(new_dir)):
        with patch("bot.database.DB_PATH", str(db_file)):
            await init_db()
            assert new_dir.exists()
            assert db_file.exists()
