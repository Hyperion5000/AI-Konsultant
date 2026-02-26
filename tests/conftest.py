import os
import pytest
import sys
from unittest.mock import MagicMock

# Set environment variables before any other imports
# Use a valid-looking token to pass aiogram validation
os.environ["BOT_TOKEN"] = "123456789:ABCDefGHIJKlmNOPQrstUVwxyz"
os.environ["EMBEDDING_MODEL"] = "cointegrated/rubert-tiny2"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "qwen2.5:7b"
os.environ["CHROMA_DIR"] = "test_db"
os.environ["RETRIEVER_K"] = "4"
os.environ["TEMPERATURE"] = "0.3"
os.environ["MAX_DISTANCE"] = "0.45"
os.environ["LLM_CONCURRENCY"] = "1"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

@pytest.fixture(autouse=True)
def reload_config():
    """
    Ensure config is reloaded for each test to pick up any environment changes.
    """
    if 'bot.config' in sys.modules:
        import importlib
        from bot import config
        importlib.reload(config)
