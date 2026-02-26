import os
import pytest
from bot import config

def test_config_defaults():
    # Since we mocked env vars in conftest.py, we should check against those mocked values
    assert config.BOT_TOKEN == "123456789:ABCDefGHIJKlmNOPQrstUVwxyz"
    assert config.EMBEDDING_MODEL == "cointegrated/rubert-tiny2"
    assert config.RETRIEVER_K == 4
    assert config.TEMPERATURE == 0.3
    assert config.MAX_DISTANCE == 0.45
    assert config.LLM_CONCURRENCY == 1

def test_config_override(monkeypatch):
    monkeypatch.setenv("RETRIEVER_K", "10")
    monkeypatch.setenv("TEMPERATURE", "0.5")
    monkeypatch.setenv("MAX_DISTANCE", "") # Should be None/disabled

    # Reload config to apply changes
    import importlib
    importlib.reload(config)

    assert config.RETRIEVER_K == 10
    assert config.TEMPERATURE == 0.5
    assert config.MAX_DISTANCE is None
