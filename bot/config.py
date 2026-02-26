import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_env_variable(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    value = os.getenv(name, default)
    if required and not value:
        logger.error(f"Environment variable {name} is required but not set.")
        raise ValueError(f"Environment variable {name} is required but not set.")
    return value

def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {name}: {value}. Using default: {default}")
        return default

def get_float_env(name: str, default: Optional[float] = None) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}: {value}. Using default: {default}")
        return default

# Bot Configuration
BOT_TOKEN: str = get_env_variable("BOT_TOKEN", required=True)

# RAG Configuration
EMBEDDING_MODEL: str = get_env_variable("EMBEDDING_MODEL", "cointegrated/rubert-tiny2")
OLLAMA_BASE_URL: str = get_env_variable("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = get_env_variable("OLLAMA_MODEL", "qwen2.5:7b")
CHROMA_DIR: str = get_env_variable("CHROMA_DIR", "db")
RETRIEVER_K: int = get_int_env("RETRIEVER_K", 4)
TEMPERATURE: float = get_float_env("TEMPERATURE", 0.3)
MAX_DISTANCE: Optional[float] = get_float_env("MAX_DISTANCE", None) # None implies disabled

# Concurrency
LLM_CONCURRENCY: int = get_int_env("LLM_CONCURRENCY", 1)
