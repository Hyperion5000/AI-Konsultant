import logging
import sys

_configured = False

def configure_logging():
    """
    Configures the root logger with a standardized format.
    Ensures idempotency.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    _configured = True

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance. Ensures logging is configured.
    """
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
