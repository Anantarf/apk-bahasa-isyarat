"""Application logging setup."""
import logging

_LOG_FORMAT = "%(levelname)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a simple default console configuration."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    return logging.getLogger(name)
