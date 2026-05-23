"""
Centralized logging setup with Rich console output and rotating file handler.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

try:
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def get_logger(name: str = "xai_ids", log_file: str = "logs/xai_ids.log",
               level: str = "INFO") -> logging.Logger:
    """
    Returns a configured logger.
    - Console: colored output via Rich (if installed), else plain
    - File: rotating handler (10 MB per file, keep 5 backups)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    # Console handler
    if RICH_AVAILABLE:
        console_handler = RichHandler(rich_tracebacks=True, markup=True,
                                      show_time=True, show_path=False)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
