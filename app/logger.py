"""Structured logging for the CRAG pipeline.

Every agent imports get_logger(name) and logs through it. Output goes to both
the terminal (colored by level) and logs/crag.log.

Format: timestamp | level | agent | message
  2024-01-15 14:23:01 | INFO | RETRIEVER | Searching for: "What is SQL injection?"
"""

import logging
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "crag.log"

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ANSI color codes keyed by level, applied to the terminal handler only.
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that wraps the level name in an ANSI color for the console."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        # Color only the levelname, restoring it afterward so the record stays reusable.
        original = record.levelname
        record.levelname = f"{color}{original}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def get_logger(name: str) -> logging.Logger:
    """Return a logger named for the calling agent (e.g. "RETRIEVER").

    The name is uppercased so it reads as an agent tag in the output, and console
    + file handlers are attached once per name. Propagation is disabled so records
    aren't duplicated by the root logger.
    """
    logger = logging.getLogger(name.upper())
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    console = logging.StreamHandler()
    console.setFormatter(_ColorFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger
