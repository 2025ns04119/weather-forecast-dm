"""
Shared logging: every pipeline step writes to the console AND to a log file.

    logs/ingestion.log     ingestion (API + CSV)
    logs/validation.log    validation / cleaning
    ...

Format:  2026-11-16 13:17:05 UTC | INFO    | ingestion | message
The log folder can be changed with the LOG_DIR environment variable.
"""
import logging
import os
import time
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_FORMAT = "%(asctime)s UTC | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to logs/<name>.log and to the console."""
    logger = logging.getLogger(name)
    if logger.handlers:                       # already configured in this process
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime         # timestamps in UTC, same as the database

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
