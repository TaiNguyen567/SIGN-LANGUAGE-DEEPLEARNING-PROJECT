"""Quiet console logging with persistent project log files."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logger(name: str, log_path: str | Path, *, console: bool = False) -> logging.Logger:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path.resolve()
               for handler in logger.handlers):
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)
    if console and not any(getattr(handler, "_sign_language_console", False) for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler._sign_language_console = True
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(console_handler)
    return logger
