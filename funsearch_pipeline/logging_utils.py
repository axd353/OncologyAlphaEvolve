from __future__ import annotations

import logging
from pathlib import Path


def configure_main_logger(log_path: Path, level_name: str) -> logging.Logger:
    return configure_file_logger(
        log_path,
        level_name,
        logger_name="funsearch_pipeline",
        include_stream=True,
    )


def configure_file_logger(
    log_path: Path,
    level_name: str,
    *,
    logger_name: str,
    include_stream: bool = False,
) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if include_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
