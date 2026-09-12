"""Same minimal logging setup as youtube-intelligence-engine."""
from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("video_factory")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] video_factory: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
