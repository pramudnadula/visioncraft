"""Structured logging configuration for FaceVision."""

from __future__ import annotations

import logging
import sys


def configure_logging(debug: bool = False) -> None:
    """Configure consistent structured logging without leaking sensitive paths."""
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("facevision")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under `facevision`."""
    return logging.getLogger(f"facevision.{name}")
