"""Console logging. One configuration, applied once, shared by CLI and library code."""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

console = Console()
_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("SD_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=lvl,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # These libraries log a request line per call; at scale that buries our own output.
    for noisy in ("httpx", "httpcore", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
