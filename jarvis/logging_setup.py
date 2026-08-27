"""Logging bootstrap: readable console output + rotating file log."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .config import LoggingConfig

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def setup_logging(config: LoggingConfig, *, force: bool = False) -> None:
    """Configure the root logger once per process."""
    global _configured
    if _configured and not force:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(getattr(logging, config.level))
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    if config.file is not None:
        log_path = Path(config.file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - depends on filesystem
            root.warning("Không ghi được log ra file %s: %s", log_path, exc)

    # Third-party chatter that is not useful at INFO level.
    for noisy in ("urllib3", "httpx", "httpcore", "websockets", "PIL", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
