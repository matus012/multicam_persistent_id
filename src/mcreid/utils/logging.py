"""Logging setup. Never use `print` in library code — use `get_logger(__name__)`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    force: bool = False,
) -> None:
    """Configure root logging once per process.

    Args:
        level: Logging level (int or name such as ``"DEBUG"``).
        log_file: Optional file to mirror logs into. Parent dirs are created.
        force: Re-configure even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    if isinstance(level, str):
        level = logging.getLevelNamesMapping()[level.upper()]

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    if force:
        for existing in list(root.handlers):
            root.removeHandler(existing)
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    # Third-party noise.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
