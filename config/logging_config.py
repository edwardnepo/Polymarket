"""Standardised, structured logging for the whole project.

Call :func:`configure_logging` once at every entry point (pipeline, Streamlit
app, ad-hoc scripts) and obtain module loggers via :func:`get_logger`.

When ``LOG_JSON`` is true, logs are emitted as one JSON object per line using
``python-json-logger`` — ideal for ingestion by Cloud Logging / ELK.  When
false, a compact human-readable format is used for local development.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from pythonjsonlogger import jsonlogger

from config.settings import get_settings

_CONFIGURED = False

# Fields surfaced in every JSON log record.
_JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    level: Optional[str] = None,
    *,
    json_logs: Optional[bool] = None,
    force: bool = False,
) -> None:
    """Configure the root logger exactly once (idempotent).

    Args:
        level: Override log level (defaults to ``settings.log_level``).
        json_logs: Override JSON vs. text formatting (defaults to
            ``settings.log_json``).
        force: Re-apply configuration even if already configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = get_settings()
    level = (level or settings.log_level).upper()
    json_logs = settings.log_json if json_logs is None else json_logs

    handler = logging.StreamHandler(stream=sys.stdout)
    if json_logs:
        handler.setFormatter(
            jsonlogger.JsonFormatter(
                _JSON_FORMAT,
                rename_fields={"asctime": "timestamp", "levelname": "level"},
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quieten noisy third-party libraries.
    for noisy in ("urllib3", "requests", "transformers", "torch", "filelock", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; auto-configures on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
