"""Structured job logs. Messages and ``l2kc`` fields are redacted."""

from __future__ import annotations

import logging
import re
from typing import Any

from link2kaggle_colab.core.redact import redact, redact_value

_LOGGER_NAME = "link2kaggle_colab.jobs"
_MAX_FIELD = 300
_PLAIN = re.compile(r"^[A-Za-z0-9_.:/+-]{1,200}$")
# Callers must not pass job inputs. ``message`` is allowed and redacted.
_DROPPED_FIELDS = frozenset(
    {
        "args",
        "env",
        "script",
        "script_path",
        "credential",
        "credentials",
        "token",
        "secret",
        "password",
        "spec",
        "input_files",
    }
)


class RedactingFilter(logging.Filter):
    """Mask secrets on the formatted message and on ``extra={"l2kc": ...}``."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.msg
        if record.args:
            try:
                rendered = message % record.args if isinstance(message, str) else str(message)
            except (TypeError, ValueError):
                rendered = str(message)
            message = rendered
        elif not isinstance(message, str):
            message = str(message)
        record.msg = redact(message)
        record.args = ()
        payload = getattr(record, "l2kc", None)
        if isinstance(payload, dict):
            redacted = redact_value(payload)
            if isinstance(redacted, dict):
                record.l2kc = redacted
        return True


def get_logger() -> logging.Logger:
    """Return the job logger, attaching :class:`RedactingFilter` once."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(item, RedactingFilter) for item in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = ""
    else:
        text = str(value)
    redacted = redact(text)
    if len(redacted) > _MAX_FIELD:
        return redacted[:_MAX_FIELD]
    return redacted


def _quote(text: str) -> str:
    if _PLAIN.fullmatch(text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def log_event(logger: logging.Logger, level: int, event: str, **fields: object) -> None:
    """Emit one line ``event key=value ...`` plus a redacted ``l2kc`` dict.

    Spec args, environment, file contents, and credentials are dropped.
    Pass job id, provider, states, attempt, error code, and a short message.
    """
    safe: dict[str, Any] = {"event": event}
    parts = [event]
    for key, value in fields.items():
        if key in _DROPPED_FIELDS or not key.replace("_", "").isalnum():
            continue
        text = _render_value(value)
        safe[key] = text
        parts.append(f"{key}={_quote(text)}")
    logger.log(level, " ".join(parts), extra={"l2kc": safe})
