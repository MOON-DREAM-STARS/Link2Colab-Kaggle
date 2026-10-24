"""Retry status, logs, and download. Submit and cancel are never retried."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from link2kaggle_colab.core.errors import (
    InvalidInputError,
    ProviderCLIError,
    RuntimeDisconnectedError,
    TransientProviderError,
)
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.jobs.logging import log_event

SAFE_RETRY_OPERATIONS = frozenset({"status", "logs", "download"})

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Full-jitter backoff. ``attempt`` is 1-based and counts the try that failed."""

    max_attempts: int
    base_delay: float
    max_delay: float

    def delay(self, attempt: int, rng: random.Random) -> float:
        """Return ``uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1)))``."""
        if attempt < 1:
            raise InvalidInputError(
                "retry attempt must be >= 1",
                details={"attempt": attempt},
            )
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        # Jitter only. Not used for secrets or identifiers.
        return rng.uniform(0, ceiling)  # noqa: S311


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is a transport blip that status/logs/download may retry.

    :class:`RuntimeDisconnectedError` is a job outcome and is not transient.
    :class:`ProviderCLIError` is transient only when ``details["timed_out"]``
    or ``details["transient"]`` is exactly ``True``.
    """
    if isinstance(exc, RuntimeDisconnectedError):
        return False
    if isinstance(exc, TransientProviderError):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ConnectionError):
        return True
    if isinstance(exc, ProviderCLIError):
        return exc.details.get("timed_out") is True or exc.details.get("transient") is True
    return False


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__


def _error_message(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return redact(str(exc))


async def retry_async(
    operation: str,
    func: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random,
    logger: logging.Logger,
) -> T:
    """Call ``func`` up to ``policy.max_attempts`` while :func:`is_transient`.

    ``operation`` must be in :data:`SAFE_RETRY_OPERATIONS`. Submit and cancel
    are rejected before ``func`` runs. The last exception is re-raised.
    """
    if operation not in SAFE_RETRY_OPERATIONS:
        raise InvalidInputError(
            f"operation {operation} is not retryable",
            details={"operation": operation},
        )
    if policy.max_attempts < 1:
        raise InvalidInputError(
            "retry max_attempts must be >= 1",
            details={"max_attempts": policy.max_attempts},
        )
    last: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await func()
        except Exception as exc:
            last = exc
            if not is_transient(exc) or attempt >= policy.max_attempts:
                raise
            wait = policy.delay(attempt, rng)
            log_event(
                logger,
                logging.WARNING,
                f"{operation}_retry",
                operation=operation,
                attempt=attempt,
                delay_seconds=wait,
                error_code=_error_code(exc),
                message=_error_message(exc),
            )
            await sleep(wait)
    if last is not None:
        raise last
    raise InvalidInputError(
        "retry finished without a result",
        details={"operation": operation},
    )
