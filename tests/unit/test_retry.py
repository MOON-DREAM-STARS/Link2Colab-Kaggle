"""Retry policy: safe operations, jitter, and transient classification."""

from __future__ import annotations

import logging
import random

import pytest

from link2kaggle_colab.core.errors import (
    AccountDisabledError,
    AccountNotFoundError,
    AuthMissingError,
    CredentialRefError,
    InvalidInputError,
    L2KCError,
    NotSupportedError,
    PathSecurityError,
    PremiumNotAllowed,
    ProviderCLIError,
    QuotaExceededError,
    RealModeDisabledError,
    RuntimeDisconnectedError,
    TransientProviderError,
)
from link2kaggle_colab.jobs.logging import get_logger
from link2kaggle_colab.jobs.retry import (
    SAFE_RETRY_OPERATIONS,
    RetryPolicy,
    is_transient,
    retry_async,
)


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311


class _Sleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_safe_operations_are_status_logs_and_download() -> None:
    assert SAFE_RETRY_OPERATIONS == frozenset({"status", "logs", "download"})
    assert "submit" not in SAFE_RETRY_OPERATIONS
    assert "cancel" not in SAFE_RETRY_OPERATIONS


def test_delay_is_full_jitter_inside_the_exponential_cap() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0)
    rng = _rng(7)
    first = policy.delay(1, rng)
    second = policy.delay(2, rng)
    third = policy.delay(3, rng)
    assert 0 <= first <= 0.5
    assert 0 <= second <= 1.0
    assert 0 <= third <= 2.0
    again = _rng(7)
    assert first == RetryPolicy(4, 0.5, 8.0).delay(1, again)
    assert second == RetryPolicy(4, 0.5, 8.0).delay(2, again)
    capped = RetryPolicy(max_attempts=4, base_delay=5.0, max_delay=8.0)
    assert 0 <= capped.delay(4, _rng(3)) <= 8.0


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TransientProviderError("blip"), True),
        # asyncio.TimeoutError is TimeoutError on Python 3.11+.
        (TimeoutError("slow"), True),
        (ConnectionError("reset"), True),
        (ConnectionResetError("reset"), True),
        (RuntimeDisconnectedError("gone"), False),
        (ProviderCLIError("timed out", details={"timed_out": True}), True),
        (ProviderCLIError("again", details={"transient": True}), True),
        (ProviderCLIError("nope", details={"timed_out": "true"}), False),
        (ProviderCLIError("cli failed"), False),
        (AuthMissingError("missing"), False),
        (InvalidInputError("bad"), False),
        (PathSecurityError("escape"), False),
        (QuotaExceededError("quota"), False),
        (PremiumNotAllowed("premium"), False),
        (NotSupportedError("no"), False),
        (RealModeDisabledError("off"), False),
        (AccountNotFoundError("missing"), False),
        (AccountDisabledError("disabled"), False),
        (CredentialRefError("ref"), False),
        (L2KCError("other"), False),
        (OSError("disk"), False),
        (RuntimeError("boom"), False),
    ],
)
def test_is_transient_matches_the_transport_blip_rules(exc: BaseException, expected: bool) -> None:
    assert is_transient(exc) is expected


@pytest.mark.parametrize("operation", ["submit", "cancel"])
async def test_retry_async_refuses_submit_and_cancel(operation: str) -> None:
    calls = 0

    async def _func() -> str:
        nonlocal calls
        calls += 1
        return "nope"

    with pytest.raises(InvalidInputError):
        await retry_async(
            operation,
            _func,
            policy=RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=8.0),
            sleep=_Sleep(),
            rng=_rng(1),
            logger=get_logger(),
        )
    assert calls == 0


async def test_retry_async_retries_only_transient_errors() -> None:
    sleep = _Sleep()
    attempts = {"count": 0}

    async def _flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("dropped")
        return "ok"

    policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0)
    result = await retry_async(
        "status",
        _flaky,
        policy=policy,
        sleep=sleep,
        rng=_rng(11),
        logger=get_logger(),
    )
    assert result == "ok"
    assert attempts["count"] == 3
    expected = _rng(11)
    assert sleep.delays == [policy.delay(1, expected), policy.delay(2, expected)]

    calls = {"count": 0}

    async def _permanent() -> str:
        calls["count"] += 1
        raise InvalidInputError("no")

    quiet = _Sleep()
    with pytest.raises(InvalidInputError):
        await retry_async(
            "logs",
            _permanent,
            policy=policy,
            sleep=quiet,
            rng=_rng(1),
            logger=get_logger(),
        )
    assert calls["count"] == 1
    assert quiet.delays == []


async def test_retry_async_reraises_the_last_transient_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="link2kaggle_colab.jobs")
    sleep = _Sleep()
    calls = {"count": 0}

    async def _always() -> str:
        calls["count"] += 1
        raise ConnectionError("still down")

    with pytest.raises(ConnectionError, match="still down"):
        await retry_async(
            "download",
            _always,
            policy=RetryPolicy(max_attempts=3, base_delay=0.25, max_delay=8.0),
            sleep=sleep,
            rng=_rng(2),
            logger=get_logger(),
        )
    assert calls["count"] == 3
    assert len(sleep.delays) == 2
    assert "download_retry" in caplog.text
    assert "still down" in caplog.text
