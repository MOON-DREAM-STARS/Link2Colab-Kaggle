"""Quota cache TTL, invalidation, and unknown snapshots."""

from collections.abc import Callable

import pytest

from link2kaggle_colab.core.models import ProviderName, QuotaInfo, QuotaUnit
from link2kaggle_colab.quota.cache import QuotaCache


def _clock(now: list[float]) -> Callable[[], float]:
    def clock() -> float:
        return now[0]

    return clock


def _hours(remaining: float) -> QuotaInfo:
    """TEST FIXTURE balance. Not a live Kaggle quota."""
    return QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.gpu_hours,
        total=30,
        used=30 - remaining,
        remaining=remaining,
        is_fixture=True,
        source="test fixture",
        message="TEST FIXTURE quota",
    )


def test_hit_within_ttl_calls_loader_once() -> None:
    now = [1000.0]
    calls = 0

    def loader() -> QuotaInfo:
        nonlocal calls
        calls += 1
        return _hours(12)

    cache = QuotaCache(ttl_seconds=50, clock=_clock(now))
    first = cache.get(ProviderName.kaggle, loader)
    assert calls == 1
    assert first.from_cache is False
    assert first.fetched_at == 1000.0
    assert first.age_seconds == 0

    now[0] = 1049.0
    second = cache.get(ProviderName.kaggle, loader)
    assert calls == 1
    assert second.from_cache is True
    assert second.fetched_at == 1000.0
    assert second.age_seconds == 49
    assert second.quota.remaining == 12


def test_expiry_at_exactly_ttl_reloads() -> None:
    now = [0.0]
    calls = 0

    def loader() -> QuotaInfo:
        nonlocal calls
        calls += 1
        return _hours(float(calls))

    cache = QuotaCache(ttl_seconds=30, clock=_clock(now))
    cache.get(ProviderName.kaggle, loader)
    now[0] = 30.0
    second = cache.get(ProviderName.kaggle, loader)
    assert calls == 2
    assert second.from_cache is False
    assert second.fetched_at == 30.0
    assert second.quota.remaining == 2


def test_ttl_zero_always_reloads() -> None:
    now = [5.0]
    calls = 0

    def loader() -> QuotaInfo:
        nonlocal calls
        calls += 1
        return _hours(1)

    cache = QuotaCache(ttl_seconds=0, clock=_clock(now))
    first = cache.get(ProviderName.kaggle, loader)
    second = cache.get(ProviderName.kaggle, loader)
    assert calls == 2
    assert first.from_cache is False
    assert second.from_cache is False


def test_invalidate_one_provider_and_all() -> None:
    now = [0.0]
    calls = {"kaggle": 0, "colab": 0}

    def kaggle_loader() -> QuotaInfo:
        calls["kaggle"] += 1
        return _hours(4)

    def colab_loader() -> QuotaInfo:
        calls["colab"] += 1
        return QuotaInfo(
            provider=ProviderName.colab,
            unit=QuotaUnit.compute_units,
            remaining=80,
            is_fixture=True,
            source="test fixture",
            message="TEST FIXTURE balance",
        )

    cache = QuotaCache(ttl_seconds=300, clock=_clock(now))
    cache.get(ProviderName.kaggle, kaggle_loader)
    cache.get(ProviderName.colab, colab_loader)
    cache.invalidate(ProviderName.kaggle)
    kaggle = cache.get(ProviderName.kaggle, kaggle_loader)
    colab = cache.get(ProviderName.colab, colab_loader)
    assert calls == {"kaggle": 2, "colab": 1}
    assert kaggle.from_cache is False
    assert colab.from_cache is True

    cache.invalidate(None)
    cache.get(ProviderName.kaggle, kaggle_loader)
    cache.get(ProviderName.colab, colab_loader)
    assert calls == {"kaggle": 3, "colab": 2}


def test_loader_exception_is_not_cached() -> None:
    now = [0.0]
    calls = 0

    def loader() -> QuotaInfo:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary")
        return _hours(9)

    cache = QuotaCache(ttl_seconds=300, clock=_clock(now))
    with pytest.raises(RuntimeError, match="temporary"):
        cache.get(ProviderName.kaggle, loader)
    loaded = cache.get(ProviderName.kaggle, loader)
    assert calls == 2
    assert loaded.from_cache is False
    assert loaded.quota.remaining == 9


def test_unknown_quota_is_cached_and_stays_unknown() -> None:
    now = [10.0]
    calls = 0

    def loader() -> QuotaInfo:
        nonlocal calls
        calls += 1
        return QuotaInfo.unknown(ProviderName.colab, "could not parse usage")

    cache = QuotaCache(ttl_seconds=300, clock=_clock(now))
    first = cache.get(ProviderName.colab, loader)
    now[0] = 40.0
    second = cache.get(ProviderName.colab, loader)
    assert calls == 1
    assert first.from_cache is False
    assert first.quota.remaining is None
    assert second.from_cache is True
    assert second.age_seconds == 30
    assert second.quota.remaining is None
    assert second.quota.known is False
