"""In-process quota snapshot cache.

A cached ``QuotaInfo`` is returned unchanged. ``QuotaInfo.unknown`` stays
unknown: this cache never invents ``remaining``. Loader failures are not stored.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from link2kaggle_colab.core.models import ProviderName, QuotaInfo


class CachedQuota(BaseModel):
    """One cache lookup. ``age_seconds`` is measured at the lookup, not at fetch."""

    model_config = ConfigDict(extra="forbid")

    quota: QuotaInfo
    fetched_at: float
    age_seconds: float
    from_cache: bool


class _Stored(NamedTuple):
    quota: QuotaInfo
    fetched_at: float


def _cache_key(provider: ProviderName, account_id: str | None) -> tuple[ProviderName, str | None]:
    return (ProviderName(provider), account_id)


class QuotaCache:
    """Reuse a provider snapshot until ``ttl_seconds`` has elapsed.

    Keys are ``(provider, account_id)``. ``account_id`` None is its own slot,
    used by callers that do not name an account. ``ttl_seconds`` of 0 calls
    ``loader`` on every ``get``. An entry is fresh only while
    ``clock() - fetched_at < ttl_seconds``; age equal to the TTL reloads.
    The clock defaults to ``time.monotonic``.
    """

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[tuple[ProviderName, str | None], _Stored] = {}

    def get(
        self,
        provider: ProviderName,
        loader: Callable[[], QuotaInfo],
        account_id: str | None = None,
    ) -> CachedQuota:
        """Return a fresh or cached snapshot for one provider account.

        ``loader`` runs only on a miss, a TTL expiry, or ``ttl_seconds == 0``.
        Exceptions from ``loader`` propagate and leave the previous entry, if
        any, unchanged. Another account of the same provider is a different key.
        """
        key = _cache_key(provider, account_id)
        now = self._clock()
        stored = self._entries.get(key)
        if (
            stored is not None
            and self._ttl_seconds > 0
            and (now - stored.fetched_at) < self._ttl_seconds
        ):
            return CachedQuota(
                quota=stored.quota,
                fetched_at=stored.fetched_at,
                age_seconds=now - stored.fetched_at,
                from_cache=True,
            )
        quota = loader()
        loaded_at = self._clock()
        self._entries[key] = _Stored(quota=quota, fetched_at=loaded_at)
        return CachedQuota(
            quota=quota,
            fetched_at=loaded_at,
            age_seconds=0.0,
            from_cache=False,
        )

    def invalidate(
        self,
        provider: ProviderName | None = None,
        account_id: str | None = None,
    ) -> None:
        """Drop cache entries.

        No provider drops every entry. A provider without ``account_id`` drops
        every account of that provider. A provider and ``account_id`` drop only
        that account.
        """
        if provider is None:
            self._entries.clear()
            return
        name = ProviderName(provider)
        if account_id is None:
            for key in [key for key in self._entries if key[0] == name]:
                self._entries.pop(key, None)
            return
        self._entries.pop(_cache_key(name, account_id), None)
