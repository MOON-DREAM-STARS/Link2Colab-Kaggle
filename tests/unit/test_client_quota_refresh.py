"""Client.get_quota(refresh=True) invalidates the quota cache."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider


def _settings(state_dir: Path) -> Settings:
    return Settings.model_validate(
        {
            "mode": "mock",
            "allow_real_jobs": False,
            "state_dir": state_dir,
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-a-ref",
                    "default": True,
                    "mock_scenario": "ok",
                }
            ],
            "credentials": {
                "kaggle-a-ref": {"source": "env", "env": "KAGGLE_A_API_TOKEN"},
            },
        }
    )


def test_get_quota_refresh_invalidates_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts: dict[str | None, int] = defaultdict(int)
    original = MockKaggleProvider.get_quota

    def wrapped(self: MockKaggleProvider):  # type: ignore[no-untyped-def]
        counts[self._account_id] += 1
        return original(self)

    monkeypatch.setattr(MockKaggleProvider, "get_quota", wrapped)
    client = Client(_settings(tmp_path / "state"))

    first = client.get_quota(account="kaggle-a")[0]
    assert first.account_id == "kaggle-a"
    assert counts["kaggle-a"] == 1
    cached = client.get_quota(account="kaggle-a")[0]
    assert cached.remaining == first.remaining
    assert counts["kaggle-a"] == 1

    refreshed = client.get_quota(account="kaggle-a", refresh=True)[0]
    assert refreshed.account_id == "kaggle-a"
    assert counts["kaggle-a"] == 2

    again = client.refresh_quota(account="kaggle-a")[0]
    assert again.account_id == "kaggle-a"
    assert counts["kaggle-a"] == 3
    assert (ProviderName.kaggle, "kaggle-a") in client._quota_cache._entries
