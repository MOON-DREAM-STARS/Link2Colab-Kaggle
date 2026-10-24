"""Per-account quota cache and real CLI environments. Offline only."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pytest

from link2kaggle_colab.accounts.models import Account
from link2kaggle_colab.accounts.registry import AccountRegistry
from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.envspec import AccountEnv
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.providers.colab.cli_provider import ColabCliProvider
from link2kaggle_colab.providers.kaggle.cli_provider import KaggleCliProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.providers.registry import build_provider
from tests.mocks.runner import RecordingRunner


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
                    "mock_scenario": "quota_45h",
                    "default": True,
                },
                {
                    "id": "kaggle-b",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-b-ref",
                    "mock_scenario": "quota_insufficient",
                },
                {
                    "id": "colab-a",
                    "provider": "colab",
                    "credential_ref": "colab-a-ref",
                    "mock_scenario": "ai_pro",
                    "default": True,
                },
                {
                    "id": "colab-b",
                    "provider": "colab",
                    "credential_ref": "colab-b-ref",
                    "mock_scenario": "free",
                },
            ],
            "credentials": {
                "kaggle-a-ref": {"source": "env", "env": "KAGGLE_A_API_TOKEN"},
                "kaggle-b-ref": {"source": "env", "env": "KAGGLE_B_API_TOKEN"},
                "colab-a-ref": {"source": "env", "env": "COLAB_A_API_TOKEN"},
                "colab-b-ref": {"source": "env", "env": "COLAB_B_API_TOKEN"},
            },
        }
    )


def test_quota_cache_is_isolated_by_account(
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
    assert first.remaining == 45
    assert first.is_fixture is True
    assert "TEST FIXTURE" in first.message
    assert counts["kaggle-a"] == 1

    cached = client.get_quota(account="kaggle-a")[0]
    assert cached.remaining == 45
    assert counts["kaggle-a"] == 1

    second = client.get_quota(account="kaggle-b")[0]
    assert second.account_id == "kaggle-b"
    assert second.remaining == 0
    assert second.is_fixture is True
    assert "TEST FIXTURE" in second.message
    assert counts["kaggle-b"] == 1
    assert counts["kaggle-a"] == 1
    assert (ProviderName.kaggle, "kaggle-a") in client._quota_cache._entries
    assert (ProviderName.kaggle, "kaggle-b") in client._quota_cache._entries

    client._quota_cache.invalidate(ProviderName.kaggle, "kaggle-a")
    assert (ProviderName.kaggle, "kaggle-a") not in client._quota_cache._entries
    assert (ProviderName.kaggle, "kaggle-b") in client._quota_cache._entries

    reloaded = client.get_quota(account="kaggle-a")[0]
    assert reloaded.remaining == 45
    assert counts["kaggle-a"] == 2
    kept = client.get_quota(account="kaggle-b")[0]
    assert kept.remaining == 0
    assert counts["kaggle-b"] == 1


def test_real_cli_environments_stay_per_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_marker = "process-token-must-not-leak"
    monkeypatch.setenv("KAGGLE_API_TOKEN", process_marker)
    adc = tmp_path / "process-adc.json"
    adc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    before = dict(os.environ)

    kaggle_a = tmp_path / "kaggle-a"
    kaggle_b = tmp_path / "kaggle-b"
    colab_home = tmp_path / "colab-home"
    colab_adc = tmp_path / "colab-b-adc.json"
    for path in (kaggle_a, kaggle_b, colab_home):
        path.mkdir()
    settings = Settings.model_validate(
        {
            "mode": "real",
            "allow_real_jobs": True,
            "state_dir": tmp_path / "state",
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-a-ref",
                    "default": True,
                },
                {
                    "id": "kaggle-b",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-b-ref",
                },
                {
                    "id": "colab-home",
                    "provider": "colab",
                    "credential_ref": "colab-home-ref",
                    "default": True,
                },
                {
                    "id": "colab-adc",
                    "provider": "colab",
                    "credential_ref": "colab-adc-ref",
                },
            ],
            "credentials": {
                "kaggle-a-ref": {"source": "kaggle_config_dir", "path": str(kaggle_a)},
                "kaggle-b-ref": {"source": "kaggle_config_dir", "path": str(kaggle_b)},
                "colab-home-ref": {"source": "colab_home", "path": str(colab_home)},
                "colab-adc-ref": {"source": "adc_file", "path": str(colab_adc)},
            },
        }
    )
    registry = AccountRegistry.from_settings(settings)
    runner = RecordingRunner()
    runner.script(["kaggle"], stdout="{}", permanent=True)
    runner.script(["colab"], stdout="balance unknown\n", permanent=True)

    built: list[tuple[Account, KaggleCliProvider | ColabCliProvider]] = []
    for account in registry.list():
        if account.provider is ProviderName.fake:
            continue
        provider = build_provider(
            account.provider,
            settings,
            runner=runner,
            account=account,
        )
        assert isinstance(provider, KaggleCliProvider | ColabCliProvider)
        provider.auth_status()
        built.append((account, provider))
    assert runner.calls == []

    for _account, provider in built:
        provider.get_quota()

    assert dict(os.environ) == before
    assert os.environ["KAGGLE_API_TOKEN"] == process_marker
    envs = [item["env"] for item in runner.invocations]
    assert envs
    assert all(isinstance(item, AccountEnv) for item in envs)
    by_marker: dict[str, AccountEnv] = {}
    for env in envs:
        assert isinstance(env, AccountEnv)
        assert process_marker not in env.set.values()
        assert str(adc) not in env.set.values()
        if "KAGGLE_CONFIG_DIR" in env.set:
            by_marker[env.set["KAGGLE_CONFIG_DIR"]] = env
        elif "HOME" in env.set:
            by_marker[env.set["HOME"]] = env
        elif "GOOGLE_APPLICATION_CREDENTIALS" in env.set:
            by_marker[env.set["GOOGLE_APPLICATION_CREDENTIALS"]] = env
    assert set(by_marker) == {str(kaggle_a), str(kaggle_b), str(colab_home), str(colab_adc)}
    assert "KAGGLE_API_TOKEN" in by_marker[str(kaggle_a)].unset
    assert "GOOGLE_APPLICATION_CREDENTIALS" in by_marker[str(colab_home)].unset
    assert by_marker[str(kaggle_a)].set["KAGGLE_CONFIG_DIR"] != by_marker[str(kaggle_b)].set[
        "KAGGLE_CONFIG_DIR"
    ]
