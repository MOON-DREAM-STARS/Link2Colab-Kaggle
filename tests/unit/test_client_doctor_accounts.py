"""Real-mode doctor auth follows the default account, not process HOME."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.models import AuthState, ProviderName
from tests.mocks.runner import RecordingRunner


def _forbid_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("doctor or auth_status executed a process")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr("link2kaggle_colab.core.process.subprocess.run", explode)


def _isolate_ambient_kaggle(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point HOME at a poisoned kaggle dir and drop process credential env."""
    monkeypatch.setenv("HOME", str(home))
    for name in (
        "KAGGLE_CONFIG_DIR",
        "KAGGLE_API_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_real_doctor_uses_account_kaggle_config_dir_not_ambient_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_processes(monkeypatch)
    account_dir = tmp_path / "account-kaggle"
    account_dir.mkdir()
    (account_dir / "kaggle.json").write_text("{}\n", encoding="utf-8")
    ambient_home = tmp_path / "ambient-home"
    ambient_kaggle = ambient_home / ".kaggle"
    ambient_kaggle.mkdir(parents=True)
    (ambient_kaggle / "kaggle.json").write_text("{}\n", encoding="utf-8")
    _isolate_ambient_kaggle(monkeypatch, ambient_home)

    settings = Settings.model_validate(
        {
            "mode": "real",
            "allow_real_jobs": True,
            "state_dir": str(tmp_path / "state"),
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-a-ref",
                    "default": True,
                }
            ],
            "credentials": {
                "kaggle-a-ref": {"source": "kaggle_config_dir", "path": str(account_dir)},
            },
        }
    )
    runner = RecordingRunner()
    runner.script(
        ["kaggle", "quota", "--format", "json"],
        stdout='[{"resource": "GPU", "total": 30, "used": 0, "remaining": 30}]\n',
        permanent=True,
    )
    client = Client(settings, runner=runner)

    report = client.doctor()
    assert runner.calls == []
    by_name = {row["name"]: row for row in report["providers"]}
    auth = by_name["kaggle"]["auth"]
    assert "error" not in auth
    assert auth["account_id"] == "kaggle-a"
    assert auth["is_mock"] is False
    assert auth["state"] == "unknown"
    source = auth["details"]["source"]
    assert "KAGGLE_CONFIG_DIR" in source or str(account_dir) in source
    assert "~/.kaggle" not in source
    assert by_name["kaggle"]["cli"]["executed"] is False
    assert by_name["colab"]["cli"]["executed"] is False

    status = client.auth_status(account="kaggle-a")[0]
    assert status.provider is ProviderName.kaggle
    assert status.account_id == "kaggle-a"
    assert status.state is AuthState.authenticated
    assert status.details["source"] == source
    assert runner.calls == [["kaggle", "quota", "--format", "json"]]
    overlay = runner.invocations[0]["env"]
    assert overlay is not None
    assert overlay.set["KAGGLE_CONFIG_DIR"] == str(account_dir)


def test_real_doctor_reports_disabled_default_without_a_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_processes(monkeypatch)
    disabled_dir = tmp_path / "disabled-kaggle"
    sibling_dir = tmp_path / "sibling-kaggle"
    disabled_dir.mkdir()
    sibling_dir.mkdir()
    (disabled_dir / "kaggle.json").write_text("{}\n", encoding="utf-8")
    (sibling_dir / "kaggle.json").write_text("{}\n", encoding="utf-8")
    _isolate_ambient_kaggle(monkeypatch, tmp_path / "empty-home")

    settings = Settings.model_validate(
        {
            "mode": "real",
            "allow_real_jobs": True,
            "state_dir": str(tmp_path / "state"),
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-a-ref",
                    "enabled": False,
                    "default": True,
                },
                {
                    "id": "kaggle-b",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-b-ref",
                },
            ],
            "credentials": {
                "kaggle-a-ref": {"source": "kaggle_config_dir", "path": str(disabled_dir)},
                "kaggle-b-ref": {"source": "kaggle_config_dir", "path": str(sibling_dir)},
            },
        }
    )
    client = Client(settings, runner=RecordingRunner())
    report = client.doctor()
    by_name = {row["name"]: row for row in report["providers"]}
    auth = by_name["kaggle"]["auth"]
    assert auth["error"]["code"] == "account_disabled"
    assert auth["error"]["details"]["account_id"] == "kaggle-a"
    blob = json.dumps(auth)
    assert "kaggle-b" not in blob
    assert str(sibling_dir) not in blob
    assert "error" not in by_name["fake"]["auth"]
    assert by_name["fake"]["auth"]["account_id"] == "fake-default"
    assert by_name["colab"]["auth"]["account_id"] == "colab-default"
