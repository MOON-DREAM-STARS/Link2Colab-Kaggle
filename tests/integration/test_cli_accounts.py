"""CLI account commands. Offline mock mode. Credential values must not be printed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from link2kaggle_colab.cli.main import app
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider

runner = CliRunner()
_SENTINEL = "SENTINEL_l2kc_secret_value_9f3a"
_SENTINEL_ENV = "KAGGLE_SENTINEL_TOKEN"


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: dict[str, object]) -> Path:
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    state = tmp_path / "state"
    payload = {
        "mode": "mock",
        "allow_real_jobs": False,
        "state_dir": str(state),
        **body,
    }
    path = tmp_path / "l2kc.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _accounts_body() -> dict[str, object]:
    return {
        "accounts": [
            {
                "id": "kaggle-a",
                "provider": "kaggle",
                "credential_ref": "kaggle-a-ref",
                "default": True,
                "labels": {"team": "alpha"},
                "mock_scenario": "quota_45h",
            },
            {
                "id": "kaggle-b",
                "provider": "kaggle",
                "credential_ref": "kaggle-b-ref",
                "labels": {"team": "beta"},
                "mock_scenario": "ok",
            },
            {
                "id": "colab-a",
                "provider": "colab",
                "credential_ref": "colab-a-ref",
                "default": True,
                "labels": {"team": "colab"},
                "mock_scenario": "ai_pro",
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


def _run(config: Path, *args: str):
    return runner.invoke(app, ["--config", str(config), *args])


def _json(config: Path, *args: str) -> tuple[object, object]:
    """``--json`` is a root option, so it stays before the command."""
    result = _run(config, "--json", *args)
    assert result.exit_code == 0, result.stderr
    assert "MOCK MODE" not in result.stdout
    return result, json.loads(result.stdout)


def test_accounts_list_text_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, monkeypatch, _accounts_body())
    text = _run(config, "accounts", "list")
    assert text.exit_code == 0, text.stderr
    assert "MOCK MODE" in text.stderr
    assert "MOCK MODE" not in text.stdout
    header = text.stdout.splitlines()[0]
    for column in ("id", "provider", "enabled", "default", "credential"):
        assert column in header
    assert "kaggle-a" in text.stdout
    assert "kaggle-b" in text.stdout
    assert "env missing" in text.stdout
    assert "fake-default" in text.stdout
    assert "ambient present" in text.stdout

    _result, payload = _json(config, "accounts", "list")
    assert isinstance(payload, list)
    by_id = {row["id"]: row for row in payload}
    assert set(by_id["kaggle-a"]) == {"id", "provider", "enabled", "default", "credential"}
    assert by_id["kaggle-a"]["provider"] == "kaggle"
    assert by_id["kaggle-a"]["enabled"] is True
    assert by_id["kaggle-a"]["default"] is True
    assert by_id["kaggle-b"]["default"] is False
    assert by_id["kaggle-a"]["credential"] == {
        "kind": "env",
        "state": "missing",
        "source": "env:KAGGLE_A_API_TOKEN",
    }
    assert by_id["fake-default"]["credential"]["kind"] == "ambient"
    assert by_id["fake-default"]["credential"]["state"] == "present"
    assert "KAGGLE_A_API_TOKEN" not in json.dumps(by_id["kaggle-a"]["credential"]["state"])


def test_accounts_show_text_and_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, monkeypatch, _accounts_body())
    text = _run(config, "accounts", "show", "kaggle-a")
    assert text.exit_code == 0, text.stderr
    assert "id: kaggle-a" in text.stdout
    assert "provider: kaggle" in text.stdout
    assert "enabled: true" in text.stdout
    assert "default: true" in text.stdout
    assert "labels: team=alpha" in text.stdout
    assert "credential_kind: env" in text.stdout
    assert "credential_state: missing" in text.stdout
    assert "credential_source: env:KAGGLE_A_API_TOKEN" in text.stdout
    assert "TEST FIXTURE" in text.stdout
    assert "auth_state: authenticated" in text.stdout
    assert "MOCK MODE" not in text.stdout

    _result, payload = _json(config, "accounts", "show", "kaggle-a")
    assert isinstance(payload, dict)
    assert payload["id"] == "kaggle-a"
    assert payload["labels"] == {"team": "alpha"}
    assert payload["credential"]["state"] == "missing"
    assert payload["credential"]["kind"] == "env"
    assert "TEST FIXTURE" in payload["fixture_notice"]
    assert payload["auth_status"]["account_id"] == "kaggle-a"
    assert payload["auth_status"]["is_mock"] is True
    assert "TEST FIXTURE" in payload["auth_status"]["message"]

    missing = _run(config, "accounts", "show", "missing-account")
    assert missing.exit_code == 2
    assert missing.stdout.strip() == ""
    assert "error[account_not_found]:" in missing.stderr


def test_accounts_enable_disable_text_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, monkeypatch, _accounts_body())
    disabled = _run(config, "accounts", "disable", "kaggle-b")
    assert disabled.exit_code == 0, disabled.stderr
    assert "id: kaggle-b" in disabled.stdout
    assert "enabled: false" in disabled.stdout
    assert "MOCK MODE" not in disabled.stdout

    _result, payload = _json(config, "accounts", "disable", "colab-b")
    assert isinstance(payload, dict)
    assert payload["id"] == "colab-b"
    assert payload["enabled"] is False
    assert payload["provider"] == "colab"

    listed = json.loads(_run(config, "--json", "accounts", "list").stdout)
    by_id = {row["id"]: row for row in listed}
    assert by_id["kaggle-b"]["enabled"] is False
    assert by_id["colab-b"]["enabled"] is False
    assert by_id["kaggle-a"]["enabled"] is True

    enabled = _run(config, "--json", "accounts", "enable", "kaggle-b")
    assert enabled.exit_code == 0, enabled.stderr
    assert json.loads(enabled.stdout)["enabled"] is True
    text = _run(config, "accounts", "enable", "colab-b")
    assert text.exit_code == 0, text.stderr
    assert "enabled: true" in text.stdout
    again = json.loads(_run(config, "--json", "accounts", "list").stdout)
    restored = {row["id"]: row["enabled"] for row in again}
    assert restored["kaggle-b"] is True
    assert restored["colab-b"] is True

    unknown = _run(config, "accounts", "disable", "missing-account")
    assert unknown.exit_code == 2
    assert unknown.stdout.strip() == ""
    assert "error[account_not_found]:" in unknown.stderr


def test_accounts_enable_exits_when_provider_mask_keeps_it_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _accounts_body()
    body["providers"] = {"kaggle": {"enabled": False}}
    config = _write_config(tmp_path, monkeypatch, body)

    enabled = _run(config, "--json", "accounts", "enable", "kaggle-a")
    assert enabled.exit_code == 2, enabled.stderr
    payload = json.loads(enabled.stdout)
    assert payload["id"] == "kaggle-a"
    assert payload["provider"] == "kaggle"
    assert payload["enabled"] is False
    assert payload["reason"] == "provider_disabled"
    assert "providers.kaggle.enabled is false" in payload["warning"]
    assert "MOCK MODE" not in enabled.stdout

    text = _run(config, "accounts", "enable", "kaggle-b")
    assert text.exit_code == 2, text.stderr
    assert "enabled: false" in text.stdout
    assert "reason: provider_disabled" in text.stdout
    assert "warning: account remains disabled" in text.stdout

    disabled = _run(config, "--json", "accounts", "disable", "kaggle-a")
    assert disabled.exit_code == 0, disabled.stderr
    disabled_payload = json.loads(disabled.stdout)
    assert disabled_payload["enabled"] is False
    assert "reason" not in disabled_payload

    colab = _run(config, "--json", "accounts", "enable", "colab-a")
    assert colab.exit_code == 0, colab.stderr
    colab_payload = json.loads(colab.stdout)
    assert colab_payload["enabled"] is True
    assert "reason" not in colab_payload


def test_disable_account_submit_fails_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch, _accounts_body())
    assert _run(config, "accounts", "disable", "kaggle-b").exit_code == 0
    calls = {"n": 0}

    def _count_kaggle(self: MockKaggleProvider, spec: object) -> object:
        del self, spec
        calls["n"] += 1
        raise AssertionError("disabled account must not submit")

    def _count_colab(self: MockColabProvider, spec: object) -> object:
        del self, spec
        calls["n"] += 1
        raise AssertionError("disabled account must not submit")

    monkeypatch.setattr(MockKaggleProvider, "submit", _count_kaggle)
    monkeypatch.setattr(MockColabProvider, "submit", _count_colab)
    blocked = _run(
        config,
        "--json",
        "submit",
        str(py_script),
        "--account",
        "kaggle-b",
        "--provider",
        "kaggle",
        "--gpu",
        "T4",
        "--gpu-count",
        "2",
    )
    assert blocked.exit_code == 2
    assert json.loads(blocked.stdout)["code"] == "account_disabled"
    assert "error[account_disabled]:" in blocked.stderr
    assert calls["n"] == 0
    listed = json.loads(_run(config, "jobs", "list", "--json").stdout)
    assert listed == []


def test_account_wrong_provider_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch, _accounts_body())
    calls = {"n": 0}

    def _count(self: object, spec: object) -> object:
        del self, spec
        calls["n"] += 1
        raise AssertionError("provider mismatch must not submit")

    monkeypatch.setattr(MockKaggleProvider, "submit", _count)
    monkeypatch.setattr(MockColabProvider, "submit", _count)
    blocked = _run(
        config,
        "submit",
        str(py_script),
        "--account",
        "colab-a",
        "--provider",
        "kaggle",
        "--gpu",
        "T4",
        "--gpu-count",
        "2",
    )
    assert blocked.exit_code == 2
    assert blocked.stdout.strip() == ""
    assert "error[config_error]:" in blocked.stderr
    assert calls["n"] == 0
    quota = _run(config, "quota", "--account", "colab-a", "--provider", "kaggle")
    assert quota.exit_code == 2
    assert quota.stdout.strip() == ""
    assert "error[config_error]:" in quota.stderr
    recommend = _run(
        config,
        "recommend",
        "--account",
        "kaggle-b",
        "--provider",
        "colab",
        "--gpu",
        "T4",
        "--timeout",
        "3600",
    )
    assert recommend.exit_code == 2
    assert recommend.stdout.strip() == ""
    assert "error[config_error]:" in recommend.stderr


def test_secrets_are_never_printed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_SENTINEL_ENV, _SENTINEL)
    config = _write_config(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {
                    "id": "kaggle-sentinel",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-sentinel-ref",
                    "default": True,
                    "labels": {"team": "red"},
                    "mock_scenario": "ok",
                }
            ],
            "credentials": {
                "kaggle-sentinel-ref": {"source": "env", "env": _SENTINEL_ENV},
            },
        },
    )
    commands = [
        ("accounts", "list"),
        ("accounts", "list", "--json"),
        ("accounts", "show", "kaggle-sentinel"),
        ("accounts", "show", "kaggle-sentinel", "--json"),
        ("accounts", "disable", "kaggle-sentinel"),
        ("accounts", "enable", "kaggle-sentinel", "--json"),
        ("quota", "--account", "kaggle-sentinel"),
        ("--json", "quota", "--account", "kaggle-sentinel"),
        ("auth", "status", "--account", "kaggle-sentinel"),
        ("--json", "resources", "--account", "kaggle-sentinel"),
    ]
    saw_source = False
    for args in commands:
        result = _run(config, *args)
        assert result.exit_code == 0, result.stderr
        combined = result.stdout + result.stderr
        assert _SENTINEL not in combined
        if "env:KAGGLE_SENTINEL_TOKEN" in result.stdout or "env present" in result.stdout:
            saw_source = True
    assert saw_source is True


def test_account_flag_and_stored_account_on_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch, _accounts_body())
    _quota_a, kaggle_a = _json(config, "quota", "--account", "kaggle-a")
    assert isinstance(kaggle_a, list)
    assert kaggle_a[0]["account_id"] == "kaggle-a"
    assert kaggle_a[0]["remaining"] == 45
    assert kaggle_a[0]["is_fixture"] is True
    assert "TEST FIXTURE" in kaggle_a[0]["message"]
    _quota_b, kaggle_b = _json(config, "quota", "--account", "kaggle-b")
    assert isinstance(kaggle_b, list)
    assert kaggle_b[0]["remaining"] == 17.5

    auth = json.loads(_run(config, "--json", "auth", "status", "--account", "colab-a").stdout)
    assert auth[0]["account_id"] == "colab-a"
    resources = json.loads(_run(config, "--json", "resources", "--account", "kaggle-b").stdout)
    assert resources
    assert all(row["account_id"] == "kaggle-b" and row["provider"] == "kaggle" for row in resources)

    _recommend, pinned = _json(
        config,
        "recommend",
        "--account",
        "kaggle-b",
        "--gpu",
        "T4",
        "--gpu-count",
        "2",
    )
    assert isinstance(pinned, dict)
    assert pinned["provider"] == "kaggle"
    assert pinned["account_id"] == "kaggle-b"
    assert any("17.5" in reason for reason in pinned["reasons"])

    submitted = _run(
        config,
        "submit",
        str(py_script),
        "--account",
        "colab-a",
        "--gpu",
        "T4",
        "--timeout",
        "3600",
    )
    assert submitted.exit_code == 0, submitted.stderr
    job_id = submitted.stdout.splitlines()[0].strip()
    assert "account_id: colab-a" in submitted.stdout
    assert "state: QUEUED" in submitted.stdout

    status = _run(config, "status", job_id)
    assert status.exit_code == 0, status.stderr
    assert "account_id: colab-a" in status.stdout
    status_json = json.loads(_run(config, "--json", "status", job_id).stdout)
    assert status_json["account_id"] == "colab-a"
    logs = _run(config, "logs", job_id)
    assert logs.exit_code == 0, logs.stderr
    assert "account_id: colab-a" in logs.stderr
    logs_json = json.loads(_run(config, "--json", "logs", job_id).stdout)
    assert logs_json["account_id"] == "colab-a"
    assert "MOCK" in logs_json["logs"]
