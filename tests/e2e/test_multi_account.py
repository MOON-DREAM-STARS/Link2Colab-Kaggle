"""Two Kaggle and two Colab mock accounts. Offline. No network, GPU, or CCU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from link2kaggle_colab.cli.main import app
from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import load_config
from link2kaggle_colab.core.errors import AccountDisabledError
from link2kaggle_colab.core.models import JobState
from link2kaggle_colab.providers.mock_fixtures import MOCK_RESULT_TEXT

runner = CliRunner()


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    state = tmp_path / "state"
    payload = {
        "mode": "mock",
        "allow_real_jobs": False,
        "state_dir": str(state),
        "artifact_root": str(tmp_path),
        "accounts": [
            {
                "id": "kaggle-a",
                "provider": "kaggle",
                "credential_ref": "kaggle-a-ref",
                "default": True,
                "mock_scenario": "quota_45h",
            },
            {
                "id": "kaggle-b",
                "provider": "kaggle",
                "credential_ref": "kaggle-b-ref",
                "mock_scenario": "ok",
            },
            {
                "id": "colab-a",
                "provider": "colab",
                "credential_ref": "colab-a-ref",
                "default": True,
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
    path = tmp_path / "multi.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _run(config: Path, *args: str):
    return runner.invoke(app, ["--config", str(config), *args])


def _job_id(result: object) -> str:
    return str(getattr(result, "stdout", "")).splitlines()[0].strip()


def _quota(config: Path, account_id: str) -> dict[str, object]:
    result = _run(config, "--json", "quota", "--account", account_id)
    assert result.exit_code == 0, result.stderr
    assert "MOCK MODE" not in result.stdout
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["account_id"] == account_id
    assert payload[0]["is_fixture"] is True
    assert "TEST FIXTURE" in payload[0]["message"]
    return payload[0]


def test_multi_account_cli_quota_submit_restart_and_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    loaded = load_config(config)
    assert loaded.mode == "mock"
    assert loaded.accounts is not None
    assert [account.id for account in loaded.accounts] == [
        "kaggle-a",
        "kaggle-b",
        "colab-a",
        "colab-b",
    ]
    assert loaded.accounts[0].mock_scenario == "quota_45h"
    assert loaded.accounts[1].mock_scenario == "ok"
    assert loaded.accounts[2].mock_scenario == "ai_pro"
    assert loaded.accounts[3].mock_scenario == "free"

    kaggle_a = _quota(config, "kaggle-a")
    kaggle_b = _quota(config, "kaggle-b")
    colab_a = _quota(config, "colab-a")
    colab_b = _quota(config, "colab-b")
    assert kaggle_a["remaining"] == 45
    assert kaggle_b["remaining"] == 17.5
    assert colab_a["remaining"] == 200
    assert colab_b["remaining"] is None
    assert len({kaggle_a["remaining"], kaggle_b["remaining"], colab_a["remaining"]}) == 3

    submitted_kaggle = _run(
        config,
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
    assert submitted_kaggle.exit_code == 0, submitted_kaggle.stderr
    kaggle_id = _job_id(submitted_kaggle)
    assert "account_id: kaggle-b" in submitted_kaggle.stdout

    submitted_colab = _run(
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
    assert submitted_colab.exit_code == 0, submitted_colab.stderr
    colab_id = _job_id(submitted_colab)
    assert "account_id: colab-a" in submitted_colab.stdout

    listed = _run(config, "jobs", "list", "--json")
    assert listed.exit_code == 0, listed.stderr
    rows = {row["job_id"]: row for row in json.loads(listed.stdout)}
    assert rows[kaggle_id]["account_id"] == "kaggle-b"
    assert rows[colab_id]["account_id"] == "colab-a"
    human = _run(config, "jobs", "list")
    assert "kaggle-b" in human.stdout
    assert "colab-a" in human.stdout

    restarted = Client(config_path=config)
    restarted_rows = {row["job_id"]: row for row in restarted.list_jobs()}
    assert restarted_rows[kaggle_id]["account_id"] == "kaggle-b"
    assert restarted_rows[colab_id]["account_id"] == "colab-a"
    kaggle_mock = (
        loaded.state_dir
        / "accounts"
        / "kaggle-b"
        / "mock"
        / "kaggle"
        / f"{restarted_rows[kaggle_id]['provider_job_id']}.json"
    )
    colab_mock = (
        loaded.state_dir
        / "accounts"
        / "colab-a"
        / "mock"
        / "colab"
        / f"{restarted_rows[colab_id]['provider_job_id']}.json"
    )
    assert kaggle_mock.is_file()
    assert colab_mock.is_file()
    other = (
        loaded.state_dir
        / "accounts"
        / "kaggle-a"
        / "mock"
        / "kaggle"
        / f"{restarted_rows[kaggle_id]['provider_job_id']}.json"
    )
    assert not other.exists()

    for job_id, account_id in ((kaggle_id, "kaggle-b"), (colab_id, "colab-a")):
        seen: list[JobState] = []
        for _ in range(6):
            current = restarted.status(job_id)
            seen.append(current.state)
            if current.state is JobState.SUCCEEDED:
                break
        assert seen[-1] is JobState.SUCCEEDED
        dest = tmp_path / account_id
        artifacts = restarted.download(job_id, dest)
        assert artifacts[0].name == "result.txt"
        assert (dest / "result.txt").read_text(encoding="utf-8") == MOCK_RESULT_TEXT
        assert restarted_rows[job_id]["account_id"] == account_id

    held = _run(
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
    assert held.exit_code == 0, held.stderr
    held_id = _job_id(held)
    held_rows = Client(config_path=config).list_jobs()
    held_row = next(row for row in held_rows if row["job_id"] == held_id)
    assert held_row["state"] == "QUEUED"
    assert held_row["account_id"] == "colab-a"
    held_mock = (
        loaded.state_dir
        / "accounts"
        / "colab-a"
        / "mock"
        / "colab"
        / f"{held_row['provider_job_id']}.json"
    )
    before = json.loads(held_mock.read_text(encoding="utf-8"))
    assert before["state"] == "QUEUED"
    assert before["step_index"] == 0

    disabled = _run(config, "accounts", "disable", "colab-a")
    assert disabled.exit_code == 0, disabled.stderr
    reconciled = _run(config, "jobs", "reconcile", "--json")
    assert reconciled.exit_code == 0, reconciled.stderr
    reports = json.loads(reconciled.stdout)
    skipped = next(row for row in reports if row["job_id"] == held_id)
    assert skipped["skipped"] is True
    assert skipped["reason"] == "account_disabled"
    assert skipped["before"] == "QUEUED"
    assert skipped["after"] == "QUEUED"
    after = json.loads(held_mock.read_text(encoding="utf-8"))
    assert after["state"] == "QUEUED"
    assert after["step_index"] == 0

    after_restart = Client(config_path=config)
    kept = next(row for row in after_restart.list_jobs() if row["job_id"] == held_id)
    assert kept["state"] == "QUEUED"
    assert kept["account_id"] == "colab-a"
    assert after_restart.status(kaggle_id).state is JobState.SUCCEEDED
    with pytest.raises(AccountDisabledError) as caught:
        after_restart.status(colab_id)
    assert caught.value.code == "account_disabled"


def test_example_multi_account_config_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    path = Path(__file__).resolve().parents[2] / "examples" / "config.multi-account.yaml"
    text = path.read_text(encoding="utf-8")
    assert "KGAT_" not in text
    assert "ya29." not in text
    loaded = load_config(path)
    assert loaded.mode == "mock"
    assert loaded.allow_real_jobs is False
    assert loaded.accounts is not None
    assert [account.id for account in loaded.accounts] == [
        "kaggle-primary",
        "kaggle-alt",
        "colab-primary",
        "colab-alt",
    ]
    assert loaded.credentials["kaggle-alt"].source == "env"
    assert loaded.credentials["kaggle-alt"].env == "KAGGLE_ALT_API_TOKEN"
    assert loaded.credentials["colab-alt"].source == "colab_home"
    assert loaded.credentials["vault-x"].source == "keyring"
    assert loaded.accounts[0].mock_scenario == "quota_45h"
