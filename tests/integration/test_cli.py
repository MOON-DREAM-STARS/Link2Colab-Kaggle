"""CLI commands in mock mode. No network and no real CLIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from link2kaggle_colab import __version__
from link2kaggle_colab.cli.main import app
from link2kaggle_colab.core.errors import SubmitOutcomeUnknownError

runner = CliRunner()


def _job_id(result: object) -> str:
    stdout = getattr(result, "stdout", "")
    return str(stdout).splitlines()[0].strip()


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **mock: str) -> Path:
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    state = tmp_path / "state"
    path = tmp_path / "l2kc.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mode": "mock",
                "allow_real_jobs": False,
                "state_dir": str(state),
                "artifact_root": str(tmp_path),
                "mock": {
                    "kaggle_scenario": mock.get("kaggle_scenario", "ok"),
                    "colab_scenario": mock.get("colab_scenario", "ok"),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(config: Path, *args: str):
    return runner.invoke(app, ["--config", str(config), *args])


def test_version_prints_package_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, monkeypatch)
    result = _run(config, "version")
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
    assert "MOCK MODE" in result.stderr
    assert "MOCK MODE" not in result.stdout


def test_every_command_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    doctor = _run(config, "doctor")
    assert doctor.exit_code == 0, doctor.stderr
    assert "mode: mock" in doctor.stdout
    assert "executed=false" in doctor.stdout
    assert "TEST FIXTURE" in doctor.stdout
    assert "MOCK MODE" in doctor.stderr

    doctor_json = runner.invoke(app, ["--json", "--config", str(config), "doctor"])
    payload = json.loads(doctor_json.stdout)
    assert payload["mode"] == "mock"
    assert payload["allow_real_jobs"] is False
    assert "MOCK MODE" not in doctor_json.stdout

    auth = _run(config, "auth", "status")
    assert auth.exit_code == 0, auth.stderr
    assert "kaggle" in auth.stdout
    assert "TEST FIXTURE" in auth.stdout
    one = _run(config, "auth", "status", "--provider", "colab")
    assert one.exit_code == 0
    assert "colab" in one.stdout
    assert "kaggle" not in one.stdout.splitlines()[0]

    providers = _run(config, "providers")
    assert providers.exit_code == 0
    assert "kaggle" in providers.stdout and "fake" in providers.stdout
    providers_json = json.loads(_run(config, "--json", "providers").stdout)
    assert {row["name"] for row in providers_json} == {"kaggle", "colab", "fake"}

    quota = _run(config, "quota")
    assert quota.exit_code == 0
    assert "TEST FIXTURE" in quota.stdout
    assert "kaggle" in _run(config, "quota", "--provider", "kaggle").stdout

    resources = _run(config, "resources")
    assert resources.exit_code == 0
    assert "gb_per_gpu=" in resources.stdout
    assert "kaggle" in _run(config, "resources", "--provider", "kaggle").stdout

    recommend = _run(config, "recommend", "--gpu", "T4", "--gpu-count", "2")
    assert recommend.exit_code == 0, recommend.stderr
    assert "provider: kaggle" in recommend.stdout
    assert "not a live account" in recommend.stdout
    parsed = json.loads(
        _run(config, "--json", "recommend", "--gpu", "T4", "--gpu-count", "2").stdout
    )
    assert parsed["provider"] == "kaggle"
    assert parsed["resource"]["gpu_count"] == 2

    submitted = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "colab",
        "--gpu",
        "T4",
        "--timeout",
        "3600",
    )
    assert submitted.exit_code == 0, submitted.stderr
    job_id = _job_id(submitted)
    assert len(job_id) == 32
    assert "MOCK MODE" not in submitted.stdout

    state = ""
    for _ in range(6):
        status = _run(config, "status", job_id)
        assert status.exit_code == 0, status.stderr
        if "state: SUCCEEDED" in status.stdout:
            state = "SUCCEEDED"
            break
    assert state == "SUCCEEDED"
    status_json = json.loads(_run(config, "--json", "status", job_id).stdout)
    assert status_json["state"] == "SUCCEEDED"
    assert status_json["is_mock"] is True

    logs = _run(config, "logs", job_id)
    assert logs.exit_code == 0
    assert "MOCK" in logs.stdout
    assert "TEST FIXTURE" in logs.stdout

    dest = tmp_path / "downloaded"
    downloaded = _run(config, "download", job_id, "--dest", str(dest))
    assert downloaded.exit_code == 0, downloaded.stderr
    assert (dest / "result.txt").is_file()
    assert "MOCK TEST FIXTURE" in (dest / "result.txt").read_text(encoding="utf-8")

    cancelled = _run(config, "cancel", job_id)
    assert cancelled.exit_code == 0, cancelled.stderr
    assert "state: SUCCEEDED" in cancelled.stdout

    live = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "colab",
        "--gpu",
        "T4",
        "--timeout",
        "3600",
    )
    assert live.exit_code == 0, live.stderr
    stopped = _run(config, "cancel", _job_id(live))
    assert stopped.exit_code == 0, stopped.stderr
    assert "state: CANCELLED" in stopped.stdout


def test_unknown_job_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, monkeypatch)
    result = _run(config, "status", "ab" * 16)
    assert result.exit_code == 2
    assert "error[job_not_found]:" in result.stderr
    assert result.stdout.strip() == ""
    missing = _run(config, "logs", "nope")
    assert missing.exit_code == 2
    assert "error[job_not_found]:" in missing.stderr


def test_kaggle_cancel_is_not_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    submitted = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "kaggle",
        "--gpu",
        "T4",
        "--gpu-count",
        "2",
    )
    assert submitted.exit_code == 0, submitted.stderr
    cancelled = _run(config, "cancel", _job_id(submitted))
    assert cancelled.exit_code == 2
    assert "error[not_supported]:" in cancelled.stderr


def test_quota_insufficient_scenario_recommends_colab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, monkeypatch, kaggle_scenario="quota_insufficient")
    recommend = _run(
        config,
        "recommend",
        "--gpu",
        "T4",
        "--gpu-count",
        "1",
        "--timeout",
        "3600",
    )
    assert recommend.exit_code == 0, recommend.stderr
    assert "provider: colab" in recommend.stdout
    quota = _run(config, "quota", "--provider", "kaggle")
    assert "remaining=0" in quota.stdout
    assert "TEST FIXTURE" in quota.stdout


def test_fake_submit_forwards_script_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, monkeypatch)
    script = tmp_path / "echo_args.py"
    script.write_text(
        "import sys\nprint('args=' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    submitted = _run(
        config,
        "submit",
        str(script),
        "--provider",
        "fake",
        "--gpu",
        "NONE",
        "--",
        "--alpha",
        "beta",
    )
    assert submitted.exit_code == 0, submitted.stderr
    job_id = _job_id(submitted)
    logs = _run(config, "logs", job_id)
    assert "args=--alpha beta" in logs.stdout
    default_dest = _run(config, "download", job_id)
    assert default_dest.exit_code == 0
    assert f"dest_dir: {tmp_path / job_id}" in default_dest.stdout


def test_unexpected_error_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, monkeypatch)

    def boom(self: object) -> dict[str, object]:
        del self
        raise RuntimeError("boom")

    monkeypatch.setattr("link2kaggle_colab.cli.main.Client.doctor", boom)
    result = _run(config, "doctor")
    assert result.exit_code == 1
    assert "error[internal_error]:" in result.stderr


def test_bad_script_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _write_config(tmp_path, monkeypatch)
    missing = tmp_path / "missing.py"
    result = _run(config, "submit", str(missing), "--provider", "fake")
    assert result.exit_code == 2
    assert "error[invalid_input]:" in result.stderr


def test_idempotency_key_jobs_list_and_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    script = tmp_path / "once.py"
    script.write_text("print('once')\n", encoding="utf-8")
    args = [
        "submit",
        str(script),
        "--provider",
        "fake",
        "--gpu",
        "NONE",
        "--gpu-count",
        "0",
        "--idempotency-key",
        "batch-1",
    ]
    first = _run(config, *args)
    assert first.exit_code == 0, first.stderr
    job_id = _job_id(first)
    assert len(job_id) == 32
    assert "state: QUEUED" in first.stdout
    assert "deduplicated" not in first.stdout
    assert "MOCK MODE" in first.stderr
    assert "MOCK MODE" not in first.stdout

    second = _run(config, "--json", *args)
    assert second.exit_code == 0, second.stderr
    payload = json.loads(second.stdout)
    assert payload["deduplicated"] is True
    assert payload["job"]["job_id"] == job_id
    assert payload["state"] == "QUEUED"
    assert "MOCK MODE" in second.stderr

    listed = json.loads(_run(config, "jobs", "list", "--json").stdout)
    assert [row["job_id"] for row in listed] == [job_id]
    assert listed[0]["provider"] == "fake"
    assert listed[0]["idempotency_key"] == "batch-1"
    queued = json.loads(
        _run(
            config,
            "jobs",
            "list",
            "--json",
            "--state",
            "QUEUED",
            "--provider",
            "fake",
            "--limit",
            "10",
        ).stdout
    )
    assert [row["job_id"] for row in queued] == [job_id]
    human = _run(config, "jobs", "list")
    assert human.exit_code == 0, human.stderr
    assert job_id in human.stdout
    assert "MOCK MODE" in human.stderr
    assert "MOCK MODE" not in human.stdout

    reconciled = json.loads(_run(config, "jobs", "reconcile", "--json", job_id).stdout)
    assert reconciled[0]["job_id"] == job_id
    assert reconciled[0]["ok"] is True
    assert _run(config, "jobs", "reconcile", job_id).exit_code == 0
    bad = _run(config, "jobs", "reconcile", "--json", "ab" * 16)
    assert bad.exit_code == 1
    assert json.loads(bad.stdout)[0]["ok"] is False
    assert "MOCK MODE" in bad.stderr
    assert "MOCK MODE" not in bad.stdout


def test_idempotency_key_rejects_a_different_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    script = tmp_path / "once.py"
    script.write_text("print('once')\n", encoding="utf-8")
    common = [
        "submit",
        str(script),
        "--provider",
        "fake",
        "--gpu",
        "NONE",
        "--gpu-count",
        "0",
        "--idempotency-key",
        "batch-mismatch",
    ]
    first = _run(config, *common, "--", "alpha")
    assert first.exit_code == 0, first.stderr
    job_id = _job_id(first)
    assert "state: QUEUED" in first.stdout

    second = _run(config, "--json", *common, "--", "beta")
    assert second.exit_code != 0
    assert "error[invalid_input]:" in second.stderr
    payload = json.loads(second.stdout)
    assert payload["code"] == "invalid_input"
    assert payload["details"]["reason"] == "spec_mismatch"

    listed = json.loads(_run(config, "jobs", "list", "--json").stdout)
    assert [row["job_id"] for row in listed] == [job_id]
    assert listed[0]["state"] == "QUEUED"


def test_ambiguous_submit_names_the_job_and_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    job_id = "cd" * 16

    def _raise(self: object, spec: object, account: str | None = None) -> object:
        del self, spec, account
        raise SubmitOutcomeUnknownError(
            "submit outcome is ambiguous; not resubmitted; run reconcile",
            provider="fake",
            details={"job_id": job_id, "job_state": "UNKNOWN"},
        )

    monkeypatch.setattr("link2kaggle_colab.cli.main.Client.submit_detailed", _raise)
    result = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "fake",
        "--gpu",
        "NONE",
        "--gpu-count",
        "0",
    )
    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "error[submit_outcome_unknown]:" in result.stderr
    assert f"job_id: {job_id}" in result.stderr
    assert "l2kc jobs reconcile" in result.stderr
    assert "will not be resubmitted" in result.stderr
