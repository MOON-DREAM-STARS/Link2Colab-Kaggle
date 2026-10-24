"""CLI placement in mock mode. No network and no real CLIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from link2kaggle_colab.cli.main import app

runner = CliRunner()


def _write_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scheduler: dict[str, object] | None = None,
    **mock: str,
) -> Path:
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    state = tmp_path / "state"
    body: dict[str, object] = {
        "mode": "mock",
        "allow_real_jobs": False,
        "state_dir": str(state),
        "mock": {
            "kaggle_scenario": mock.get("kaggle_scenario", "ok"),
            "colab_scenario": mock.get("colab_scenario", "ok"),
        },
    }
    if scheduler is not None:
        body["scheduler"] = scheduler
    path = tmp_path / "l2kc.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _run(config: Path, *args: str):
    return runner.invoke(app, ["--config", str(config), *args])


def test_cli_recommend_shows_reasons_and_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(
        tmp_path,
        monkeypatch,
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
    )
    result = _run(
        config,
        "recommend",
        "--gpu",
        "T4",
        "--gpu-count",
        "1",
        "--timeout",
        "3600",
        "--provider",
        "auto",
    )
    assert result.exit_code == 0, result.stderr
    assert "provider: colab" in result.stdout
    assert "candidate: colab:T4" in result.stdout
    assert "estimated_ccu:" in result.stdout
    assert "fallback_used: true" in result.stdout
    assert "0 GPU-hours remaining" in result.stdout
    assert "TEST FIXTURE" in result.stdout
    assert "reasons:" in result.stdout


def test_cli_submit_provider_auto_uses_kaggle_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    result = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "auto",
        "--gpu-count",
        "1",
        "--timeout",
        "3600",
    )
    assert result.exit_code == 0, result.stderr
    job_id = result.stdout.splitlines()[0].strip()
    assert len(job_id) == 32
    assert "provider=kaggle" in result.stderr
    assert "candidate=kaggle:T4x2" in result.stderr
    assert "fallback_used=false" in result.stderr
    status = _run(config, "status", job_id)
    assert status.exit_code == 0, status.stderr
    assert "provider: kaggle" in status.stdout


def test_cli_premium_blocked_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    for provider in ("colab", "auto"):
        result = _run(
            config,
            "submit",
            str(py_script),
            "--provider",
            provider,
            "--gpu",
            "A100",
            "--allow-premium",
            "--timeout",
            "3600",
        )
        assert result.exit_code != 0
        assert "error[premium_not_allowed]:" in result.stderr
        assert result.stdout.strip() == ""


def test_cli_confirm_premium_alias_with_config_allows_a100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(
        tmp_path,
        monkeypatch,
        colab_scenario="ai_pro",
        scheduler={"allow_premium_gpu": True},
    )
    result = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "colab",
        "--gpu",
        "A100",
        "--confirm-premium",
        "--timeout",
        "3600",
    )
    assert result.exit_code == 0, result.stderr
    job_id = result.stdout.splitlines()[0].strip()
    assert len(job_id) == 32
    status = _run(config, "status", job_id)
    assert "provider: colab" in status.stdout
    logs = _run(config, "logs", job_id)
    assert "gpu=A100" in logs.stdout


def test_cli_rejects_gpu_count_zero_with_concrete_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    result = _run(
        config,
        "--json",
        "submit",
        str(py_script),
        "--provider",
        "colab",
        "--gpu",
        "T4",
        "--gpu-count",
        "0",
    )
    assert result.exit_code != 0
    assert "error[invalid_input]:" in result.stderr
    assert "gpu_count 0 requests CPU" in result.stderr


def test_cli_colab_cpu_requires_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    result = _run(
        config,
        "submit",
        str(py_script),
        "--provider",
        "colab",
        "--gpu-count",
        "0",
    )
    assert result.exit_code != 0
    assert "error[invalid_input]:" in result.stderr
    assert "timeout_seconds is required for Colab jobs" in result.stderr


def test_cli_submit_json_includes_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    py_script: Path,
) -> None:
    config = _write_config(tmp_path, monkeypatch)
    result = _run(
        config,
        "--json",
        "submit",
        str(py_script),
        "--provider",
        "auto",
        "--gpu",
        "T4",
        "--gpu-count",
        "1",
        "--timeout",
        "3600",
    )
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["job"]["provider"] == "kaggle"
    assert len(payload["job"]["job_id"]) == 32
    assert payload["decision"]["provider"] == "kaggle"
    assert payload["decision"]["candidate_id"] == "kaggle:T4x2"
    assert payload["decision"]["fallback_used"] is False

def test_cli_recommend_tpu_and_high_mem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_config(
        tmp_path,
        monkeypatch,
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
        scheduler={
            "allow_premium_gpu": True,
            "min_colab_ccu_reserve": 0,
            "max_ccu_per_job": 100,
        },
    )
    tpu = _run(
        config,
        "--json",
        "recommend",
        "--gpu",
        "TPU_V5E1",
        "--timeout",
        "3600",
        "--provider",
        "colab",
        "--allow-premium",
    )
    assert tpu.exit_code == 0, tpu.stderr
    payload = json.loads(tpu.stdout)
    assert payload["provider"] == "colab"
    assert payload["candidate_id"] == "colab:TPU-v5e1"
    assert payload["estimated_ccu"] == 4.5

    high = _run(
        config,
        "--json",
        "recommend",
        "--gpu",
        "T4",
        "--timeout",
        "3600",
        "--provider",
        "colab",
        "--high-mem",
    )
    assert high.exit_code == 0, high.stderr
    high_payload = json.loads(high.stdout)
    assert high_payload["candidate_id"] == "colab:T4"
    assert high_payload["estimated_ccu"] == 1.45

