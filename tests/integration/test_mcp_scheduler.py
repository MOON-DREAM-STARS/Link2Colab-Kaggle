"""MCP placement tools. Mock quotas are TEST FIXTURES. No network."""

from __future__ import annotations

from pathlib import Path

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.mcp.server import gpu_recommend_backend, gpu_submit_job


def _client(tmp_path: Path) -> Client:
    return Client(Settings(mode="mock", allow_real_jobs=False, state_dir=tmp_path / "state"))


def test_mcp_rejects_gpu_count_zero_with_concrete_type(tmp_path: Path, py_script: Path) -> None:
    result = gpu_submit_job(
        _client(tmp_path),
        str(py_script),
        provider="colab",
        gpu_type="T4",
        gpu_count=0,
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_input"
    assert "gpu_count 0 requests CPU" in result["error"]["message"]


def test_mcp_colab_cpu_requires_timeout(tmp_path: Path, py_script: Path) -> None:
    result = gpu_submit_job(
        _client(tmp_path),
        str(py_script),
        provider="colab",
        gpu_count=0,
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_input"
    assert "timeout_seconds is required for Colab jobs" in result["error"]["message"]


def test_mcp_submit_provider_auto(tmp_path: Path, py_script: Path) -> None:
    result = gpu_submit_job(
        _client(tmp_path),
        str(py_script),
        provider="auto",
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert result["ok"] is True
    assert result["job"]["provider"] == "kaggle"
    assert result["decision"]["provider"] == "kaggle"
    assert result["decision"]["candidate_id"] == "kaggle:T4x2"


def test_mcp_premium_blocked_without_config(tmp_path: Path, py_script: Path) -> None:
    client = _client(tmp_path)
    for provider in ("auto", "colab"):
        result = gpu_submit_job(
            client,
            str(py_script),
            provider=provider,
            gpu_type="A100",
            allow_premium=True,
            timeout_seconds=3600,
        )
        assert result["ok"] is False
        assert result["error"]["code"] == "premium_not_allowed"


def test_mcp_recommend_with_timeout_and_reasons(tmp_path: Path) -> None:
    result = gpu_recommend_backend(
        _client(tmp_path),
        gpu_count=1,
        min_gpu_memory_gb_per_gpu=24,
        timeout_seconds=3600,
        provider="auto",
    )
    assert result["ok"] is True
    recommendation = result["recommendation"]
    assert recommendation["provider"] == "colab"
    assert recommendation["candidate_id"] == "colab:L4"
    assert recommendation["reasons"]
    rejected = [item for item in recommendation["evaluations"] if not item["accepted"]]
    assert rejected
    assert all(item["reasons"] for item in rejected)
    assert any("16 GB per GPU" in reason for reason in recommendation["reasons"])
