"""MCP tool functions and a local stdio smoke test. No network."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import anyio
import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import SchedulerSettings, Settings
from link2kaggle_colab.mcp.server import (
    ALIAS_NAMES,
    TOOL_NAMES,
    build_server,
    gpu_auth_status,
    gpu_cancel_job,
    gpu_doctor,
    gpu_download_artifacts,
    gpu_get_job_logs,
    gpu_get_job_status,
    gpu_get_quota,
    gpu_list_accounts,
    gpu_list_jobs,
    gpu_list_providers,
    gpu_list_resources,
    gpu_recommend_backend,
    gpu_reconcile_jobs,
    gpu_refresh_quota,
    gpu_submit_job,
    main,
)


def _client(tmp_path: Path) -> Client:
    return Client(
        Settings(
            mode="mock",
            allow_real_jobs=False,
            state_dir=tmp_path / "state",
            artifact_root=tmp_path,
        )
    )


def test_every_tool_function(tmp_path: Path, py_script: Path) -> None:
    client = _client(tmp_path)
    providers = gpu_list_providers(client)
    assert providers["ok"] is True
    assert providers["mode"] == "mock"
    assert providers["is_mock"] is True
    assert "TEST FIXTURE" in providers["fixture_notice"]
    assert {row["name"] for row in providers["providers"]} == {"kaggle", "colab", "fake"}

    doctor = gpu_doctor(client)
    assert doctor["ok"] is True
    assert doctor["report"]["mode"] == "mock"
    assert "providers" in doctor["report"]

    statuses = gpu_auth_status(client)
    assert statuses["ok"] is True
    assert len(statuses["statuses"]) == 3
    assert gpu_auth_status(client, "kaggle")["statuses"][0]["provider"] == "kaggle"

    quotas = gpu_get_quota(client)
    assert quotas["ok"] is True
    assert any(row["is_fixture"] for row in quotas["quotas"])
    colab_quota = gpu_get_quota(client, "colab")["quotas"][0]
    assert "TEST FIXTURE" in colab_quota["message"]
    assert colab_quota["eligible_gpus"] == ["T4", "L4", "A100", "G4", "TPU_V5E1", "TPU_V6E1"]
    refreshed = gpu_refresh_quota(client, "colab")
    assert refreshed["ok"] is True
    assert refreshed["quotas"][0]["provider"] == "colab"

    resources = gpu_list_resources(client, "kaggle")
    t4 = next(row for row in resources["resources"] if row["gpu_type"] == "T4")
    assert t4["gpu_count"] == 2
    assert t4["gpu_memory_gb_per_gpu"] == 16
    assert t4["eligible"] is None
    colab_resources = gpu_list_resources(client, "colab")
    colab_t4 = next(row for row in colab_resources["resources"] if row["gpu_type"] == "T4")
    assert colab_t4["eligible"] is True
    colab_h100 = next(row for row in colab_resources["resources"] if row["gpu_type"] == "H100")
    assert colab_h100["eligible"] is False

    recommended = gpu_recommend_backend(client, gpu_type="T4", gpu_count=2)
    assert recommended["recommendation"]["provider"] == "kaggle"
    blocked = gpu_recommend_backend(
        client,
        gpu_type="H100",
        allow_premium=False,
        timeout_seconds=3600,
        provider="colab",
    )
    assert blocked["recommendation"]["provider"] is None
    assert any(
        "premium" in reason and "allow_premium" in reason
        for reason in blocked["recommendation"]["reasons"]
    )
    premium_client = Client(
        Settings(
            mode="mock",
            allow_real_jobs=False,
            state_dir=tmp_path / "premium-state",
            artifact_root=tmp_path,
            scheduler=SchedulerSettings(allow_premium_gpu=True),
        )
    )
    allowed = gpu_recommend_backend(
        premium_client,
        gpu_type="A100",
        allow_premium=True,
        timeout_seconds=3600,
    )
    assert allowed["recommendation"]["provider"] == "colab"
    assert allowed["recommendation"]["candidate_id"] == "colab:A100-40GB"
    refused_h100 = gpu_recommend_backend(
        premium_client,
        gpu_type="H100",
        allow_premium=True,
        timeout_seconds=3600,
        provider="colab",
    )
    assert refused_h100["recommendation"]["provider"] is None
    assert any("no CCU rate" in reason for reason in refused_h100["recommendation"]["reasons"])
    cpu = gpu_recommend_backend(client, gpu_type="NONE", gpu_count=0)
    assert cpu["recommendation"]["provider"] == "fake"

    before_jobs = gpu_list_jobs(client)
    assert before_jobs["ok"] is True
    before_count = len(before_jobs["jobs"])
    dry = gpu_submit_job(
        client,
        str(py_script),
        "colab",
        gpu_type="T4",
        gpu_count=1,
        timeout_seconds=3600,
        dry_run=True,
    )
    assert dry["ok"] is True
    assert dry["dry_run"] is True
    assert dry["recommendation"]["provider"] == "colab"
    assert "estimated_ccu" in dry
    after_dry = gpu_list_jobs(client)
    assert len(after_dry["jobs"]) == before_count

    submitted = gpu_submit_job(
        client,
        str(py_script),
        "colab",
        gpu_type="T4",
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert submitted["ok"] is True
    job_id = submitted["job"]["job_id"]
    state = ""
    for _ in range(6):
        current = gpu_get_job_status(client, job_id)
        state = current["job"]["state"]
        if state == "SUCCEEDED":
            break
    assert state == "SUCCEEDED"

    logs = gpu_get_job_logs(client, job_id)
    assert "MOCK" in logs["logs"]
    assert logs["truncated"] is False
    tail = gpu_get_job_logs(client, job_id, max_chars=8)
    assert tail["truncated"] is True
    assert tail["logs"] == logs["logs"][-8:]

    dest = tmp_path / "artifacts"
    downloaded = gpu_download_artifacts(client, job_id, str(dest))
    assert downloaded["ok"] is True
    assert (dest / "result.txt").is_file()

    cancelled = gpu_cancel_job(client, job_id)
    assert cancelled["ok"] is True
    assert cancelled["job"]["state"] == "SUCCEEDED"

    live = gpu_submit_job(
        client,
        str(py_script),
        "colab",
        gpu_type="T4",
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert live["ok"] is True
    stopped = gpu_cancel_job(client, live["job"]["job_id"])
    assert stopped["ok"] is True
    assert stopped["job"]["state"] == "CANCELLED"

    missing = gpu_get_job_status(client, "c" * 32)
    assert missing["ok"] is False
    assert missing["error"]["code"] == "job_not_found"
    assert missing["mode"] == "mock"
    assert missing["is_mock"] is True

    bad = gpu_submit_job(client, str(tmp_path / "missing.py"), "fake")
    assert bad["ok"] is False
    assert bad["error"]["code"] == "invalid_input"


async def test_build_server_lists_every_tool(tmp_path: Path) -> None:
    server = build_server(_client(tmp_path))
    tools = await server.list_tools()
    names = {tool.name for tool in tools}
    assert names == set(TOOL_NAMES) | set(ALIAS_NAMES)
    assert len(TOOL_NAMES) == 15
    assert "gpu_doctor" in TOOL_NAMES
    assert "gpu_refresh_quota" in TOOL_NAMES
    assert "gpu_list_accounts" in names
    assert "gpu_list_jobs" in names
    assert "gpu_reconcile_jobs" in names
    assert "l2kc_doctor" in names
    assert "l2kc_refresh_quota" in names
    assert "l2kc_list_providers" in names
    for tool in tools:
        assert tool.description
        assert "TEST FIXTURE" in tool.description


def test_main_uses_stdio_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    class _Server:
        def run(self, transport: str = "stdio") -> None:
            seen["transport"] = transport

    monkeypatch.setattr("link2kaggle_colab.mcp.server.build_server", lambda client=None: _Server())
    main([])
    assert seen["transport"] == "stdio"


def test_main_refuses_remote_transport(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--transport", "sse"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "disabled" in err.lower() or "stdio" in err.lower()


def test_python_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("link2kaggle_colab.mcp.server.main", lambda: called.append("ran"))
    runpy.run_module("link2kaggle_colab.mcp", run_name="__main__")
    assert called == ["ran"]


def test_console_script_targets_server_main() -> None:
    import importlib.metadata as metadata

    matches = list(
        metadata.entry_points().select(group="console_scripts", name="link2kaggle-colab-mcp")
    )
    assert matches
    assert matches[0].value == "link2kaggle_colab.mcp.server:main"


async def test_job_tools_run_inside_a_loop(tmp_path: Path, py_script: Path) -> None:
    """Call the sync tools on the running loop. run_sync must not deadlock."""
    client = _client(tmp_path)
    submitted = gpu_submit_job(
        client,
        str(py_script),
        provider="fake",
        gpu_type="NONE",
        gpu_count=0,
        idempotency_key="loop-key",
    )
    assert submitted["ok"] is True
    assert submitted["mode"] == "mock"
    assert submitted["is_mock"] is True
    assert submitted["deduplicated"] is False
    job_id = submitted["job"]["job_id"]

    again = gpu_submit_job(
        client,
        str(py_script),
        provider="fake",
        gpu_type="NONE",
        gpu_count=0,
        idempotency_key="loop-key",
    )
    assert again["ok"] is True
    assert again["deduplicated"] is True
    assert again["job"]["job_id"] == job_id

    listed = gpu_list_jobs(client, provider="fake", state=["QUEUED"])
    assert listed["ok"] is True
    assert listed["mode"] == "mock"
    assert [row["job_id"] for row in listed["jobs"]] == [job_id]

    reports = gpu_reconcile_jobs(client, [job_id])
    assert reports["ok"] is True
    assert reports["is_mock"] is True
    assert reports["reports"][0]["job_id"] == job_id
    assert reports["reports"][0]["ok"] is True
    assert reports["reports"][0]["after"] == "SUCCEEDED"


def test_gpu_submit_job_rejects_a_different_spec(tmp_path: Path, py_script: Path) -> None:
    client = _client(tmp_path)
    first = gpu_submit_job(
        client,
        str(py_script),
        provider="fake",
        gpu_type="NONE",
        gpu_count=0,
        name="first",
        idempotency_key="mismatch-key",
    )
    assert first["ok"] is True
    job_id = first["job"]["job_id"]
    second = gpu_submit_job(
        client,
        str(py_script),
        provider="fake",
        gpu_type="NONE",
        gpu_count=0,
        name="second",
        idempotency_key="mismatch-key",
    )
    assert second["ok"] is False
    assert second["error"]["code"] == "invalid_input"
    assert second["error"]["details"]["reason"] == "spec_mismatch"
    listed = gpu_list_jobs(client)
    assert listed["ok"] is True
    assert [row["job_id"] for row in listed["jobs"]] == [job_id]


async def test_stdio_client_lists_providers(tmp_path: Path) -> None:
    """Spawn the server locally and call one tool. Bounded so a hang fails the test."""
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    config = tmp_path / "l2kc.yaml"
    state = tmp_path / "state"
    config.write_text(
        f"mode: mock\nallow_real_jobs: false\nstate_dir: {state}\n",
        encoding="utf-8",
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "link2kaggle_colab.mcp"],
        env={"L2KC_CONFIG": str(config), "L2KC_STATE_DIR": str(state)},
        cwd=str(tmp_path),
    )
    with anyio.fail_after(30):
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("gpu_list_providers", {})
    assert result.is_error is False
    assert result.content
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is True
    assert payload["mode"] == "mock"
    assert payload["is_mock"] is True
    assert "TEST FIXTURE" in payload["fixture_notice"]
    assert {row["name"] for row in payload["providers"]} == {"kaggle", "colab", "fake"}


def _multi_account_client(tmp_path: Path) -> Client:
    return Client(
        Settings.model_validate(
            {
                "mode": "mock",
                "allow_real_jobs": False,
                "state_dir": tmp_path / "state",
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
                ],
                "credentials": {
                    "kaggle-a-ref": {"source": "env", "env": "KAGGLE_A_API_TOKEN"},
                    "kaggle-b-ref": {"source": "env", "env": "KAGGLE_B_API_TOKEN"},
                    "colab-a-ref": {"source": "env", "env": "COLAB_A_API_TOKEN"},
                },
            }
        )
    )


def test_gpu_list_accounts(tmp_path: Path) -> None:
    client = _multi_account_client(tmp_path)
    listed = gpu_list_accounts(client)
    assert listed["ok"] is True
    assert listed["mode"] == "mock"
    assert listed["is_mock"] is True
    assert {row["id"] for row in listed["accounts"]} >= {"kaggle-a", "kaggle-b", "colab-a"}
    quota = gpu_get_quota(client, account="kaggle-b")
    assert quota["ok"] is True
    assert quota["quotas"][0]["account_id"] == "kaggle-b"
    assert quota["quotas"][0]["remaining"] == 17.5
    assert quota["quotas"][0]["is_fixture"] is True
    recommended = gpu_recommend_backend(
        client,
        gpu_type="T4",
        gpu_count=2,
        account="kaggle-b",
    )
    assert recommended["ok"] is True
    assert recommended["recommendation"]["provider"] == "kaggle"
    assert recommended["recommendation"]["account_id"] == "kaggle-b"
    auth = gpu_auth_status(client, account="colab-a")
    assert auth["ok"] is True
    assert auth["statuses"][0]["account_id"] == "colab-a"
    resources = gpu_list_resources(client, account="kaggle-a")
    assert resources["ok"] is True
    assert resources["resources"]
    assert all(row["account_id"] == "kaggle-a" for row in resources["resources"])


def test_gpu_submit_job_with_account_and_disabled_account(
    tmp_path: Path,
    py_script: Path,
) -> None:
    client = _multi_account_client(tmp_path)
    submitted = gpu_submit_job(
        client,
        str(py_script),
        provider="kaggle",
        gpu_type="T4",
        gpu_count=2,
        account="kaggle-b",
    )
    assert submitted["ok"] is True
    assert submitted["job"]["account_id"] == "kaggle-b"
    assert submitted["mode"] == "mock"
    job_id = submitted["job"]["job_id"]
    status = gpu_get_job_status(client, job_id)
    assert status["ok"] is True
    assert status["job"]["account_id"] == "kaggle-b"
    listed = gpu_list_jobs(client, provider="kaggle")
    assert listed["ok"] is True
    assert listed["jobs"][0]["account_id"] == "kaggle-b"

    client.disable_account("kaggle-b")
    blocked = gpu_submit_job(
        client,
        str(py_script),
        provider="kaggle",
        gpu_type="T4",
        gpu_count=2,
        account="kaggle-b",
    )
    assert blocked["ok"] is False
    assert blocked["error"]["code"] == "account_disabled"
    assert blocked["mode"] == "mock"
    assert blocked["is_mock"] is True
    again = gpu_list_jobs(client, provider="kaggle")
    assert [row["job_id"] for row in again["jobs"]] == [job_id]

def test_mcp_tpu_and_high_mem(tmp_path: Path, py_script: Path) -> None:
    client = Client(
        Settings(
            mode="mock",
            allow_real_jobs=False,
            state_dir=tmp_path / "state-tpu",
            artifact_root=tmp_path,
            scheduler=SchedulerSettings(allow_premium_gpu=True),
            mock={"colab_scenario": "ai_pro", "kaggle_scenario": "quota_insufficient"},
        )
    )
    resources = gpu_list_resources(client, "colab")
    types = {row["gpu_type"] for row in resources["resources"]}
    assert "TPU_V5E1" in types
    assert "TPU_V6E1" in types
    t4 = next(row for row in resources["resources"] if row["gpu_type"] == "T4")
    assert t4["supports_high_mem"] is True
    l4 = next(row for row in resources["resources"] if row["gpu_type"] == "L4")
    assert l4["supports_high_mem"] is False

    recommended = gpu_recommend_backend(
        client,
        gpu_type="TPU_V5E1",
        gpu_count=1,
        allow_premium=True,
        timeout_seconds=3600,
        provider="colab",
    )
    assert recommended["ok"] is True
    assert recommended["recommendation"]["provider"] == "colab"
    assert recommended["recommendation"]["candidate_id"] == "colab:TPU-v5e1"
    assert recommended["recommendation"]["estimated_ccu"] == 4.5

    high = gpu_recommend_backend(
        client,
        gpu_type="T4",
        gpu_count=1,
        high_mem=True,
        timeout_seconds=3600,
        provider="colab",
    )
    assert high["ok"] is True
    assert high["recommendation"]["candidate_id"] == "colab:T4"
    assert high["recommendation"]["estimated_ccu"] == 1.45

    submitted = gpu_submit_job(
        client,
        str(py_script),
        provider="colab",
        gpu_type="T4",
        high_mem=True,
        timeout_seconds=3600,
    )
    assert submitted["ok"] is True
    assert submitted["job"]["provider"] == "colab"

