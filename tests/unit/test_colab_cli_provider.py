"""Colab CLI adapter tests. The real ``colab`` binary is never executed."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import (
    NotSupportedError,
    PremiumNotAllowed,
    ProviderCLIError,
    RealModeDisabledError,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import (
    AuthMethod,
    AuthState,
    GpuType,
    JobSpec,
    JobState,
    ProviderName,
    QuotaUnit,
)
from link2kaggle_colab.providers.colab.cli_provider import (
    ColabCliProvider,
    ColabJobRecord,
    _parse_usage,
)
from tests.mocks.runner import RecordingRunner

_USAGE = "Current balance: 12.5 compute units\nUsage rate: 1.8/hr\nActive assignments: 0\n"
_LS = "ok.txt\n../x\n-rf\nsub/dir.txt\n"


def _provider(
    tmp_path: Path,
    runner: RecordingRunner,
    *,
    mode: str = "real",
    allow: bool = True,
    auth: str = "adc",
    keep_runtime: bool = False,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
    default_exec_timeout_seconds: int | None = None,
    stop_backoff_seconds: float = 0.0,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[ColabCliProvider, Settings]:
    colab: dict[str, object] = {"auth": auth, "keep_runtime": keep_runtime}
    if default_exec_timeout_seconds is not None:
        colab["default_exec_timeout_seconds"] = default_exec_timeout_seconds
    settings = Settings(
        mode=mode,  # type: ignore[arg-type]
        allow_real_jobs=allow,
        state_dir=tmp_path / "state",
        providers={"colab": colab},
    )
    provider = ColabCliProvider(
        settings,
        runner=runner,
        home=home or (tmp_path / "home"),
        environ={} if environ is None else environ,
        stop_backoff_seconds=stop_backoff_seconds,
        sleeper=sleeper,
    )
    return provider, settings


def _record(settings: Settings, job_id: str) -> ColabJobRecord:
    path = settings.state_dir / "colab" / "jobs" / f"{job_id}.json"
    return ColabJobRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _only_record(settings: Settings) -> ColabJobRecord:
    jobs = sorted((settings.state_dir / "colab" / "jobs").glob("*.json"))
    assert len(jobs) == 1
    return ColabJobRecord.model_validate_json(jobs[0].read_text(encoding="utf-8"))


def _queue_stop(runner: RecordingRunner, kind: str) -> None:
    if kind == "returncode":
        runner.script(
            ["colab", "--auth", "adc", "stop"],
            returncode=1,
            stderr="runtime still allocated",
        )
        return
    runner.script(
        ["colab", "--auth", "adc", "stop"],
        error=ProviderCLIError("command timed out", code="cli_timeout", provider="colab"),
    )


def _plant_download(args: list[str]) -> None:
    Path(args[-1]).write_text("output-bytes", encoding="utf-8")


def _script_flow(
    runner: RecordingRunner,
    *,
    exec_error: BaseException | None = None,
    exec_rc: int = 0,
    exec_stdout: str = "hello\n",
    exec_stderr: str = "",
    ls: str = _LS,
) -> None:
    runner.script(["colab", "--auth", "adc", "new"])
    runner.script(["colab", "--auth", "adc", "upload"], permanent=True)
    if exec_error is not None:
        runner.script(["colab", "--auth", "adc", "exec"], error=exec_error)
    else:
        runner.script(
            ["colab", "--auth", "adc", "exec"],
            returncode=exec_rc,
            stdout=exec_stdout,
            stderr=exec_stderr,
        )
        runner.script(["colab", "--auth", "adc", "ls"], stdout=ls)
        runner.script(
            ["colab", "--auth", "adc", "download"],
            permanent=True,
            on_call=_plant_download,
        )
    runner.script(["colab", "--auth", "adc", "stop"])


def _session(calls: list[list[str]]) -> str:
    new = next(call for call in calls if call[3] == "new")
    return new[new.index("-s") + 1]


def test_real_mode_disabled_never_calls_the_runner(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, mode="mock", allow=True)
    spec = JobSpec(script_path=py_script, gpu_type=GpuType.T4)
    job_id = "a" * 32
    calls = (
        provider.detect_cli,
        provider.auth_status,
        provider.get_quota,
        provider.list_resources,
        provider.list_sessions,
        lambda: provider.submit(spec),
        lambda: provider.status(job_id),
        lambda: provider.logs(job_id),
        lambda: provider.download(job_id, tmp_path / "out"),
        lambda: provider.cancel(job_id),
    )
    for call in calls:
        with pytest.raises(RealModeDisabledError):
            call()
    assert runner.calls == []
    caps = provider.capabilities()
    assert caps.supports_cancel is True
    assert caps.supports_interactive_session is True
    assert caps.is_batch_only is False
    assert any("list_sessions" in note or "colab sessions" in note for note in caps.notes)


def test_submit_blocked_when_real_jobs_are_off_but_quota_is_allowed(
    tmp_path: Path, py_script: Path
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, allow=False)
    with pytest.raises(RealModeDisabledError):
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    assert runner.calls == []
    runner.script(["colab", "--auth", "adc", "usage"], stdout=_USAGE)
    quota = provider.get_quota()
    assert quota.known is True
    assert runner.calls == [["colab", "--auth", "adc", "usage"]]


@pytest.mark.parametrize("gpu", [GpuType.A100, GpuType.H100, GpuType.G4])
def test_premium_without_allow_premium_makes_no_cli_call(
    tmp_path: Path,
    py_script: Path,
    gpu: GpuType,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    with pytest.raises(PremiumNotAllowed) as caught:
        provider.submit(JobSpec(script_path=py_script, gpu_type=gpu))
    assert caught.value.code == "premium_not_allowed"
    assert runner.calls == []


def test_args_and_unknown_gpu_are_rejected_before_cli(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    with pytest.raises(NotSupportedError, match="does not forward args"):
        provider.submit(JobSpec(script_path=py_script, args=["--epochs", "1"]))
    with pytest.raises(NotSupportedError):
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.P100))
    assert runner.calls == []


def test_submit_argv_uploads_pulls_and_stops(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    data = tmp_path / "data.csv"
    data.write_text("a,b\n", encoding="utf-8")
    _script_flow(runner)
    handle = provider.submit(
        JobSpec(
            script_path=py_script,
            input_files=[data],
            gpu_type=GpuType.T4,
            timeout_seconds=15,
        )
    )
    session = _session(runner.calls)
    assert handle.remote_id == session
    assert handle.provider is ProviderName.colab
    assert session.startswith("l2kc-")
    assert [call[3] for call in runner.calls] == [
        "new",
        "upload",
        "exec",
        "ls",
        "download",
        "stop",
    ]
    assert runner.calls[0] == ["colab", "--auth", "adc", "new", "-s", session, "--gpu", "T4"]
    assert runner.calls[1] == [
        "colab",
        "--auth",
        "adc",
        "upload",
        "-s",
        session,
        str(data.resolve()),
        "data.csv",
    ]
    assert runner.calls[2] == [
        "colab",
        "--auth",
        "adc",
        "exec",
        "-s",
        session,
        "-f",
        str(py_script.resolve()),
    ]
    assert runner.calls[3] == ["colab", "--auth", "adc", "ls", "-s", session, "outputs"]
    download = runner.calls[4]
    assert download[:7] == [
        "colab",
        "--auth",
        "adc",
        "download",
        "-s",
        session,
        "outputs/ok.txt",
    ]
    assert Path(download[7]).is_absolute()
    assert runner.calls[5] == ["colab", "--auth", "adc", "stop", "-s", session]
    for call in runner.calls:
        assert call[1:3] == ["--auth", "adc"]
        assert call.index("--auth") < call.index(call[3])
    assert not any(part in {"../x", "-rf", "sub/dir.txt"} for call in runner.calls for part in call)
    exec_call = next(item for item in runner.invocations if item["args"][3] == "exec")  # type: ignore[index]
    assert exec_call["timeout"] == 15.0
    record_path = settings.state_dir / "colab" / "jobs" / f"{handle.job_id}.json"
    record = ColabJobRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
    assert record.state is JobState.SUCCEEDED
    assert record.runtime_stopped is True
    assert record.artifacts == ["ok.txt"]
    assert record.exit_code == 0
    before = len(runner.calls)
    assert provider.status(handle.job_id).state is JobState.SUCCEEDED
    assert len(runner.calls) == before

    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    outputs = Path(record.outputs_dir)
    (outputs / "escape").symlink_to(outside)
    dest = tmp_path / "dest"
    copied = provider.download(handle.job_id, dest)
    assert [item.name for item in copied] == ["ok.txt"]
    assert (dest / "ok.txt").read_text(encoding="utf-8") == "output-bytes"
    assert not (dest / "escape").exists()
    assert len(runner.calls) == before


def test_stop_runs_when_exec_times_out_or_raises(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(
        runner,
        exec_error=ProviderCLIError("timed out", code="cli_timeout", provider="colab"),
    )
    with pytest.raises(ProviderCLIError) as timed:
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4, timeout_seconds=15))
    assert timed.value.code == "cli_timeout"
    assert [call[3] for call in runner.calls] == ["new", "exec", "stop"]

    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner, exec_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert [call[3] for call in runner.calls] == ["new", "exec", "stop"]
    assert "--gpu" not in runner.calls[0]


@pytest.mark.parametrize(
    "stderr",
    ["GPU unavailable", "quota exceeded", "not eligible for this runtime"],
)
def test_new_failure_maps_to_resource_unavailable_and_still_stops(
    tmp_path: Path,
    py_script: Path,
    stderr: str,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["colab", "--auth", "adc", "new"], returncode=1, stderr=stderr)
    runner.script(["colab", "--auth", "adc", "stop"])
    with pytest.raises(ResourceUnavailableError):
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    assert [call[3] for call in runner.calls] == ["new", "stop"]


@pytest.mark.parametrize("gpu", [GpuType.T4, GpuType.A100])
def test_zero_gpu_count_sends_no_gpu_flag_before_premium_gate(
    tmp_path: Path,
    py_script: Path,
    gpu: GpuType,
) -> None:
    """model_construct bypasses JobSpec so the adapter guard is what runs."""
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner)
    provider.submit(
        JobSpec.model_construct(
            script_path=py_script,
            gpu_type=gpu,
            gpu_count=0,
            args=[],
            input_files=[],
            allow_premium=False,
            timeout_seconds=30,
        )
    )
    new = next(call for call in runner.calls if len(call) > 3 and call[3] == "new")
    assert new[0] == "colab"
    assert "-s" in new
    assert new[new.index("-s") + 1]
    assert "--gpu" not in new


def test_none_gpu_with_count_defaults_to_t4_and_zero_is_cpu(
    tmp_path: Path,
    py_script: Path,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner)
    provider.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=1))
    assert runner.calls[0][-2:] == ["--gpu", "T4"]

    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner)
    provider.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=0))
    assert "--gpu" not in runner.calls[0]


def test_keep_runtime_skips_stop_and_terminal_cancel_does_not_stop(
    tmp_path: Path,
    py_script: Path,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, keep_runtime=True)
    _script_flow(runner, ls="ok.txt\n")
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.L4))
    assert [call[3] for call in runner.calls] == ["new", "exec", "ls", "download"]
    assert "--gpu" in runner.calls[0]
    assert runner.calls[0][-1] == "L4"
    before = len(runner.calls)
    assert provider.cancel(handle.job_id).state is JobState.SUCCEEDED
    assert len(runner.calls) == before


def test_cancel_does_not_stop_a_terminal_job_that_already_stopped(
    tmp_path: Path,
    py_script: Path,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner, ls="")
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    before = len(runner.calls)
    assert provider.cancel(handle.job_id).state is JobState.SUCCEEDED
    assert len(runner.calls) == before


@pytest.mark.parametrize("stop_failure", ["returncode", "error"])
def test_stop_fails_twice_then_cancel_retries_stop(
    tmp_path: Path,
    py_script: Path,
    stop_failure: str,
) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    runner.script(["colab", "--auth", "adc", "new"])
    runner.script(["colab", "--auth", "adc", "exec"], stdout="hello\n")
    runner.script(["colab", "--auth", "adc", "ls"], stdout="")
    _queue_stop(runner, stop_failure)
    _queue_stop(runner, stop_failure)
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    session = _session(runner.calls)
    assert [call[3] for call in runner.calls].count("stop") == 2
    result = provider.status(handle.job_id)
    assert result.state is JobState.SUCCEEDED
    assert "runtime_stop_failed" in result.message
    assert "consuming CCU" in result.message
    assert f"colab stop {session}" in result.message
    assert result.details["runtime_stop_failed"] is True
    saved = _record(settings, handle.job_id)
    assert saved.state is JobState.SUCCEEDED
    assert saved.runtime_stopped is False
    assert saved.details["runtime_stop_failed"] is True
    if stop_failure == "returncode":
        assert saved.stop_returncode == 1
    else:
        assert saved.stop_returncode is None

    before = len(runner.calls)
    runner.script(["colab", "--auth", "adc", "stop"])
    cancelled = provider.cancel(handle.job_id)
    assert [call[3] for call in runner.calls[before:]] == ["stop"]
    assert runner.calls[-1] == ["colab", "--auth", "adc", "stop", "-s", session]
    assert cancelled.state is JobState.SUCCEEDED
    assert "runtime_stop_failed" not in cancelled.message
    assert cancelled.details.get("runtime_stop_failed") is not True
    finished = _record(settings, handle.job_id)
    assert finished.state is JobState.SUCCEEDED
    assert finished.runtime_stopped is True
    assert "runtime_stop_failed" not in finished.details


@pytest.mark.parametrize("stop_failure", ["returncode", "error"])
def test_stop_fails_once_then_succeeds_without_warning(
    tmp_path: Path,
    py_script: Path,
    stop_failure: str,
) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    runner.script(["colab", "--auth", "adc", "new"])
    runner.script(["colab", "--auth", "adc", "exec"], stdout="hello\n")
    runner.script(["colab", "--auth", "adc", "ls"], stdout="")
    _queue_stop(runner, stop_failure)
    runner.script(["colab", "--auth", "adc", "stop"])
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert [call[3] for call in runner.calls].count("stop") == 2
    result = provider.status(handle.job_id)
    assert result.state is JobState.SUCCEEDED
    assert "runtime_stop_failed" not in result.message
    assert result.details.get("runtime_stop_failed") is not True
    saved = _record(settings, handle.job_id)
    assert saved.runtime_stopped is True
    assert saved.stop_returncode == 0


def test_stop_failure_keeps_failed_execution_state(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["colab", "--auth", "adc", "new"])
    runner.script(["colab", "--auth", "adc", "exec"], returncode=1, stderr="script blew up\n")
    runner.script(["colab", "--auth", "adc", "ls"], stdout="")
    _queue_stop(runner, "returncode")
    _queue_stop(runner, "returncode")
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    result = provider.status(handle.job_id)
    assert result.state is JobState.FAILED
    assert result.exit_code == 1
    assert "script blew up" in result.message
    assert "runtime_stop_failed" in result.message
    assert result.details["runtime_stop_failed"] is True


def test_stop_retry_backoff_is_injectable_and_zero_by_default(
    tmp_path: Path,
    py_script: Path,
) -> None:
    def refuse(delay: float) -> None:
        raise AssertionError(f"unexpected sleep {delay}")

    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, sleeper=refuse)
    runner.script(["colab", "--auth", "adc", "new"])
    runner.script(["colab", "--auth", "adc", "exec"], stdout="hello\n")
    runner.script(["colab", "--auth", "adc", "ls"], stdout="")
    _queue_stop(runner, "returncode")
    runner.script(["colab", "--auth", "adc", "stop"])
    provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert [call[3] for call in runner.calls].count("stop") == 2

    sleeps: list[float] = []
    runner = RecordingRunner()
    provider, _settings = _provider(
        tmp_path,
        runner,
        stop_backoff_seconds=0.2,
        sleeper=sleeps.append,
    )
    runner.script(["colab", "--auth", "adc", "new"])
    runner.script(["colab", "--auth", "adc", "exec"], stdout="hello\n")
    runner.script(["colab", "--auth", "adc", "ls"], stdout="")
    _queue_stop(runner, "returncode")
    runner.script(["colab", "--auth", "adc", "stop"])
    provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert sleeps == [0.2]


def test_exec_timeout_fails_the_job_and_still_stops(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    _script_flow(
        runner,
        exec_error=ProviderCLIError("command timed out", code="cli_timeout", provider="colab"),
    )
    with pytest.raises(ProviderCLIError) as caught:
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    assert caught.value.code == "cli_timeout"
    assert [call[3] for call in runner.calls] == ["new", "exec", "stop"]
    record = _only_record(settings)
    assert record.state is JobState.FAILED
    assert record.runtime_stopped is True


def test_exec_timeout_with_keep_runtime_does_not_stop(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, keep_runtime=True)
    _script_flow(
        runner,
        exec_error=ProviderCLIError("command timed out", code="cli_timeout", provider="colab"),
    )
    with pytest.raises(ProviderCLIError) as caught:
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert caught.value.code == "cli_timeout"
    assert [call[3] for call in runner.calls] == ["new", "exec"]
    assert _only_record(settings).state is JobState.FAILED


def test_unset_exec_timeout_uses_configured_default(tmp_path: Path, py_script: Path) -> None:
    assert Settings(state_dir=tmp_path).providers.colab.default_exec_timeout_seconds == 21600
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner, ls="")
    provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    exec_call = next(item for item in runner.invocations if item["args"][3] == "exec")  # type: ignore[index]
    assert exec_call["timeout"] == 21600

    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, default_exec_timeout_seconds=90)
    _script_flow(runner, ls="")
    provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    exec_call = next(item for item in runner.invocations if item["args"][3] == "exec")  # type: ignore[index]
    assert exec_call["timeout"] == 90


def test_cancel_stops_a_live_session_without_allow_real_jobs(tmp_path: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, allow=False)
    job_id = "c" * 32
    session = "l2kc-cccccccccccc"
    record = ColabJobRecord(
        job_id=job_id,
        session=session,
        state=JobState.RUNNING,
        submitted_at=datetime.now(UTC),
        runtime_stopped=False,
    )
    path = settings.state_dir / "colab" / "jobs" / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(), encoding="utf-8")
    runner.script(["colab", "--auth", "adc", "stop"])
    result = provider.cancel(job_id)
    assert result.state is JobState.CANCELLED
    assert result.is_mock is False
    assert runner.calls == [["colab", "--auth", "adc", "stop", "-s", session]]


def test_exec_success_with_disconnect_substring_succeeds(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(
        runner,
        exec_rc=0,
        exec_stdout="disconnected from wandb\n",
        exec_stderr="",
        ls="",
    )
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    status = provider.status(handle.job_id)
    assert status.state is JobState.SUCCEEDED
    assert status.exit_code == 0
    assert status.message == ""

    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(
        runner,
        exec_rc=1,
        exec_stdout="disconnected from wandb\n",
        exec_stderr="script blew up\n",
        ls="",
    )
    failed = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    status = provider.status(failed.job_id)
    assert status.state is JobState.FAILED
    assert status.exit_code == 1
    assert "colab runtime disconnected" not in status.message
    assert "disconnected from wandb" in status.message


def test_exec_failure_disconnect_and_redaction(tmp_path: Path, py_script: Path) -> None:
    token = "KGAT_" + ("q" * 16)
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner, exec_rc=1, exec_stdout="", exec_stderr="script blew up\n")
    failed = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert provider.status(failed.job_id).state is JobState.FAILED
    assert provider.status(failed.job_id).exit_code == 1
    assert runner.calls[-1][3] == "stop"

    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner, exec_rc=1, exec_stderr="the runtime disconnected\n", ls="")
    dropped = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    status = provider.status(dropped.job_id)
    assert status.state is JobState.FAILED
    assert "disconnected" in status.message

    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    _script_flow(runner, exec_stdout=f"see {token}\n", ls="")
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    log_text = provider.logs(handle.job_id)
    assert token not in log_text
    assert "REDACTED" in log_text
    stored = Path(
        ColabJobRecord.model_validate_json(
            (settings.state_dir / "colab" / "jobs" / f"{handle.job_id}.json").read_text(
                encoding="utf-8"
            )
        ).log_path
    )
    assert token not in stored.read_text(encoding="utf-8")


def test_auth_usage_quota_sessions_and_detect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    home = tmp_path / "home"
    provider, _settings = _provider(tmp_path, runner, home=home, auth="adc")
    assert provider.auth_status().state is AuthState.unauthenticated
    assert runner.calls == []

    adc = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text("{}", encoding="utf-8")
    oauth_provider, _settings = _provider(tmp_path, runner, home=home, auth="oauth2")
    assert oauth_provider.auth_status().state is AuthState.unauthenticated
    assert runner.calls == []

    token = home / ".config" / "colab-cli" / "token.json"
    token.parent.mkdir(parents=True)
    token.write_text("{}", encoding="utf-8")
    runner.script(["colab", "--auth", "oauth2", "usage"], stdout=_USAGE)
    ok = oauth_provider.auth_status()
    assert ok.state is AuthState.authenticated
    assert ok.method is AuthMethod.oauth
    assert ok.is_mock is False
    assert ok.account_label is None
    assert runner.calls == [["colab", "--auth", "oauth2", "usage"]]

    runner.script(["colab", "--auth", "adc", "usage"], stdout=_USAGE)
    adc_status = provider.auth_status()
    assert adc_status.state is AuthState.authenticated
    assert adc_status.method is AuthMethod.adc

    runner.script(["colab", "--auth", "adc", "usage"], returncode=1, stderr="401 invalid token")
    assert provider.auth_status().state is AuthState.invalid
    runner.script(
        ["colab", "--auth", "adc", "usage"],
        returncode=1,
        stderr="invalid_grant: token expired; Reauthentication required",
    )
    assert provider.auth_status().state is AuthState.expired
    runner.script(["colab", "--auth", "adc", "usage"], returncode=2, stderr="connection reset")
    failed = provider.auth_status()
    assert failed.state is AuthState.unknown
    assert failed.state is not AuthState.authenticated

    quota_runner = RecordingRunner()
    quota_provider, _settings = _provider(tmp_path, quota_runner)
    quota_runner.script(["colab", "--auth", "adc", "usage"], stdout=_USAGE)
    quota = quota_provider.get_quota()
    assert quota.known is True
    assert quota.remaining == 12.5
    assert quota.total is None
    assert quota.unit is QuotaUnit.compute_units
    assert quota.source == "colab usage"
    assert quota.is_fixture is False
    assert quota.details["membership"] == "paid_pro"
    assert quota.details["usage_rate_per_hour"] == 1.8
    assert "informational" in quota.message
    quota_runner.script(["colab", "--auth", "adc", "usage"], stdout="not usage output")
    assert quota_provider.get_quota().known is False
    quota_runner.script(["colab", "--auth", "adc", "usage"], returncode=1, stdout=_USAGE)
    assert quota_provider.get_quota().known is False

    quota_runner.script(["colab", "--auth", "adc", "sessions"], stdout="alpha\n\nbeta\n")
    assert quota_provider.list_sessions() == ["alpha", "beta"]
    assert quota_runner.calls[-1] == ["colab", "--auth", "adc", "sessions"]

    resources = quota_provider.list_resources()
    by_id = {item.provider_resource_id: item for item in resources}
    assert by_id["T4"].gpu_memory_gb_per_gpu == 16
    assert by_id["T4"].premium is False
    assert by_id["L4"].gpu_memory_gb_per_gpu == 24
    assert by_id["A100"].gpu_memory_gb_per_gpu is None
    assert by_id["A100"].premium is True
    assert by_id["H100"].gpu_memory_gb_per_gpu == 80
    assert by_id["H100"].premium is True
    assert by_id["G4"].gpu_memory_gb_per_gpu is None
    assert by_id["G4"].premium is True
    assert by_id["cpu"].cpu_only is True
    assert all("colab new" in item.label for item in resources)

    monkeypatch.setattr(
        "link2kaggle_colab.providers.colab.cli_provider.shutil.which",
        lambda cmd: None,
    )
    assert quota_provider.detect_cli()["found"] is False
    monkeypatch.setattr(
        "link2kaggle_colab.providers.colab.cli_provider.shutil.which",
        lambda cmd: "/usr/bin/colab" if cmd == "colab" else None,
    )
    quota_runner.script(["colab", "version"], stdout="colab 0.7.5\n")
    assert quota_provider.detect_cli() == {
        "found": True,
        "path": "/usr/bin/colab",
        "version": "colab 0.7.5",
    }
    assert quota_runner.calls[-1] == ["colab", "version"]


def test_allow_premium_sends_the_gpu_flag(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    settings.scheduler.allow_premium_gpu = True
    _script_flow(runner, ls="")
    handle = provider.submit(
        JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=True)
    )
    assert handle.remote_id is not None
    assert runner.calls[0][-2:] == ["--gpu", "A100"]
    assert runner.calls[-1][3] == "stop"


def test_parse_usage_reads_eligible_gpus_when_present() -> None:
    plain = _parse_usage(_USAGE)
    assert plain.eligible_gpus is None
    with_line = _parse_usage(_USAGE + "Eligible GPUs: T4, L4, A100\n")
    assert with_line.eligible_gpus == ["T4", "L4", "A100"]
    assert with_line.remaining == 12.5

def test_submit_emits_tpu_flag(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    settings.scheduler.allow_premium_gpu = True
    _script_flow(runner, ls="")
    handle = provider.submit(
        JobSpec(script_path=py_script, gpu_type=GpuType.TPU_V5E1, allow_premium=True)
    )
    assert handle.remote_id is not None
    assert runner.calls[0] == [
        "colab",
        "--auth",
        "adc",
        "new",
        "-s",
        handle.remote_id,
        "--tpu",
        "v5e1",
    ]
    assert "--gpu" not in runner.calls[0]


def test_submit_emits_high_mem_with_a100(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    settings.scheduler.allow_premium_gpu = True
    _script_flow(runner, ls="")
    handle = provider.submit(
        JobSpec(
            script_path=py_script,
            gpu_type=GpuType.A100,
            allow_premium=True,
            high_mem=True,
        )
    )
    assert handle.remote_id is not None
    assert runner.calls[0] == [
        "colab",
        "--auth",
        "adc",
        "new",
        "-s",
        handle.remote_id,
        "--gpu",
        "A100",
        "--high-mem",
    ]


def test_submit_emits_high_mem_for_cpu(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    _script_flow(runner, ls="")
    handle = provider.submit(
        JobSpec(script_path=py_script, gpu_type=GpuType.NONE, gpu_count=0, high_mem=True)
    )
    assert handle.remote_id is not None
    assert runner.calls[0] == [
        "colab",
        "--auth",
        "adc",
        "new",
        "-s",
        handle.remote_id,
        "--high-mem",
    ]


def test_high_mem_does_not_bypass_premium_gate(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    with pytest.raises(PremiumNotAllowed):
        provider.submit(
            JobSpec(
                script_path=py_script,
                gpu_type=GpuType.A100,
                allow_premium=False,
                high_mem=True,
            )
        )
    assert runner.calls == []


def test_list_resources_includes_tpu_and_high_mem_flags(tmp_path: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    by_id = {item.provider_resource_id: item for item in provider.list_resources()}
    assert by_id["TPU_V5E1"].gpu_type is GpuType.TPU_V5E1
    assert by_id["TPU_V5E1"].premium is True
    assert by_id["TPU_V5E1"].supports_high_mem is False
    assert by_id["TPU_V6E1"].supports_high_mem is False
    assert by_id["T4"].supports_high_mem is True
    assert by_id["A100"].supports_high_mem is True
    assert by_id["L4"].supports_high_mem is False
    assert by_id["cpu"].supports_high_mem is True



def test_get_quota_writes_identity_without_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    settings = Settings(
        mode="real",
        allow_real_jobs=True,
        state_dir=tmp_path / "state",
        providers={"colab": {"auth": "adc"}},
    )
    home = tmp_path / "home"
    token_dir = home / ".config" / "colab-cli"
    token_dir.mkdir(parents=True)
    token_dir.joinpath("token.json").write_text(
        json.dumps(
            {
                "account": "tluc@example.com",
                "token": "ya29.live-secret",
                "refresh_token": "1//secret",
                "client_secret": "shh",
            }
        ),
        encoding="utf-8",
    )
    provider = ColabCliProvider(
        settings,
        runner=runner,
        home=home,
        environ={},
        account_id="colab-tluc",
        state_dir=tmp_path / "state" / "accounts" / "colab-tluc",
    )
    # Default path is ``colab usage``; experimental ccu-info must not be called.
    import link2kaggle_colab.providers.colab.cli_provider as mod

    calls: list[object] = []

    def _boom(_home):
        calls.append(_home)
        raise AssertionError("ccu-info must not be fetched by default")

    monkeypatch.setattr(mod, "ccu_info_snapshot", _boom)
    runner.script(
        ["colab", "--auth", "adc", "usage"],
        stdout="Current balance: 0 compute units\nEligible GPUs: T4\n",
    )
    quota = provider.get_quota()
    assert calls == []
    assert quota.source == "colab usage"
    assert quota.remaining == 0.0
    assert quota.eligible_gpus == ["T4"]
    assert quota.details["membership"] == "free_pro"
    identity = Path(tmp_path / "state" / "accounts" / "colab-tluc" / "identity.json")
    assert identity.is_file()
    raw = identity.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["email"] == "tluc@example.com"
    assert data["membership_tier"] == "free_pro"
    assert data["eligible_gpus"] == ["T4"]
    assert data["ccu_remaining"] == 0.0
    assert "ya29" not in raw
    assert "secret" not in raw


def test_get_quota_default_does_not_call_ccu_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    import link2kaggle_colab.providers.colab.cli_provider as mod

    def _boom(_home):
        raise AssertionError("default quota path must not call ccu_info_snapshot")

    monkeypatch.setattr(mod, "ccu_info_snapshot", _boom)
    runner.script(["colab", "--auth", "adc", "usage"], stdout=_USAGE)
    quota = provider.get_quota()
    assert quota.source == "colab usage"
    assert quota.remaining == 12.5
    assert runner.calls == [["colab", "--auth", "adc", "usage"]]


def test_get_quota_experimental_ccu_info_via_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    settings = Settings(
        mode="real",
        allow_real_jobs=True,
        state_dir=tmp_path / "state",
        providers={"colab": {"auth": "adc", "experimental_ccu_info": True}},
    )
    provider = ColabCliProvider(
        settings,
        runner=runner,
        home=tmp_path / "home",
        environ={},
    )
    import link2kaggle_colab.providers.colab.cli_provider as mod

    def _snap(_home):
        return {
            "balance": 42.0,
            "eligible_gpus": ["T4", "L4"],
            "email": "user@example.com",
            "source": "ccu-info",
            "raw_body": "",
        }

    monkeypatch.setattr(mod, "ccu_info_snapshot", _snap)
    quota = provider.get_quota()
    assert quota.source == "ccu-info"
    assert quota.remaining == 42.0
    assert quota.eligible_gpus == ["T4", "L4"]
    assert runner.calls == []


def test_get_quota_experimental_ccu_info_via_env_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    monkeypatch.setenv("L2KC_COLAB_EXPERIMENTAL_CCU_INFO", "1")
    import link2kaggle_colab.providers.colab.cli_provider as mod

    def _boom(_home):
        raise ValueError("simulated ccu-info HTTP failure")

    monkeypatch.setattr(mod, "ccu_info_snapshot", _boom)
    runner.script(
        ["colab", "--auth", "adc", "usage"],
        stdout=_USAGE + "Eligible GPUs: T4\n",
    )
    quota = provider.get_quota()
    assert quota.source == "colab usage"
    assert quota.remaining == 12.5
    assert quota.eligible_gpus == ["T4"]
    assert runner.calls == [["colab", "--auth", "adc", "usage"]]
