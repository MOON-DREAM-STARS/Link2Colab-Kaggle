"""Kaggle CLI adapter tests. The real ``kaggle`` binary is never executed."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import (
    ConfigError,
    InvalidInputError,
    NotSupportedError,
    PremiumNotAllowed,
    ProviderCLIError,
    QuotaExceededError,
    RealModeDisabledError,
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
from link2kaggle_colab.providers.kaggle.cli_provider import KaggleCliProvider, KaggleJobRecord
from tests.mocks.runner import RecordingRunner

_JOB = "a" * 32
_KERNEL = "tester/my-kernel"
_GPU_LIST = json.dumps(
    [
        {"resource": "CPU", "used": "1h", "remaining": "2h", "total": "3h"},
        {
            "resource": "GPU",
            "used": "12.50h",
            "remaining": "17.50h",
            "total": "30h",
            "refreshAt": "2026-11-01T00:00:00Z",
        },
    ]
)


_PUSH_OK = "Kernel version 1 successfully pushed.\n"


def _provider(
    tmp_path: Path,
    runner: RecordingRunner,
    *,
    mode: str = "real",
    allow: bool = True,
    username: str | None = "tester",
    push_timeout: int | None = None,
    download_timeout: int | None = None,
) -> tuple[KaggleCliProvider, Settings]:
    kaggle: dict[str, object] = {"username": username}
    if push_timeout is not None:
        kaggle["default_push_timeout_seconds"] = push_timeout
    if download_timeout is not None:
        kaggle["download_timeout_seconds"] = download_timeout
    settings = Settings(
        mode=mode,  # type: ignore[arg-type]
        allow_real_jobs=allow,
        state_dir=tmp_path / "state",
        providers={"kaggle": kaggle},
    )
    provider = KaggleCliProvider(
        settings,
        runner=runner,
        home=tmp_path / "home",
        environ={},
    )
    return provider, settings


def _plant_token(home: Path) -> None:
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    (kaggle_dir / "access_token").write_text("present\n", encoding="utf-8")


def _job_records(state_dir: Path) -> list[KaggleJobRecord]:
    jobs = state_dir / "kaggle" / "jobs"
    if not jobs.is_dir():
        return []
    return [
        KaggleJobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(jobs.glob("*.json"))
    ]


def _write_job(state_dir: Path, job_id: str = _JOB, kernel_ref: str = _KERNEL) -> None:
    record = KaggleJobRecord(
        job_id=job_id,
        kernel_ref=kernel_ref,
        state=JobState.QUEUED,
        submitted_at=datetime.now(UTC),
    )
    path = state_dir / "kaggle" / "jobs" / f"{job_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(), encoding="utf-8")


def test_real_mode_disabled_never_calls_the_runner(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, mode="mock", allow=True)
    spec = JobSpec(script_path=py_script, name="my-kernel", gpu_type=GpuType.T4)
    calls = (
        provider.detect_cli,
        provider.auth_status,
        provider.get_quota,
        provider.list_resources,
        lambda: provider.submit(spec),
        lambda: provider.status(_JOB),
        lambda: provider.logs(_JOB),
        lambda: provider.download(_JOB, tmp_path / "out"),
        lambda: provider.cancel(_JOB),
    )
    for call in calls:
        with pytest.raises(RealModeDisabledError):
            call()
    assert runner.calls == []
    caps = provider.capabilities()
    assert caps.supports_cancel is False
    assert caps.is_batch_only is True
    assert caps.supports_interactive_session is False
    assert runner.calls == []


def test_submit_blocked_when_real_jobs_are_off_but_quota_is_allowed(
    tmp_path: Path,
    py_script: Path,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner, allow=False)
    with pytest.raises(RealModeDisabledError):
        provider.submit(JobSpec(script_path=py_script, name="my-kernel", gpu_type=GpuType.T4))
    assert runner.calls == []
    runner.script(["kaggle", "quota", "--format", "json"], stdout=_GPU_LIST)
    quota = provider.get_quota()
    assert quota.known is True
    assert quota.unit is QuotaUnit.gpu_hours
    assert runner.calls == [["kaggle", "quota", "--format", "json"]]


def test_submit_writes_private_metadata_and_exact_argv(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK)
    handle = provider.submit(
        JobSpec(
            script_path=py_script,
            name="my-kernel",
            gpu_type=GpuType.T4,
            timeout_seconds=30,
        )
    )
    assert handle.remote_id == "tester/l2kc-my-kernel"
    # The -t run limit does not extend the push wait; it stays bounded.
    assert runner.invocations[0]["timeout"] == 300
    assert handle.provider is ProviderName.kaggle
    assert runner.calls == [
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            runner.calls[0][4],
            "--accelerator",
            "NvidiaTeslaT4",
            "-t",
            "30",
        ]
    ]
    staging = Path(runner.calls[0][4])
    assert staging.is_absolute()
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta == {
        "id": "tester/l2kc-my-kernel",
        "title": "l2kc-my-kernel",
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    assert (staging / "script.py").read_text(encoding="utf-8") == py_script.read_text(
        encoding="utf-8"
    )
    saved = json.loads(
        (settings.state_dir / "kaggle" / "jobs" / f"{handle.job_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["state"] == "QUEUED"
    assert saved["kernel_ref"] == "tester/l2kc-my-kernel"
    assert saved["job_id"] == handle.job_id
    assert not any(part == "delete" for call in runner.calls for part in call)


def test_generated_slug_cpu_and_notebook(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK, permanent=True)
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert handle.remote_id is not None
    assert handle.remote_id.startswith("tester/l2kc-")
    staging = Path(runner.calls[0][4])
    cpu_meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert cpu_meta["enable_gpu"] is False
    assert cpu_meta["enable_internet"] is False
    assert cpu_meta["is_private"] is True
    assert "machine_shape" not in cpu_meta
    assert runner.calls[0][5:] == []

    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}\n", encoding="utf-8")
    provider.submit(JobSpec(script_path=notebook, name="nb-kernel", gpu_type=GpuType.NONE))
    nb_meta = json.loads(
        (Path(runner.calls[1][4]) / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert nb_meta["kernel_type"] == "notebook"
    assert nb_meta["code_file"] == "notebook.ipynb"

    provider.submit(JobSpec(script_path=py_script, name="l4-kernel", gpu_type=GpuType.L4))
    assert runner.calls[2][5:] == ["--accelerator", "NvidiaL4X1"]
    provider.submit(
        JobSpec(script_path=py_script, name="l4-pair", gpu_type=GpuType.L4, gpu_count=2)
    )
    assert runner.calls[3][5:] == ["--accelerator", "NvidiaL4"]


def test_none_gpu_with_count_defaults_to_t4_and_zero_is_cpu(
    tmp_path: Path,
    py_script: Path,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK, permanent=True)
    provider.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=1, name="any-gpu"))
    assert runner.calls[0][5:] == ["--accelerator", "NvidiaTeslaT4"]
    staging = Path(runner.calls[0][4])
    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_gpu"] is True

    provider.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=0, name="cpu-job"))
    assert "--accelerator" not in runner.calls[1]
    cpu_staging = Path(runner.calls[1][4])
    cpu = json.loads((cpu_staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert cpu["enable_gpu"] is False
    assert "machine_shape" not in cpu


@pytest.mark.parametrize("gpu", [GpuType.A100, GpuType.H100])
def test_premium_without_allow_makes_no_push(
    tmp_path: Path,
    py_script: Path,
    gpu: GpuType,
) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    with pytest.raises(PremiumNotAllowed) as caught:
        provider.submit(JobSpec(script_path=py_script, name="premium-kernel", gpu_type=gpu))
    assert caught.value.code == "premium_not_allowed"
    assert caught.value.provider == ProviderName.kaggle.value
    assert runner.calls == []
    assert _job_records(settings.state_dir) == []

    # Per-job confirmation does not override scheduler.allow_premium_gpu.
    with pytest.raises(PremiumNotAllowed):
        provider.submit(
            JobSpec(
                script_path=py_script,
                name="premium-kernel",
                gpu_type=gpu,
                allow_premium=True,
            )
        )
    assert runner.calls == []


@pytest.mark.parametrize(
    ("gpu", "shape"),
    [
        (GpuType.A100, "NvidiaTeslaA100"),
        (GpuType.H100, "NvidiaH100"),
    ],
)
def test_premium_gate_allows_push(
    tmp_path: Path,
    py_script: Path,
    gpu: GpuType,
    shape: str,
) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    settings.scheduler.allow_premium_gpu = True
    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK)
    handle = provider.submit(
        JobSpec(
            script_path=py_script,
            name="premium-kernel",
            gpu_type=gpu,
            allow_premium=True,
        )
    )
    assert handle.remote_id == "tester/l2kc-premium-kernel"
    assert runner.calls == [
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            runner.calls[0][4],
            "--accelerator",
            shape,
        ]
    ]


def test_submit_rejects_args_inputs_and_bad_username(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    with pytest.raises(NotSupportedError, match="CLI args"):
        provider.submit(JobSpec(script_path=py_script, args=["--epochs", "1"]))
    data = tmp_path / "data.csv"
    data.write_text("a\n", encoding="utf-8")
    with pytest.raises(NotSupportedError, match="dataset_sources"):
        provider.submit(JobSpec(script_path=py_script, input_files=[data]))
    missing, _settings = _provider(tmp_path, runner, username=None)
    with pytest.raises(ConfigError):
        missing.submit(JobSpec(script_path=py_script))
    bad, _settings = _provider(tmp_path, runner, username="AB")
    with pytest.raises(ConfigError):
        bad.submit(JobSpec(script_path=py_script))
    with pytest.raises(NotSupportedError):
        provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.P100, name="my-kernel"))
    assert runner.calls == []


def test_push_failure_redacts_tokens_and_quota_is_distinct(tmp_path: Path, py_script: Path) -> None:
    token = "KGAT_" + ("c" * 16)
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    spec = JobSpec(script_path=py_script, name="my-kernel", gpu_type=GpuType.T4)
    runner.script(["kaggle", "kernels", "push"], returncode=1, stderr=f"failed {token}")
    with pytest.raises(ProviderCLIError) as failed:
        provider.submit(spec)
    blob = json.dumps(failed.value.to_dict())
    assert token not in blob
    assert token not in str(failed.value)
    assert "REDACTED" in blob
    records = _job_records(settings.state_dir)
    assert len(records) == 1
    assert records[0].state is JobState.FAILED
    assert token not in records[0].model_dump_json()

    runner.script(["kaggle", "kernels", "push"], returncode=0, stdout="error: still bad\n")
    with pytest.raises(ProviderCLIError):
        provider.submit(spec)
    records = _job_records(settings.state_dir)
    assert len(records) == 2
    assert all(item.state is JobState.FAILED for item in records)

    runner.script(["kaggle", "kernels", "push"], returncode=1, stderr=f"quota exceeded {token}")
    with pytest.raises(QuotaExceededError) as quota:
        provider.submit(spec)
    assert token not in str(quota.value)
    records = _job_records(settings.state_dir)
    assert len(records) == 3
    assert all(item.state is JobState.FAILED for item in records)
    assert all(token not in item.model_dump_json() for item in records)


def test_auth_verifies_with_quota_and_never_invents_success(tmp_path: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    missing = provider.auth_status()
    assert missing.state is AuthState.unauthenticated
    assert missing.is_mock is False
    assert runner.calls == []

    _plant_token(tmp_path / "home")
    runner.script(["kaggle", "quota", "--format", "json"], stdout=_GPU_LIST)
    ok = provider.auth_status()
    assert ok.state is AuthState.authenticated
    assert ok.method is AuthMethod.api_token
    assert ok.is_mock is False
    assert ok.account_label == "tester"
    assert runner.calls == [["kaggle", "quota", "--format", "json"]]

    runner.script(
        ["kaggle", "quota", "--format", "json"],
        returncode=1,
        stderr="401 Client Error: Unauthenticated",
    )
    invalid = provider.auth_status()
    assert invalid.state is AuthState.invalid
    assert invalid.state is not AuthState.authenticated

    runner.script(
        ["kaggle", "quota", "--format", "json"],
        returncode=1,
        stderr="403 permission denied",
    )
    denied = provider.auth_status()
    assert denied.state is AuthState.error
    assert denied.state is not AuthState.authenticated
    assert denied.details["reason"] == "permission_denied"

    runner.script(["kaggle", "quota", "--format", "json"], returncode=2, stderr="connection reset")
    unknown = provider.auth_status()
    assert unknown.state is AuthState.unknown
    assert unknown.state is not AuthState.authenticated


@pytest.mark.parametrize(
    ("stdout", "returncode", "used", "remaining", "total"),
    [
        (_GPU_LIST, 0, 12.5, 17.5, 30.0),
        (json.dumps({"resource": "GPU", "used": 1, "remaining": 2, "total": 3}), 0, 1.0, 2.0, 3.0),
        (
            json.dumps(
                {
                    "quotas": [
                        {"resource": "GPU", "used": "1.5h", "remaining": "2.5h", "total": "4h"}
                    ]
                }
            ),
            0,
            1.5,
            2.5,
            4.0,
        ),
    ],
)
def test_quota_json_variants(
    tmp_path: Path,
    stdout: str,
    returncode: int,
    used: float,
    remaining: float,
    total: float,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "quota", "--format", "json"], returncode=returncode, stdout=stdout)
    quota = provider.get_quota()
    assert quota.known is True
    assert quota.is_fixture is False
    assert quota.unit is QuotaUnit.gpu_hours
    assert quota.source == "kaggle quota"
    assert quota.used == used
    assert quota.remaining == remaining
    assert quota.total == total
    if "refreshAt" in stdout:
        assert quota.refresh_at is not None
        assert quota.refresh_at.year == 2026


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        ("not json", 0),
        (json.dumps([{"resource": "TPU", "used": "1h", "remaining": "1h", "total": "2h"}]), 0),
        (_GPU_LIST, 1),
        ("", 0),
    ],
)
def test_quota_garbage_is_unknown(tmp_path: Path, stdout: str, returncode: int) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "quota", "--format", "json"], returncode=returncode, stdout=stdout)
    quota = provider.get_quota()
    assert quota.known is False
    assert quota.is_fixture is False
    assert quota.unit is QuotaUnit.unknown


def test_list_resources_marks_unverified_shapes(tmp_path: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    resources = provider.list_resources()
    assert [item.provider_resource_id for item in resources] == [
        "NvidiaTeslaT4",
        "NvidiaL4X1",
        "NvidiaL4",
        "cpu",
    ]
    t4 = resources[0]
    assert t4.gpu_type is GpuType.T4
    assert t4.gpu_count == 2
    assert t4.gpu_memory_gb_per_gpu == 16
    unverified = next(item for item in resources if item.provider_resource_id == "NvidiaL4")
    assert unverified.label.startswith("L4 (count unverified)")
    assert unverified.gpu_count == 1
    assert all("eligibility needs real verification" in item.label for item in resources)
    assert all(
        "unverified" in item.label
        for item in resources
        if item.gpu_type not in {GpuType.T4, GpuType.NONE}
    )
    assert runner.calls == []


@pytest.mark.parametrize(
    ("stdout", "state"),
    [
        ('kernel has status "QUEUED"\n', JobState.QUEUED),
        ('kernel has status "NEW_SCRIPT"\n', JobState.QUEUED),
        ('kernel has status "KernelWorkerStatus.NEW_SCRIPT"\n', JobState.QUEUED),
        ('kernel has status "RUNNING"\n', JobState.RUNNING),
        ('kernel has status "running"\n', JobState.RUNNING),
        ('kernel has status "COMPLETE"\n', JobState.SUCCEEDED),
        ('kernel has status "complete"\n', JobState.SUCCEEDED),
        ('kernel has status "KernelWorkerStatus.COMPLETE"\n', JobState.SUCCEEDED),
        ('kernel has status "ERROR"\n', JobState.FAILED),
        ('kernel has status "KernelWorkerStatus.ERROR"\n', JobState.FAILED),
        ('kernel has status "CANCEL_REQUESTED"\n', JobState.CANCELLING),
        ('kernel has status "CANCEL_ACKNOWLEDGED"\n', JobState.CANCELLED),
        ('kernel has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"\n', JobState.CANCELLED),
        ("this is garbage\n", JobState.UNKNOWN),
        ('kernel has status "POTATO"\n', JobState.UNKNOWN),
        ('kernel has status "KernelWorkerStatus.POTATO"\n', JobState.UNKNOWN),
    ],
)
def test_status_parsing(tmp_path: Path, stdout: str, state: JobState) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, allow=False)
    _write_job(settings.state_dir)
    runner.script(["kaggle", "kernels", "status"], stdout=stdout, returncode=0)
    result = provider.status(_JOB)
    assert result.state is state
    assert result.is_mock is False
    assert result.remote_id == _KERNEL
    assert runner.calls == [["kaggle", "kernels", "status", _KERNEL]]
    saved = json.loads(
        (settings.state_dir / "kaggle" / "jobs" / f"{_JOB}.json").read_text(encoding="utf-8")
    )
    assert saved["state"] == state.value


def test_status_failure_message_is_redacted_and_garbage_exit_is_unknown(tmp_path: Path) -> None:
    token = "KGAT_" + ("e" * 12)
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, allow=False)
    _write_job(settings.state_dir)
    runner.script(
        ["kaggle", "kernels", "status"],
        stdout=f'kernel has status "ERROR"\nFailure message: "boom {token}"\n',
    )
    failed = provider.status(_JOB)
    assert failed.state is JobState.FAILED
    assert "boom" in failed.message
    assert token not in failed.message
    assert "REDACTED" in failed.message

    _write_job(settings.state_dir, job_id="b" * 32)
    runner.script(["kaggle", "kernels", "status"], returncode=2, stdout="nope")
    unknown = provider.status("b" * 32)
    assert unknown.state is JobState.UNKNOWN

    _write_job(settings.state_dir, job_id="c" * 32, kernel_ref="-rf")
    with pytest.raises(InvalidInputError):
        provider.status("c" * 32)
    assert all(part != "-rf" for call in runner.calls for part in call)


def test_logs_download_and_cancel(tmp_path: Path) -> None:
    token = "KGAT_" + ("f" * 12)
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, allow=False)
    _write_job(settings.state_dir)
    runner.script(["kaggle", "kernels", "logs"], stdout=f"line {token}\n")
    text = provider.logs(_JOB)
    assert token not in text
    assert "REDACTED" in text
    assert runner.calls == [["kaggle", "kernels", "logs", _KERNEL]]

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    def plant(args: list[str]) -> None:
        dest = Path(args[args.index("-p") + 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "good.txt").write_text("hello", encoding="utf-8")
        (dest / "escape").symlink_to(outside)

    runner.script(["kaggle", "kernels", "output"], on_call=plant)
    dest = tmp_path / "downloaded"
    artifacts = provider.download(_JOB, dest)
    assert [item.name for item in artifacts] == ["good.txt"]
    assert artifacts[0].local_path.is_relative_to(dest.resolve())
    assert artifacts[0].size_bytes == 5
    escape = dest / "escape"
    assert escape.is_symlink()
    assert "secret" not in artifacts[0].local_path.read_text(encoding="utf-8")
    output_call = runner.calls[-1]
    assert output_call[:4] == ["kaggle", "kernels", "output", _KERNEL]
    assert output_call[4:6] == ["-p", str(dest.resolve())]
    assert runner.invocations[-1]["timeout"] == 600

    with pytest.raises(NotSupportedError, match="delete") as caught:
        provider.cancel(_JOB)
    assert "cancel_kernel_session" in caught.value.message
    assert not any(part == "delete" for call in runner.calls for part in call)


def test_source_never_passes_delete_to_the_cli() -> None:
    source = Path("src/link2kaggle_colab/providers/kaggle/cli_provider.py").read_text(
        encoding="utf-8"
    )
    assert '"delete"' not in source
    assert "'delete'" not in source


def test_detect_cli_uses_which_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)

    monkeypatch.setattr(
        "link2kaggle_colab.providers.kaggle.cli_provider.shutil.which",
        lambda cmd: None,
    )
    assert provider.detect_cli() == {"found": False, "path": None, "version": None}
    assert runner.calls == []

    monkeypatch.setattr(
        "link2kaggle_colab.providers.kaggle.cli_provider.shutil.which",
        lambda cmd: "/usr/bin/kaggle" if cmd == "kaggle" else None,
    )
    runner.script(["kaggle", "--version"], stdout="Kaggle CLI 2.2.5\n")
    assert provider.detect_cli() == {
        "found": True,
        "path": "/usr/bin/kaggle",
        "version": "Kaggle CLI 2.2.5",
    }
    runner.script(["kaggle", "--version"], returncode=1, stderr="nope")
    failed = provider.detect_cli()
    assert failed["found"] is True
    assert failed["version"] is None


def test_push_record_is_submitting_until_the_cli_returns(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    seen: dict[str, str] = {}

    def capture(_args: list[str]) -> None:
        records = _job_records(settings.state_dir)
        assert len(records) == 1
        seen["state"] = records[0].state.value
        seen["kernel_ref"] = records[0].kernel_ref

    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK, on_call=capture)
    handle = provider.submit(JobSpec(script_path=py_script, name="mynb", gpu_type=GpuType.NONE))
    assert seen == {"state": "SUBMITTING", "kernel_ref": "tester/l2kc-mynb"}
    assert handle.remote_id == "tester/l2kc-mynb"
    saved = _job_records(settings.state_dir)
    assert len(saved) == 1
    assert saved[0].state is JobState.QUEUED
    assert saved[0].job_id == handle.job_id


def test_push_timeout_records_unknown_and_returns_a_handle(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    runner.script(
        ["kaggle", "kernels", "push"],
        error=ProviderCLIError("command timed out", code="cli_timeout"),
    )
    handle = provider.submit(JobSpec(script_path=py_script, name="mynb", gpu_type=GpuType.NONE))
    assert runner.invocations[0]["timeout"] == 300
    assert handle.remote_id == "tester/l2kc-mynb"
    saved = _job_records(settings.state_dir)
    assert len(saved) == 1
    assert saved[0].state is JobState.UNKNOWN
    assert saved[0].finished_at is None
    assert saved[0].job_id == handle.job_id
    assert saved[0].kernel_ref == handle.remote_id
    assert "may be running" in saved[0].message
    assert saved[0].kernel_ref in saved[0].message


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        ("Kernel version 1 successfully pushed.\nKernel push error: lost the status\n", 0),
        ("Kernel version 1 successfully pushed.\n", 1),
        ("waiting for the server\n", 0),
    ],
)
def test_ambiguous_push_output_records_unknown_and_returns_a_handle(
    tmp_path: Path,
    py_script: Path,
    stdout: str,
    returncode: int,
) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "kernels", "push"], stdout=stdout, returncode=returncode)
    handle = provider.submit(JobSpec(script_path=py_script, name="mynb", gpu_type=GpuType.NONE))
    saved = _job_records(settings.state_dir)
    assert len(saved) == 1
    assert saved[0].state is JobState.UNKNOWN
    assert saved[0].job_id == handle.job_id
    assert handle.remote_id == "tester/l2kc-mynb"
    assert saved[0].kernel_ref == handle.remote_id
    assert "may be running" in saved[0].message
    assert saved[0].kernel_ref in saved[0].message
    assert saved[0].state is not JobState.SUBMITTING


def test_default_push_timeout_is_bounded(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner)
    assert settings.providers.kaggle.default_push_timeout_seconds == 300
    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK)
    provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.NONE))
    assert runner.invocations[0]["timeout"] == 300
    assert "-t" not in runner.calls[0]

    custom_runner = RecordingRunner()
    custom, _settings = _provider(tmp_path, custom_runner, push_timeout=45)
    custom_runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK)
    custom.submit(JobSpec(script_path=py_script, name="other", gpu_type=GpuType.NONE))
    assert custom_runner.invocations[0]["timeout"] == 45


def test_user_slug_gets_l2kc_prefix_unless_present(tmp_path: Path, py_script: Path) -> None:
    runner = RecordingRunner()
    provider, _settings = _provider(tmp_path, runner)
    runner.script(["kaggle", "kernels", "push"], stdout=_PUSH_OK, permanent=True)
    plain = provider.submit(JobSpec(script_path=py_script, name="mynb", gpu_type=GpuType.NONE))
    assert plain.remote_id == "tester/l2kc-mynb"
    plain_meta = json.loads(
        (Path(runner.calls[0][4]) / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert plain_meta["id"] == "tester/l2kc-mynb"
    assert plain_meta["title"] == "l2kc-mynb"

    prefixed = provider.submit(
        JobSpec(script_path=py_script, name="l2kc-x", gpu_type=GpuType.NONE)
    )
    assert prefixed.remote_id == "tester/l2kc-x"
    prefixed_meta = json.loads(
        (Path(runner.calls[1][4]) / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert prefixed_meta["id"] == "tester/l2kc-x"
    assert prefixed_meta["title"] == "l2kc-x"

    with pytest.raises(InvalidInputError):
        provider.submit(JobSpec(script_path=py_script, name="a" * 46, gpu_type=GpuType.NONE))
    assert len(runner.calls) == 2


def test_repeated_status_keeps_finished_at(tmp_path: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, allow=False)
    finished = datetime(2024, 5, 6, 7, 8, 9, tzinfo=UTC)
    record = KaggleJobRecord(
        job_id=_JOB,
        kernel_ref=_KERNEL,
        state=JobState.SUCCEEDED,
        submitted_at=datetime(2024, 5, 6, 7, 0, tzinfo=UTC),
        finished_at=finished,
        message="done",
    )
    path = settings.state_dir / "kaggle" / "jobs" / f"{_JOB}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(), encoding="utf-8")
    runner.script(
        ["kaggle", "kernels", "status"],
        stdout='kernel has status "COMPLETE"\n',
        permanent=True,
    )
    first = provider.status(_JOB)
    second = provider.status(_JOB)
    assert first.finished_at == finished
    assert second.finished_at == finished
    assert first.state is JobState.SUCCEEDED
    assert second.state is JobState.SUCCEEDED
    stored = KaggleJobRecord.model_validate_json(path.read_text(encoding="utf-8"))
    assert stored.finished_at == finished


def test_configured_download_timeout_is_passed(tmp_path: Path) -> None:
    runner = RecordingRunner()
    provider, settings = _provider(tmp_path, runner, allow=False, download_timeout=45)
    assert settings.providers.kaggle.download_timeout_seconds == 45
    _write_job(settings.state_dir)
    runner.script(["kaggle", "kernels", "output"])
    provider.download(_JOB, tmp_path / "out")
    assert runner.invocations[-1]["timeout"] == 45
