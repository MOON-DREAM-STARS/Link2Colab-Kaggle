"""Client behaviour in mock mode. Quotas here are test fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import (
    KaggleProviderSettings,
    MockSettings,
    ProvidersSettings,
    SchedulerSettings,
    Settings,
)
from link2kaggle_colab.core.errors import (
    ConfigError,
    InvalidInputError,
    JobNotFoundError,
    L2KCError,
    NotSupportedError,
    PremiumNotAllowed,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import (
    GpuType,
    JobHandle,
    JobSpec,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.fake.provider import FakeProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.storage.sqlite_store import SqliteJobStore, default_db_path
from tests.mocks.runner import RecordingRunner


def _settings(tmp_path: Path, **mock: str) -> Settings:
    return Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        artifact_root=tmp_path,
        mock=MockSettings(
            kaggle_scenario=mock.get("kaggle_scenario", "ok"),
            colab_scenario=mock.get("colab_scenario", "ok"),
        ),
    )


def _script(tmp_path: Path, body: str | None = None) -> Path:
    path = tmp_path / "job.py"
    path.write_text(body or "print('hello-from-script')\n", encoding="utf-8")
    return path


def _fake_script(tmp_path: Path) -> Path:
    return _script(
        tmp_path,
        "from pathlib import Path\n"
        "out = Path('outputs')\n"
        "out.mkdir()\n"
        "(out / 'result.txt').write_text('from-fake\\n', encoding='utf-8')\n"
        "print('fake-ran')\n",
    )


def _poll_until_succeeded(client: Client, job_id: str) -> list[str]:
    seen: list[str] = []
    for _ in range(6):
        state = client.status(job_id).state.value
        seen.append(state)
        if state == "SUCCEEDED":
            return seen
    raise AssertionError(seen)


def test_list_providers_and_doctor_stay_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("doctor or list_providers executed a process")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr("link2kaggle_colab.core.process.subprocess.run", explode)
    client = Client(_settings(tmp_path))
    rows = {row["name"]: row for row in client.list_providers()}
    assert set(rows) == {"kaggle", "colab", "fake"}
    assert rows["kaggle"]["enabled"] is True
    assert rows["kaggle"]["is_mock"] is True
    assert rows["kaggle"]["mode"] == "mock"
    assert rows["kaggle"]["capabilities"]["supports_cancel"] is False
    assert rows["colab"]["capabilities"]["supports_cancel"] is True
    assert rows["fake"]["is_mock"] is True

    report = client.doctor()
    assert report["mode"] == "mock"
    assert report["allow_real_jobs"] is False
    assert report["state_dir_writable"] is True
    assert report["python_version"] == sys.version.split()[0]
    assert "TEST FIXTURE" in report["fixture_notice"]
    by_name = {row["name"]: row for row in report["providers"]}
    assert by_name["kaggle"]["cli"]["executed"] is False
    assert by_name["colab"]["cli"]["executed"] is False
    assert by_name["fake"]["cli"]["cli_path"] is None
    assert by_name["kaggle"]["auth"]["is_mock"] is True
    assert by_name["kaggle"]["auth"]["account_id"] == "kaggle-default"
    assert by_name["colab"]["auth"]["account_id"] == "colab-default"
    assert by_name["fake"]["auth"]["account_id"] == "fake-default"
    assert "TEST FIXTURE" in by_name["kaggle"]["auth"]["message"]


def test_real_mode_doctor_does_not_execute_clis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real-mode doctor executed a process")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr("link2kaggle_colab.core.process.subprocess.run", explode)
    client = Client(
        Settings(
            mode="real",
            allow_real_jobs=False,
            state_dir=tmp_path / "state",
            artifact_root=tmp_path,
        )
    )
    report = client.doctor()
    assert report["mode"] == "real"
    assert report["fixture_notice"] == ""
    by_name = {row["name"]: row for row in report["providers"]}
    assert by_name["kaggle"]["cli"]["executed"] is False
    assert by_name["colab"]["cli"]["executed"] is False
    assert by_name["kaggle"]["auth"]["is_mock"] is False
    assert by_name["kaggle"]["auth"]["account_id"] == "kaggle-default"
    assert by_name["colab"]["auth"]["account_id"] == "colab-default"
    assert by_name["fake"]["auth"]["account_id"] == "fake-default"
    assert "error" not in by_name["kaggle"]["auth"]


def test_kaggle_mock_flow_and_cross_process_routing(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = Client(settings)
    script = _script(tmp_path)
    assert client.auth_status("kaggle")[0].provider is ProviderName.kaggle
    quota = client.get_quota(ProviderName.kaggle)[0]
    assert quota.is_fixture is True
    assert quota.remaining is not None and quota.remaining > 0
    resources = client.list_resources("kaggle")
    t4 = next(item for item in resources if item.gpu_type is GpuType.T4)
    assert t4.gpu_count == 2
    assert t4.gpu_memory_gb_per_gpu == 16

    handle = client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.kaggle,
            gpu_type=GpuType.T4,
            gpu_count=2,
        )
    )
    stored = SqliteJobStore(default_db_path(settings.state_dir)).get(handle.job_id)
    assert stored.provider is ProviderName.kaggle
    assert stored.job_id == handle.job_id
    assert stored.provider_job_id
    assert stored.provider_job_id != handle.job_id
    with pytest.raises(NotSupportedError) as exc:
        client.cancel(handle.job_id)
    assert exc.value.code == "not_supported"
    seen = _poll_until_succeeded(client, handle.job_id)
    assert seen[0] == "QUEUED"
    assert "RUNNING" in seen
    assert seen[-1] == "SUCCEEDED"
    logs = client.logs(handle.job_id)
    assert "MOCK" in logs
    assert "TEST FIXTURE" in logs
    dest = tmp_path / "kaggle-out"
    artifacts = client.download(handle.job_id, dest)
    assert artifacts[0].name == "result.txt"
    assert "MOCK TEST FIXTURE" in artifacts[0].local_path.read_text(encoding="utf-8")

    restored = Client(settings)
    again = restored.status(handle.job_id)
    assert again.state.value == "SUCCEEDED"
    assert again.provider is ProviderName.kaggle
    assert "MOCK" in restored.logs(handle.job_id)


def test_colab_mock_flow_can_cancel(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path))
    handle = client.submit(
        JobSpec(
            script_path=_script(tmp_path),
            provider=ProviderName.colab,
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=3600,
        )
    )
    seen = _poll_until_succeeded(client, handle.job_id)
    assert seen[0] == "QUEUED"
    assert "RUNNING" in seen
    assert client.get_quota("colab")[0].is_fixture is True
    kept = client.cancel(handle.job_id)
    assert kept.state.value == "SUCCEEDED"
    assert "job succeeded" in kept.message
    fresh = client.submit(
        JobSpec(
            script_path=_script(tmp_path, "print('second')\n"),
            provider=ProviderName.colab,
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=3600,
        )
    )
    cancelled = client.cancel(fresh.job_id)
    assert cancelled.state.value == "CANCELLED"
    assert "TEST FIXTURE" in cancelled.message
    assert "colab stop" in cancelled.message


def test_fake_runs_a_local_script(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path))
    handle = client.submit(
        JobSpec(
            script_path=_fake_script(tmp_path),
            provider=ProviderName.fake,
            gpu_type=GpuType.NONE,
            gpu_count=0,
            args=["--unused"],
        )
    )
    with pytest.raises(NotSupportedError):
        client.cancel(handle.job_id)
    result = client.status(handle.job_id)
    assert result.state.value == "SUCCEEDED"
    assert result.is_mock is True
    assert "fake-ran" in client.logs(handle.job_id)
    artifacts = client.download(handle.job_id, tmp_path / "fake-out")
    assert artifacts[0].local_path.read_text(encoding="utf-8") == "from-fake\n"


def test_recommend_backend_rules(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path))
    t4x2 = client.recommend_backend(gpu_type=GpuType.T4, gpu_count=2)
    assert t4x2["provider"] == "kaggle"
    assert t4x2["resource"]["provider_resource_id"] == "NvidiaTeslaT4"
    assert t4x2["resource"]["gpu_count"] == 2
    assert t4x2["resource"]["gpu_memory_gb_per_gpu"] == 16
    assert any("TEST FIXTURE" in item for item in t4x2["warnings"])
    assert not (tmp_path / "state" / "jobs").exists()
    assert not (tmp_path / "state" / "l2kc.db").exists()

    too_big = client.recommend_backend(
        gpu_count=1,
        min_gpu_memory_gb_per_gpu=30,
        timeout_seconds=3600,
    )
    assert too_big["provider"] is None
    joined = " ".join(too_big["reasons"])
    assert "16 GB per GPU" in joined
    assert "never summed" in joined

    blocked = client.recommend_backend(
        gpu_type="H100",
        gpu_count=1,
        allow_premium=False,
        timeout_seconds=3600,
        provider="colab",
    )
    assert blocked["provider"] is None
    assert any("premium" in reason and "allow_premium" in reason for reason in blocked["reasons"])
    premium = Client(
        Settings(
            mode="mock",
            allow_real_jobs=False,
            state_dir=tmp_path / "premium",
            artifact_root=tmp_path,
            scheduler=SchedulerSettings(allow_premium_gpu=True),
        )
    )
    allowed = premium.recommend_backend(
        gpu_type=GpuType.A100,
        gpu_count=1,
        allow_premium=True,
        timeout_seconds=3600,
    )
    assert allowed["provider"] == "colab"
    assert allowed["candidate_id"] == "colab:A100-40GB"
    assert allowed["resource"]["gpu_type"] == "A100"
    assert allowed["resource"]["premium"] is True
    refused_h100 = premium.recommend_backend(
        gpu_type=GpuType.H100,
        gpu_count=1,
        allow_premium=True,
        timeout_seconds=3600,
        provider="colab",
    )
    assert refused_h100["provider"] is None
    assert any("no CCU rate" in reason for reason in refused_h100["reasons"])

    cpu = client.recommend_backend(gpu_type=GpuType.NONE)
    assert cpu["provider"] == "fake"
    assert cpu["resource"]["cpu_only"] is True
    assert client.recommend_backend(gpu_count=0)["provider"] == "fake"


def test_recommend_falls_back_to_colab_when_kaggle_quota_is_a_fixture_zero(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path, kaggle_scenario="quota_insufficient"))
    decision = client.recommend_backend(
        gpu_type=GpuType.T4,
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert decision["provider"] == "colab"
    assert decision["candidate_id"] == "colab:T4"
    assert decision["resource"]["gpu_type"] == "T4"
    assert decision["fallback_used"] is True
    assert any("0 GPU-hours remaining" in reason for reason in decision["reasons"])
    assert any("TEST FIXTURE" in reason for reason in decision["reasons"])


def test_recommend_flags_unknown_kaggle_quota(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class UnknownQuota(MockKaggleProvider):
        def get_quota(self) -> QuotaInfo:
            return QuotaInfo.unknown(ProviderName.kaggle, "not reported")

    client = Client(settings, providers={ProviderName.kaggle: UnknownQuota(settings)})
    decision = client.recommend_backend(gpu_type=GpuType.T4, gpu_count=2)
    assert decision["provider"] == "kaggle"
    assert any("quota unknown" in warning for warning in decision["warnings"])


def test_disabled_provider_is_skipped(tmp_path: Path) -> None:
    settings = Settings(
        state_dir=tmp_path / "state",
        artifact_root=tmp_path,
        providers=ProvidersSettings(kaggle=KaggleProviderSettings(enabled=False)),
    )
    client = Client(settings)
    rows = {row["name"]: row for row in client.list_providers()}
    assert rows["kaggle"]["enabled"] is False
    assert rows["kaggle"]["capabilities"] is None
    with pytest.raises(ConfigError) as exc:
        client.auth_status("kaggle")
    assert exc.value.code == "config_error"
    decision = client.recommend_backend(
        gpu_type=GpuType.T4,
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert decision["provider"] == "colab"
    assert any("kaggle is disabled" in reason for reason in decision["reasons"])


def test_unknown_job_and_provider_errors_are_structured(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = Client(settings)
    with pytest.raises(JobNotFoundError) as missing:
        client.status("ab" * 16)
    assert missing.value.code == "job_not_found"
    with pytest.raises(JobNotFoundError):
        client.download("not-a-job", tmp_path / "out")

    exhausted = Client(_settings(tmp_path, kaggle_scenario="quota_insufficient"))
    with pytest.raises(ResourceUnavailableError) as quota_error:
        exhausted.submit(
            JobSpec(
                script_path=_script(tmp_path),
                provider=ProviderName.kaggle,
                gpu_type=GpuType.T4,
                gpu_count=2,
            )
        )
    assert quota_error.value.code == "resource_unavailable"
    refused = quota_error.value.details["decision"]
    refused_text = " ".join(refused["reasons"])
    assert "0 GPU-hours remaining" in refused_text
    assert "TEST FIXTURE" in refused_text
    assert refused["provider"] is None

    class Boom(FakeProvider):
        def submit(self, spec: JobSpec) -> JobHandle:
            del spec
            raise RuntimeError("boom KGAT_abcdefghijklmnopqrstuvwxyz")

    wrapped = Client(settings, providers={ProviderName.fake: Boom(settings)})
    with pytest.raises(L2KCError) as internal:
        wrapped.submit(
            JobSpec(
                script_path=_script(tmp_path, "print('x')\n"),
                provider=ProviderName.fake,
                gpu_type=GpuType.NONE,
                gpu_count=0,
            )
        )
    assert internal.value.code == "submit_outcome_unknown"
    assert "KGAT_" not in internal.value.message
    assert "[REDACTED]" in internal.value.message
    ambiguous = SqliteJobStore(default_db_path(settings.state_dir)).list()
    assert len(ambiguous) == 1
    assert ambiguous[0].state.value == "UNKNOWN"


def test_failing_kaggle_auth_and_unreadable_index(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path, kaggle_scenario="token_missing"))
    decision = client.recommend_backend(
        gpu_type=GpuType.T4,
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert decision["provider"] == "colab"
    assert any("kaggle auth is unauthenticated" in reason for reason in decision["reasons"])
    with pytest.raises(InvalidInputError):
        client.auth_status("no-such-provider")
    with pytest.raises(InvalidInputError):
        client.recommend_backend(gpu_type="R100")
    with pytest.raises(PremiumNotAllowed):
        client.submit(
            JobSpec(script_path=_script(tmp_path), gpu_type=GpuType.H100, allow_premium=False)
        )

    settings = _settings(tmp_path)
    owner = Client(settings)
    handle = owner.submit(
        JobSpec(
            script_path=_script(tmp_path, "print('indexed')\n"),
            provider=ProviderName.fake,
            gpu_type=GpuType.NONE,
            gpu_count=0,
        )
    )
    stored = SqliteJobStore(default_db_path(settings.state_dir)).get(handle.job_id)
    assert stored.provider is ProviderName.fake
    assert stored.job_id == handle.job_id
    index = settings.state_dir / "jobs" / f"{handle.job_id}.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("{", encoding="utf-8")
    restored = Client(settings)
    again = restored.status(handle.job_id)
    assert again.provider is ProviderName.fake
    assert again.job_id == handle.job_id
    assert index.read_text(encoding="utf-8") == "{"
    with pytest.raises(JobNotFoundError) as malformed:
        restored.status("not-a-job")
    assert malformed.value.code == "job_not_found"
    with pytest.raises(JobNotFoundError) as unknown:
        restored.status("ab" * 16)
    assert unknown.value.code == "job_not_found"


def test_submit_without_provider_uses_recommendation(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path))
    handle = client.submit(
        JobSpec(script_path=_script(tmp_path), gpu_type=GpuType.T4, gpu_count=2)
    )
    assert handle.provider is ProviderName.kaggle
    assert Client(_settings(tmp_path)).status(handle.job_id).provider is ProviderName.kaggle


def test_unspecified_gpu_is_resolved_before_submit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class Capture(MockKaggleProvider):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self.specs: list[JobSpec] = []

        def submit(self, spec: JobSpec) -> JobHandle:
            self.specs.append(spec)
            return super().submit(spec)

    capture = Capture(settings)
    client = Client(settings, providers={ProviderName.kaggle: capture})
    handle = client.submit(
        JobSpec(
            script_path=_script(tmp_path),
            provider=ProviderName.kaggle,
            gpu_count=1,
        )
    )
    assert capture.specs[0].gpu_type is GpuType.T4
    assert client.status(handle.job_id).details["gpu_type"] == "T4"

    class OnlyL4(MockColabProvider):
        seen: JobSpec

        def list_resources(self) -> list[ResourceSpec]:
            return [
                ResourceSpec(
                    gpu_type=GpuType.L4,
                    gpu_count=1,
                    gpu_memory_gb_per_gpu=24,
                    provider_resource_id="L4",
                    premium=False,
                    label="L4",
                )
            ]

        def submit(self, spec: JobSpec) -> JobHandle:
            self.seen = spec
            return super().submit(spec)

    only = OnlyL4(settings)
    colab = Client(settings, providers={ProviderName.colab: only})
    colab_handle = colab.submit(
        JobSpec(
            script_path=_script(tmp_path, "print('l4')\n"),
            provider=ProviderName.colab,
            gpu_count=1,
            timeout_seconds=3600,
        )
    )
    assert only.seen.gpu_type is GpuType.L4
    assert colab.status(colab_handle.job_id).details["gpu_type"] == "L4"

    with pytest.raises(ResourceUnavailableError) as missing:
        client.submit(
            JobSpec(
                script_path=_script(tmp_path, "print('too many')\n"),
                provider=ProviderName.kaggle,
                gpu_count=8,
            )
        )
    assert missing.value.code == "resource_unavailable"
    plain = Client(_settings(tmp_path / "fake-gpu"))
    with pytest.raises(ResourceUnavailableError) as fake_gpu:
        plain.submit(
            JobSpec(
                script_path=_script(tmp_path, "print('fake gpu')\n"),
                provider=ProviderName.fake,
                gpu_count=1,
            )
        )
    assert fake_gpu.value.code == "resource_unavailable"


def test_real_client_none_gpu_selects_t4_and_zero_is_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RecordingRunner stands in for the real CLIs. Nothing here is networked."""
    runner = RecordingRunner()
    adc = tmp_path / "adc.json"
    adc.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fixture-token")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc))
    settings = Settings(
        mode="real",
        allow_real_jobs=True,
        state_dir=tmp_path / "state",
        artifact_root=tmp_path,
        providers={"kaggle": {"username": "tester"}},
    )
    client = Client(settings, runner=runner)
    script = _script(tmp_path)
    runner.script(
        ["kaggle", "quota", "--format", "json"],
        stdout='[{"resource": "GPU", "total": 30, "used": 10, "remaining": 20}]\n',
        permanent=True,
    )
    runner.script(
        ["colab", "--auth", "adc", "usage"],
        stdout="Current balance: 200 compute units\nUsage rate: 1.8/hr\n",
        permanent=True,
    )
    runner.script(
        ["kaggle", "kernels", "push"],
        stdout="Kernel version 1 successfully pushed.\n",
        permanent=True,
    )
    client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.kaggle,
            gpu_type=None,
            gpu_count=1,
        )
    )
    pushes = [call for call in runner.calls if call[1:3] == ["kernels", "push"]]
    push = pushes[0]
    assert push[push.index("--accelerator") + 1] == "NvidiaTeslaT4"
    meta = json.loads((Path(push[4]) / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["enable_gpu"] is True

    client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.kaggle,
            gpu_type=None,
            gpu_count=0,
        )
    )
    pushes = [call for call in runner.calls if call[1:3] == ["kernels", "push"]]
    cpu_push = pushes[1]
    assert "--accelerator" not in cpu_push
    cpu_meta = json.loads(
        (Path(cpu_push[4]) / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    assert cpu_meta["enable_gpu"] is False
    assert "machine_shape" not in cpu_meta

    def script_colab() -> None:
        runner.script(["colab", "--auth", "adc", "new"])
        runner.script(["colab", "--auth", "adc", "exec"], stdout="ok\n")
        runner.script(["colab", "--auth", "adc", "ls"], stdout="")
        runner.script(["colab", "--auth", "adc", "stop"])

    script_colab()
    before = len(runner.calls)
    client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.colab,
            gpu_type=None,
            gpu_count=1,
            timeout_seconds=3600,
        )
    )
    new = next(call for call in runner.calls[before:] if call[3] == "new")
    assert new[-2:] == ["--gpu", "T4"]

    script_colab()
    before = len(runner.calls)
    client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.colab,
            gpu_type=None,
            gpu_count=0,
            timeout_seconds=600,
        )
    )
    cpu_new = next(call for call in runner.calls[before:] if call[3] == "new")
    assert "--gpu" not in cpu_new
