"""Every Colab mock scenario. Balances and membership labels are TEST FIXTURES."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from link2kaggle_colab.core.config import SchedulerSettings, Settings
from link2kaggle_colab.core.errors import (
    AuthError,
    AuthExpiredError,
    AuthMissingError,
    ConfigError,
    InvalidInputError,
    PremiumNotAllowed,
    QuotaExceededError,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import GpuType, JobSpec, JobState
from link2kaggle_colab.providers.colab.mock import MockColabProvider


def _provider(scenario: str, state_dir: Path) -> MockColabProvider:
    return MockColabProvider(scenario=scenario, state_dir=state_dir / scenario)


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("ok", None),
        ("free", None),
        ("ai_pro", None),
        ("ccu_insufficient", QuotaExceededError),
        ("premium_unavailable", None),
        ("runtime_disconnect", None),
        ("oauth_ok", None),
        ("oauth_fail", AuthError),
        ("token_expired", AuthExpiredError),
        ("refresh_fail", AuthExpiredError),
        ("adc_valid", None),
        ("adc_missing", AuthMissingError),
    ],
)
def test_colab_scenario_submit(
    scenario: str,
    expected: type[Exception] | None,
    py_script: Path,
    state_dir: Path,
) -> None:
    provider = _provider(scenario, state_dir)
    gpu = GpuType.T4
    allow_premium = False
    if scenario == "premium_unavailable":
        gpu = GpuType.T4
    spec = JobSpec(script_path=py_script, gpu_type=gpu, allow_premium=allow_premium)
    if expected is not None:
        with pytest.raises(expected) as caught:
            provider.submit(spec)
        if expected is AuthError:
            assert type(caught.value) is AuthError
        return
    handle = provider.submit(spec)
    if scenario == "runtime_disconnect":
        assert provider.status(handle.job_id).state is JobState.QUEUED
        assert provider.status(handle.job_id).state is JobState.RUNNING
        failed = provider.status(handle.job_id)
        assert failed.state is JobState.FAILED
        assert "runtime_disconnected" in failed.message
        assert "disconnected" in failed.message
        assert failed.state.is_terminal()
        return
    assert provider.status(handle.job_id).state is JobState.QUEUED
    assert provider.status(handle.job_id).state is JobState.RUNNING
    assert provider.status(handle.job_id).state is JobState.SUCCEEDED


def test_colab_quota_resources_and_premium(py_script: Path, state_dir: Path) -> None:
    free = _provider("free", state_dir)
    quota = free.get_quota()
    assert quota.is_fixture is True
    assert quota.known is False
    assert quota.remaining is None
    assert "TEST FIXTURE" in quota.message
    assert free.auth_status().account_label == "mock-colab-free (TEST FIXTURE)"
    assert {item.gpu_type for item in free.list_resources()} == {GpuType.T4, GpuType.NONE}
    with pytest.raises(ResourceUnavailableError):
        free.submit(JobSpec(script_path=py_script, gpu_type=GpuType.L4))

    pro = _provider("ai_pro", state_dir)
    balance = pro.get_quota()
    assert balance.remaining == 200
    assert balance.is_fixture is True
    assert "Google AI Pro (MOCK)" in balance.message
    assert pro.auth_status().account_label == "Google AI Pro (MOCK)"
    with pytest.raises(PremiumNotAllowed) as denied:
        pro.submit(JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=False))
    assert denied.value.code == "premium_not_allowed"
    with pytest.raises(PremiumNotAllowed):
        pro.submit(JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=True))
    opened = MockColabProvider(
        Settings(
            state_dir=state_dir / "ai-pro-allowed",
            scheduler=SchedulerSettings(allow_premium_gpu=True),
        ),
        scenario="ai_pro",
        state_dir=state_dir / "ai-pro-allowed",
    )
    handle = opened.submit(
        JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=True)
    )
    assert handle.job_id

    blocked = _provider("premium_unavailable", state_dir)
    with pytest.raises(ResourceUnavailableError):
        blocked.submit(JobSpec(script_path=py_script, gpu_type=GpuType.H100, allow_premium=True))
    with pytest.raises(ResourceUnavailableError):
        blocked.submit(JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=True))

    short = _provider("ccu_insufficient", state_dir)
    assert short.get_quota().remaining == 2
    with pytest.raises(QuotaExceededError):
        short.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))


def test_refresh_then_submit_and_cancel(py_script: Path, state_dir: Path) -> None:
    provider = _provider("token_expired", state_dir)
    with pytest.raises(AuthExpiredError):
        provider.submit(JobSpec(script_path=py_script))
    refreshed = provider.auth.refresh()
    assert refreshed.state.value == "authenticated"
    assert refreshed.expires_at is not None
    assert refreshed.expires_at > datetime.now(UTC)
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    cancelled = provider.cancel(handle.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert "colab stop" in cancelled.message
    assert provider.status(handle.job_id).state is JobState.CANCELLED
    assert provider.capabilities().supports_cancel is True


def test_cancel_keeps_terminal_jobs_and_download_requires_success(
    py_script: Path,
    state_dir: Path,
) -> None:
    provider = MockColabProvider(scenario="ok", state_dir=state_dir)
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    with pytest.raises(InvalidInputError) as early:
        provider.download(handle.job_id, state_dir / "early")
    assert early.value.code == "invalid_input"
    assert provider.status(handle.job_id).state is JobState.QUEUED
    assert provider.status(handle.job_id).state is JobState.RUNNING
    done = provider.status(handle.job_id)
    assert done.state is JobState.SUCCEEDED
    kept = provider.cancel(handle.job_id)
    assert kept.state is JobState.SUCCEEDED
    assert kept.message == done.message
    assert "colab stop" not in kept.message
    assert provider.status(handle.job_id).state is JobState.SUCCEEDED
    artifacts = provider.download(handle.job_id, state_dir / "dest")
    assert artifacts[0].name == "result.txt"

    failed = MockColabProvider(scenario="runtime_disconnect", state_dir=state_dir / "fail")
    broken = failed.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    assert failed.status(broken.job_id).state is JobState.QUEUED
    assert failed.status(broken.job_id).state is JobState.RUNNING
    assert failed.status(broken.job_id).state is JobState.FAILED
    assert failed.cancel(broken.job_id).state is JobState.FAILED
    with pytest.raises(InvalidInputError) as denied:
        failed.download(broken.job_id, state_dir / "fail-out")
    assert denied.value.code == "invalid_input"

    live = provider.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=1))
    cancelled = provider.cancel(live.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert "colab stop" in cancelled.message
    assert cancelled.details["gpu_type"] == "T4"
    assert provider.cancel(live.job_id).state is JobState.CANCELLED
    assert "gpu=T4" in provider.logs(live.job_id)

    cpu = provider.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=0))
    assert provider.status(cpu.job_id).details["gpu_type"] == "NONE"


def test_unknown_colab_scenario(state_dir: Path) -> None:
    with pytest.raises(ConfigError):
        MockColabProvider(scenario="nope", state_dir=state_dir)


def test_colab_job_persists_across_instances(py_script: Path, state_dir: Path) -> None:
    first = MockColabProvider(scenario="ok", state_dir=state_dir)
    handle = first.submit(JobSpec(script_path=py_script, gpu_type=GpuType.L4))
    second = MockColabProvider(scenario="ok", state_dir=state_dir)
    assert second.status(handle.job_id).state is JobState.QUEUED
    with pytest.raises(InvalidInputError) as early:
        second.download(handle.job_id, state_dir / "dest")
    assert early.value.code == "invalid_input"
    assert second.status(handle.job_id).state is JobState.RUNNING
    assert second.status(handle.job_id).state is JobState.SUCCEEDED
    text = second.logs(handle.job_id)
    assert "[MOCK]" in text
    assert "gpu=L4" in text
    artifacts = second.download(handle.job_id, state_dir / "dest")
    assert artifacts[0].local_path.is_file()
    assert "MOCK" in artifacts[0].local_path.read_text(encoding="utf-8")


def test_colab_quota_publishes_eligible_gpus(state_dir: Path) -> None:
    free = _provider("free", state_dir)
    assert free.get_quota().eligible_gpus == ["T4"]
    pro = _provider("ai_pro", state_dir)
    assert pro.get_quota().eligible_gpus == ["T4", "L4", "A100", "G4", "TPU_V5E1", "TPU_V6E1"]
    ok = _provider("ok", state_dir)
    assert ok.get_quota().eligible_gpus == ["T4", "L4", "A100", "G4", "TPU_V5E1", "TPU_V6E1"]

def test_colab_mock_tpu_and_high_mem(state_dir: Path, py_script: Path) -> None:
    from pydantic import ValidationError

    pro = _provider("ai_pro", state_dir)
    resources = {item.gpu_type for item in pro.list_resources()}
    assert GpuType.TPU_V5E1 in resources
    assert GpuType.TPU_V6E1 in resources
    t4 = next(item for item in pro.list_resources() if item.gpu_type is GpuType.T4)
    assert t4.supports_high_mem is True
    with pytest.raises(PremiumNotAllowed):
        pro.submit(JobSpec(script_path=py_script, gpu_type=GpuType.TPU_V5E1, allow_premium=False))
    opened = MockColabProvider(
        Settings(
            state_dir=state_dir / "tpu-allowed",
            scheduler=SchedulerSettings(allow_premium_gpu=True),
        ),
        scenario="ai_pro",
        state_dir=state_dir / "tpu-allowed",
    )
    handle = opened.submit(
        JobSpec(script_path=py_script, gpu_type=GpuType.TPU_V5E1, allow_premium=True)
    )
    assert opened.status(handle.job_id).details["gpu_type"] == "TPU_V5E1"
    handle_hm = pro.submit(
        JobSpec(script_path=py_script, gpu_type=GpuType.T4, high_mem=True)
    )
    assert pro.status(handle_hm.job_id).details["gpu_type"] == "T4"
    with pytest.raises(ValidationError):
        JobSpec(script_path=py_script, gpu_type=GpuType.L4, high_mem=True)

