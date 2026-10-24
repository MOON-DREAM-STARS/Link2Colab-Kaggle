"""Every Kaggle mock scenario. Numbers here are TEST FIXTURES."""

from pathlib import Path

import pytest

from link2kaggle_colab.core.errors import (
    AuthError,
    AuthInvalidError,
    AuthMissingError,
    ConfigError,
    InvalidInputError,
    NotSupportedError,
    PermissionDeniedError,
    QuotaExceededError,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import GpuType, JobSpec, JobState
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.providers.mock_fixtures import MOCK_RESULT_TEXT


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("ok", None),
        ("quota_45h", None),
        ("quota_insufficient", QuotaExceededError),
        ("gpu_unavailable", ResourceUnavailableError),
        ("token_valid", None),
        ("token_invalid", AuthInvalidError),
        ("token_missing", AuthMissingError),
        ("oauth_ok", None),
        ("oauth_fail", AuthError),
        ("permission_error", PermissionDeniedError),
    ],
)
def test_kaggle_scenario_submit(
    scenario: str,
    expected: type[Exception] | None,
    py_script: Path,
    state_dir: Path,
) -> None:
    provider = MockKaggleProvider(scenario=scenario, state_dir=state_dir)
    spec = JobSpec(script_path=py_script, gpu_type=GpuType.T4, name="train-1")
    if expected is None:
        handle = provider.submit(spec)
        assert provider.status(handle.job_id).state is JobState.QUEUED
        assert provider.status(handle.job_id).state is JobState.RUNNING
        done = provider.status(handle.job_id)
        assert done.state is JobState.SUCCEEDED
        assert done.state.is_terminal()
        assert done.is_mock is True
        assert provider.status(handle.job_id).state is JobState.SUCCEEDED
        return
    with pytest.raises(expected) as caught:
        provider.submit(spec)
    if expected is AuthError:
        assert type(caught.value) is AuthError


def test_cpu_submit_ignores_gpu_quota_and_availability(py_script: Path, state_dir: Path) -> None:
    for scenario in ("quota_insufficient", "gpu_unavailable"):
        provider = MockKaggleProvider(scenario=scenario, state_dir=state_dir / scenario)
        handle = provider.submit(
            JobSpec(script_path=py_script, gpu_type=GpuType.NONE, gpu_count=0)
        )
        assert handle.provider.value == "kaggle"


def test_kaggle_quota_fixtures(state_dir: Path) -> None:
    ok = MockKaggleProvider(scenario="ok", state_dir=state_dir).get_quota()
    assert ok.is_fixture is True
    assert ok.source == "mock-fixture"
    assert ok.total == 30
    assert ok.used == 12.5
    assert ok.remaining == 17.5
    assert ok.known is True
    assert "TEST FIXTURE" in ok.message
    wider = MockKaggleProvider(scenario="quota_45h", state_dir=state_dir).get_quota()
    assert wider.total == 45
    assert wider.remaining == 45
    empty = MockKaggleProvider(scenario="quota_insufficient", state_dir=state_dir).get_quota()
    assert empty.remaining == 0
    assert empty.known is True


def test_kaggle_cancel_logs_download_and_persistence(py_script: Path, state_dir: Path) -> None:
    provider = MockKaggleProvider(scenario="ok", state_dir=state_dir)
    assert provider.capabilities().supports_cancel is False
    assert provider.capabilities().is_batch_only is True
    handle = provider.submit(JobSpec(script_path=py_script, gpu_type=GpuType.T4))
    with pytest.raises(NotSupportedError):
        provider.cancel(handle.job_id)
    text = provider.logs(handle.job_id)
    assert "[MOCK]" in text
    assert "TEST FIXTURE" in text
    assert "gpu=T4" in text
    dest = state_dir / "out"
    with pytest.raises(InvalidInputError) as early:
        provider.download(handle.job_id, dest)
    assert early.value.code == "invalid_input"
    other = MockKaggleProvider(scenario="ok", state_dir=state_dir)
    assert other.status(handle.job_id).state is JobState.QUEUED
    assert other.status(handle.job_id).state is JobState.RUNNING
    done = other.status(handle.job_id)
    assert done.state is JobState.SUCCEEDED
    assert done.details["gpu_type"] == "T4"
    with pytest.raises(NotSupportedError):
        other.cancel(handle.job_id)
    artifacts = other.download(handle.job_id, dest)
    assert artifacts[0].name == "result.txt"
    assert (dest / "result.txt").read_text(encoding="utf-8") == MOCK_RESULT_TEXT


def test_none_gpu_defaults_to_t4_and_zero_count_is_cpu(
    py_script: Path,
    state_dir: Path,
) -> None:
    gpu = MockKaggleProvider(scenario="ok", state_dir=state_dir / "gpu")
    handle = gpu.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=1))
    queued = gpu.status(handle.job_id)
    assert queued.details["gpu_type"] == "T4"
    assert "gpu=T4" in gpu.logs(handle.job_id)

    cpu = MockKaggleProvider(scenario="ok", state_dir=state_dir / "cpu")
    cpu_handle = cpu.submit(JobSpec(script_path=py_script, gpu_type=None, gpu_count=0))
    assert cpu.status(cpu_handle.job_id).details["gpu_type"] == "NONE"
    assert "gpu=NONE" in cpu.logs(cpu_handle.job_id)


def test_unknown_kaggle_scenario(state_dir: Path) -> None:
    with pytest.raises(ConfigError):
        MockKaggleProvider(scenario="nope", state_dir=state_dir)


def test_auth_failure_blocks_quota_and_cancel(state_dir: Path) -> None:
    provider = MockKaggleProvider(scenario="token_missing", state_dir=state_dir)
    with pytest.raises(AuthMissingError):
        provider.get_quota()
    with pytest.raises(AuthMissingError):
        provider.cancel("a" * 32)
    denied = MockKaggleProvider(scenario="permission_error", state_dir=state_dir)
    with pytest.raises(PermissionDeniedError):
        denied.cancel("b" * 32)
