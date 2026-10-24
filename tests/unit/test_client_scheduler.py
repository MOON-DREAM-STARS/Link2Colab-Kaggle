"""Client placement. Quotas here are TEST FIXTURES. No network."""

from __future__ import annotations

from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import (
    MockSettings,
    SchedulerSettings,
    Settings,
)
from link2kaggle_colab.core.errors import (
    InvalidInputError,
    L2KCError,
    PremiumNotAllowed,
    ProviderCLIError,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import (
    GpuType,
    JobHandle,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
)
from link2kaggle_colab.providers.colab.cli_provider import ColabCliProvider
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.scheduler.decision import ScheduleRequest
from tests.mocks.runner import RecordingRunner


def _settings(tmp_path: Path, **mock: str) -> Settings:
    scheduler = mock.pop("scheduler", None)
    return Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        mock=MockSettings(
            kaggle_scenario=mock.get("kaggle_scenario", "ok"),
            colab_scenario=mock.get("colab_scenario", "ok"),
        ),
        scheduler=scheduler if isinstance(scheduler, SchedulerSettings) else SchedulerSettings(),
    )


def _script(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "job.py"
    path.write_text("print('hello-from-script')\n", encoding="utf-8")
    return path


class _KaggleSpy(MockKaggleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.submits: list[JobSpec] = []

    def submit(self, spec: JobSpec) -> JobHandle:
        self.submits.append(spec)
        return super().submit(spec)


class _ColabSpy(MockColabProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.submits: list[JobSpec] = []

    def submit(self, spec: JobSpec) -> JobHandle:
        self.submits.append(spec)
        return super().submit(spec)


class _CountingKaggle(MockKaggleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.quota_calls = 0

    def get_quota(self) -> QuotaInfo:
        self.quota_calls += 1
        return super().get_quota()


class _UnknownKaggle(_KaggleSpy):
    def submit(self, spec: JobSpec) -> JobHandle:
        self.submits.append(spec)
        handle = MockKaggleProvider.submit(self, spec)
        stored = self._jobs.load(handle.job_id)
        self._jobs.save(stored.model_copy(update={"state": JobState.UNKNOWN, "states": []}))
        return handle


class _TimeoutKaggle(_KaggleSpy):
    def submit(self, spec: JobSpec) -> JobHandle:
        self.submits.append(spec)
        raise ProviderCLIError(
            "command timed out",
            code="cli_timeout",
            provider=self.name.value,
        )


def _spies(settings: Settings) -> tuple[_KaggleSpy, _ColabSpy, Client]:
    kaggle = _KaggleSpy(settings)
    colab = _ColabSpy(settings)
    client = Client(
        settings,
        providers={ProviderName.kaggle: kaggle, ProviderName.colab: colab},
    )
    return kaggle, colab, client


def _gpu(tmp_path: Path, **updates: object) -> JobSpec:
    body: dict[str, object] = {
        "script_path": _script(tmp_path),
        "gpu_count": 1,
        "timeout_seconds": 3600,
    }
    body.update(updates)
    return JobSpec.model_validate(body)


def test_auto_submit_prefers_kaggle(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path))
    handle, decision = client.submit_with_decision(_gpu(tmp_path))
    assert handle.provider is ProviderName.kaggle
    assert decision is not None
    assert decision.provider is ProviderName.kaggle
    assert decision.candidate_id == "kaggle:T4x2"
    assert decision.fallback_used is False


def test_auto_submit_falls_back_to_colab_when_kaggle_exhausted(tmp_path: Path) -> None:
    client = Client(
        _settings(tmp_path, kaggle_scenario="quota_insufficient", colab_scenario="ai_pro")
    )
    handle, decision = client.submit_with_decision(_gpu(tmp_path))
    assert handle.provider is ProviderName.colab
    assert decision is not None
    assert decision.provider is ProviderName.colab
    assert decision.candidate_id == "colab:T4"
    assert decision.fallback_used is True


def test_fallback_disabled_raises_resource_unavailable(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
        scheduler=SchedulerSettings(allow_colab_fallback=False),
    )
    kaggle, colab, client = _spies(settings)
    with pytest.raises(ResourceUnavailableError) as caught:
        client.submit(_gpu(tmp_path))
    assert caught.value.code == "resource_unavailable"
    assert kaggle.submits == []
    assert colab.submits == []


def test_submit_requiring_24gb_per_gpu_never_uses_t4x2(tmp_path: Path) -> None:
    settings = _settings(tmp_path, colab_scenario="ai_pro")
    kaggle, colab, client = _spies(settings)
    handle, decision = client.submit_with_decision(
        _gpu(tmp_path, min_gpu_memory_gb_per_gpu=24)
    )
    assert handle.provider is ProviderName.colab
    assert decision is not None
    assert decision.candidate_id == "colab:L4"
    assert decision.gpu_type is GpuType.L4
    assert kaggle.submits == []
    assert len(colab.submits) == 1
    assert colab.submits[0].gpu_type is GpuType.L4


def test_colab_cpu_requires_timeout_before_provider_call(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _kaggle, colab, client = _spies(settings)
    spec = JobSpec(
        script_path=_script(tmp_path),
        provider=ProviderName.colab,
        gpu_count=0,
    )
    with pytest.raises(InvalidInputError) as caught:
        client.submit(spec)
    assert caught.value.code == "invalid_input"
    assert caught.value.provider == "colab"
    assert colab.submits == []
    client.submit(spec.model_copy(update={"timeout_seconds": 600}))
    assert len(colab.submits) == 1


def test_client_keep_runtime_reaches_the_selector(tmp_path: Path) -> None:
    settings = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        mock=MockSettings(kaggle_scenario="quota_insufficient", colab_scenario="ai_pro"),
        providers={"colab": {"keep_runtime": True}},
    )
    client = Client(settings)
    auto = client.schedule(ScheduleRequest(gpu_count=1, timeout_seconds=3600))
    assert auto.provider is None
    colab_reasons = [
        reason
        for item in auto.evaluations
        if item.provider is ProviderName.colab
        for reason in item.reasons
    ]
    assert any("keep_runtime" in reason for reason in colab_reasons)
    pinned = client.schedule(
        ScheduleRequest(gpu_count=1, timeout_seconds=3600, provider=ProviderName.colab)
    )
    assert pinned.provider is ProviderName.colab
    assert pinned.candidate_id == "colab:T4"
    assert any("keep_runtime" in warning for warning in pinned.warnings)


def test_missing_timeout_refuses_paid_backend(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
    )
    _kaggle, colab, client = _spies(settings)
    with pytest.raises(ResourceUnavailableError) as caught:
        client.submit(_gpu(tmp_path, timeout_seconds=None))
    assert caught.value.code == "resource_unavailable"
    assert colab.submits == []
    text = " ".join(caught.value.details["decision"]["reasons"])
    assert "timeout_seconds is required" in text


def test_quota_cache_ttl_respected_by_client(tmp_path: Path) -> None:
    clock = {"now": 0.0}

    def now() -> float:
        return clock["now"]

    settings = _settings(tmp_path)
    counting = _CountingKaggle(settings)
    client = Client(settings, providers={ProviderName.kaggle: counting}, clock=now)
    first = client.get_quota(ProviderName.kaggle)
    second = client.get_quota(ProviderName.kaggle)
    assert first[0].remaining == second[0].remaining
    assert counting.quota_calls == 1
    clock["now"] = 299.9
    client.get_quota("kaggle")
    assert counting.quota_calls == 1
    clock["now"] = 300.0
    client.get_quota("kaggle")
    assert counting.quota_calls == 2

    before = counting.quota_calls
    client.submit(_gpu(tmp_path, provider=ProviderName.kaggle))
    assert counting.quota_calls == before
    client.get_quota("kaggle")
    assert counting.quota_calls == before + 1


def test_premium_blocked_by_default_sdk(tmp_path: Path) -> None:
    for pinned in (None, ProviderName.colab):
        settings = _settings(tmp_path / pinned.value if pinned else tmp_path / "auto")
        _kaggle, colab, client = _spies(settings)
        with pytest.raises(PremiumNotAllowed) as caught:
            client.submit(
                _gpu(tmp_path, gpu_type=GpuType.A100, allow_premium=True, provider=pinned)
            )
        assert caught.value.code == "premium_not_allowed"
        assert colab.submits == []


def test_premium_needs_confirmation_sdk(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        scheduler=SchedulerSettings(allow_premium_gpu=True),
    )
    _kaggle, colab, client = _spies(settings)
    with pytest.raises(PremiumNotAllowed) as caught:
        client.submit(_gpu(tmp_path, gpu_type=GpuType.A100, allow_premium=False))
    assert caught.value.code == "premium_not_allowed"
    assert colab.submits == []


def test_premium_allowed_with_config_and_confirmation_sdk(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        colab_scenario="ai_pro",
        scheduler=SchedulerSettings(allow_premium_gpu=True),
    )
    kaggle, colab, client = _spies(settings)
    handle, decision = client.submit_with_decision(
        _gpu(tmp_path, gpu_type=GpuType.A100, allow_premium=True)
    )
    assert handle.provider is ProviderName.colab
    assert decision is not None
    assert decision.provider is ProviderName.colab
    assert decision.gpu_type is GpuType.A100
    assert decision.candidate_id == "colab:A100-40GB"
    assert kaggle.submits == []
    assert len(colab.submits) == 1
    assert colab.submits[0].allow_premium is True
    assert colab.submits[0].gpu_type is GpuType.A100


def test_colab_provider_refuses_premium_when_config_disallows(
    tmp_path: Path,
    py_script: Path,
) -> None:
    mock = MockColabProvider(Settings(state_dir=tmp_path / "mock-state"))
    with pytest.raises(PremiumNotAllowed) as mock_caught:
        mock.submit(
            JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=True)
        )
    assert mock_caught.value.code == "premium_not_allowed"

    runner = RecordingRunner()
    settings = Settings(
        mode="real",
        allow_real_jobs=True,
        state_dir=tmp_path / "real-state",
    )
    provider = ColabCliProvider(
        settings,
        runner=runner,
        home=tmp_path / "home",
        environ={},
    )
    with pytest.raises(PremiumNotAllowed) as cli_caught:
        provider.submit(
            JobSpec(script_path=py_script, gpu_type=GpuType.A100, allow_premium=True)
        )
    assert cli_caught.value.code == "premium_not_allowed"
    assert runner.calls == []


def test_no_fallback_after_ambiguous_submit(tmp_path: Path) -> None:
    unknown_settings = _settings(tmp_path / "unknown")
    unknown = _UnknownKaggle(unknown_settings)
    unknown_colab = _ColabSpy(unknown_settings)
    unknown_client = Client(
        unknown_settings,
        providers={ProviderName.kaggle: unknown, ProviderName.colab: unknown_colab},
    )
    handle, decision = unknown_client.submit_with_decision(_gpu(tmp_path / "unknown"))
    assert decision is not None
    assert decision.provider is ProviderName.kaggle
    assert len(unknown.submits) == 1
    assert unknown_colab.submits == []
    assert handle.provider is ProviderName.kaggle
    assert unknown_client.status(handle.job_id).state is JobState.UNKNOWN

    failed_settings = _settings(tmp_path / "timeout")
    failing = _TimeoutKaggle(failed_settings)
    failed_colab = _ColabSpy(failed_settings)
    failed_client = Client(
        failed_settings,
        providers={ProviderName.kaggle: failing, ProviderName.colab: failed_colab},
    )
    with pytest.raises(L2KCError) as caught:
        failed_client.submit(_gpu(tmp_path / "timeout"))
    assert isinstance(caught.value, ProviderCLIError)
    assert caught.value.code == "cli_timeout"
    assert len(failing.submits) == 1
    assert failed_colab.submits == []


def test_submit_with_decision_returns_reasons(tmp_path: Path) -> None:
    client = Client(
        _settings(tmp_path, kaggle_scenario="quota_insufficient", colab_scenario="ai_pro")
    )
    _handle, decision = client.submit_with_decision(_gpu(tmp_path))
    assert decision is not None
    assert decision.evaluations
    rejected = [item for item in decision.evaluations if not item.accepted]
    assert rejected
    assert all(item.reasons for item in rejected)
