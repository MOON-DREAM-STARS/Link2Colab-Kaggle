"""Quota-aware Kaggle-first placement. Balances here are TEST FIXTURES."""

from __future__ import annotations

import json

import pytest

from link2kaggle_colab.core.config import SchedulerSettings
from link2kaggle_colab.core.models import (
    AuthMethod,
    AuthState,
    AuthStatus,
    GpuType,
    ProviderName,
    QuotaInfo,
    QuotaUnit,
)
from link2kaggle_colab.providers.mock_fixtures import (
    colab_quota,
    colab_resources,
    kaggle_quota,
    kaggle_resources,
)
from link2kaggle_colab.quota.cache import CachedQuota
from link2kaggle_colab.scheduler.decision import (
    CandidateEvaluation,
    SchedulerDecision,
    ScheduleRequest,
)
from link2kaggle_colab.scheduler.selector import select_backend
from link2kaggle_colab.scheduler.snapshot import ProviderSnapshot

_DEFAULT = object()
_QUOTA_KEYS = {"unit", "remaining", "known", "is_fixture", "from_cache", "age_seconds"}


def _auth(provider: ProviderName, state: AuthState) -> AuthStatus:
    return AuthStatus(
        provider=provider,
        state=state,
        method=AuthMethod.mock,
        account_label="mock-scheduler (TEST FIXTURE)",
        is_mock=True,
        message="TEST FIXTURE auth",
    )


def _cached(
    quota: QuotaInfo,
    *,
    from_cache: bool,
    age_seconds: float,
) -> CachedQuota:
    return CachedQuota(
        quota=quota,
        fetched_at=1_000.0,
        age_seconds=age_seconds,
        from_cache=from_cache,
    )


def _resolve_quota(
    override: QuotaInfo | None | object,
    scenario: str | None,
    load,
) -> QuotaInfo | None:
    if override is not _DEFAULT:
        if override is None or isinstance(override, QuotaInfo):
            return override
        raise AssertionError("quota override must be QuotaInfo or None")
    if scenario is None:
        return None
    return load(scenario)


def _decide(
    *,
    kaggle_scenario: str | None = "ok",
    colab_scenario: str | None = "ai_pro",
    colab_resources_scenario: str | None = None,
    kaggle_quota_info: QuotaInfo | None | object = _DEFAULT,
    colab_quota_info: QuotaInfo | None | object = _DEFAULT,
    gpu_type: GpuType | None = None,
    gpu_count: int = 1,
    min_memory: float | None = None,
    timeout: int | None = 3600,
    premium_confirmed: bool = False,
    provider: ProviderName | None = None,
    settings: SchedulerSettings | None = None,
    kaggle_auth: AuthState = AuthState.authenticated,
    colab_auth: AuthState = AuthState.authenticated,
    kaggle_enabled: bool = True,
    colab_enabled: bool = True,
    kaggle_error: str | None = None,
    colab_error: str | None = None,
    from_cache: bool = False,
    age_seconds: float = 0.0,
    keeps_runtime: bool = False,
    high_mem: bool = False,
) -> tuple[SchedulerDecision, list[ProviderName]]:
    kaggle_info = _resolve_quota(kaggle_quota_info, kaggle_scenario, kaggle_quota)
    colab_info = _resolve_quota(colab_quota_info, colab_scenario, colab_quota)
    resource_scenario = colab_resources_scenario or colab_scenario or "ai_pro"
    catalog = {
        ProviderName.kaggle: (
            kaggle_info,
            kaggle_resources(),
            kaggle_enabled,
            kaggle_auth,
            kaggle_error,
        ),
        ProviderName.colab: (
            colab_info,
            colab_resources(resource_scenario),
            colab_enabled,
            colab_auth,
            colab_error,
        ),
    }
    calls: list[ProviderName] = []

    def snapshots(name: ProviderName) -> ProviderSnapshot:
        calls.append(name)
        info, resources, enabled, auth_state, error = catalog[name]
        return ProviderSnapshot(
            provider=name,
            enabled=enabled,
            auth=_auth(name, auth_state),
            quota=(
                None
                if info is None
                else _cached(info, from_cache=from_cache, age_seconds=age_seconds)
            ),
            resources=resources,
            error=error,
            keeps_runtime=keeps_runtime if name is ProviderName.colab else False,
        )

    request = ScheduleRequest(
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        min_gpu_memory_gb_per_gpu=min_memory,
        timeout_seconds=timeout,
        premium_confirmed=premium_confirmed,
        high_mem=high_mem,
        provider=provider,
    )
    decision = select_backend(request, settings or SchedulerSettings(), snapshots)
    return decision, calls


def _one(decision: SchedulerDecision, candidate_id: str) -> CandidateEvaluation:
    matches = [item for item in decision.evaluations if item.candidate_id == candidate_id]
    assert len(matches) == 1, candidate_id
    return matches[0]


def _joined(reasons: list[str]) -> str:
    return " ".join(reasons)


def test_kaggle_first_when_quota_available() -> None:
    decision, calls = _decide(gpu_count=1, timeout=3600, from_cache=True, age_seconds=4.0)
    assert decision.strategy == "kaggle_first"
    assert decision.provider is ProviderName.kaggle
    assert decision.candidate_id == "kaggle:T4x2"
    assert decision.gpu_type is GpuType.T4
    assert decision.fallback_used is False
    assert decision.estimated_ccu is None
    assert decision.warnings == []
    assert calls == [ProviderName.kaggle]
    assert decision.resource is not None
    assert decision.resource.gpu_type is GpuType.T4
    assert decision.resource.gpu_count == 2
    assert decision.resource.gpu_memory_gb_per_gpu == 16
    quota = _one(decision, "kaggle:T4x2").quota
    assert quota is not None
    assert set(quota) == _QUOTA_KEYS
    assert quota["unit"] == "gpu_hours"
    assert quota["remaining"] == 17.5
    assert quota["known"] is True
    assert quota["is_fixture"] is True
    assert quota["from_cache"] is True
    assert quota["age_seconds"] == 4.0


def test_kaggle_45h_quota_selects_kaggle() -> None:
    decision, _calls = _decide(kaggle_scenario="quota_45h", timeout=3600)
    assert decision.candidate_id == "kaggle:T4x2"
    assert decision.provider is ProviderName.kaggle
    assert decision.fallback_used is False
    assert decision.estimated_ccu is None
    assert _one(decision, "kaggle:T4x2").quota is not None
    assert _one(decision, "kaggle:T4x2").quota["remaining"] == 45


def test_colab_fallback_when_kaggle_quota_exhausted() -> None:
    decision, _calls = _decide(kaggle_scenario="quota_insufficient", colab_scenario="ai_pro")
    assert decision.provider is ProviderName.colab
    assert decision.candidate_id == "colab:T4"
    assert decision.fallback_used is True
    assert decision.estimated_ccu == 1.19
    assert "exhausted" in _joined(_one(decision, "kaggle:T4x2").reasons)
    assert not any("keep_runtime" in warning for warning in decision.warnings)


def test_keep_runtime_rejects_unpinned_colab() -> None:
    decision, _calls = _decide(
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
        keeps_runtime=True,
    )
    assert decision.provider is None
    colab = [item for item in decision.evaluations if item.provider is ProviderName.colab]
    assert colab
    assert any("keep_runtime" in reason for item in colab for reason in item.reasons)


def test_keep_runtime_pinned_colab_selects_with_warning() -> None:
    decision, _calls = _decide(
        provider=ProviderName.colab,
        colab_scenario="ai_pro",
        keeps_runtime=True,
    )
    assert decision.provider is ProviderName.colab
    assert decision.candidate_id == "colab:T4"
    assert any("keep_runtime" in warning for warning in decision.warnings)


def test_kaggle_quota_less_than_timeout_falls_back() -> None:
    decision, _calls = _decide(
        kaggle_scenario="ok",
        colab_scenario="ai_pro",
        timeout=20 * 3600,
        settings=SchedulerSettings(max_ccu_per_job=100),
    )
    assert decision.candidate_id == "colab:T4"
    assert decision.provider is ProviderName.colab
    assert decision.fallback_used is True
    assert decision.estimated_ccu == 23.8
    assert "GPU-hours" in _joined(_one(decision, "kaggle:T4x2").reasons)
    assert "17.5" in _joined(_one(decision, "kaggle:T4x2").reasons)


def test_fallback_disabled_returns_no_backend() -> None:
    decision, calls = _decide(
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
        settings=SchedulerSettings(allow_colab_fallback=False),
    )
    assert decision.provider is None
    assert decision.candidate_id is None
    assert "no backend satisfies the request" in decision.reasons
    assert calls == [ProviderName.kaggle]
    colab = [item for item in decision.evaluations if item.provider is ProviderName.colab]
    assert colab
    for evaluation in colab:
        assert evaluation.accepted is False
        assert "allow_colab_fallback is false" in evaluation.reasons


def test_pinned_colab_ignores_fallback_flag() -> None:
    decision, calls = _decide(
        provider=ProviderName.colab,
        colab_scenario="ai_pro",
        settings=SchedulerSettings(allow_colab_fallback=False),
    )
    assert decision.provider is ProviderName.colab
    assert decision.candidate_id == "colab:T4"
    assert decision.fallback_used is False
    assert calls == [ProviderName.colab]
    for evaluation in decision.evaluations:
        assert "allow_colab_fallback is false" not in _joined(evaluation.reasons)


def test_per_gpu_memory_t4x2_never_32gb() -> None:
    decision, calls = _decide(min_memory=24, timeout=3600)
    kaggle = _one(decision, "kaggle:T4x2")
    assert kaggle.accepted is False
    assert "never summed" in _joined(kaggle.reasons)
    assert "16" in _joined(kaggle.reasons)
    assert "24" in _joined(kaggle.reasons)
    assert decision.candidate_id == "colab:L4"
    assert decision.gpu_type is GpuType.L4
    assert decision.estimated_ccu == 1.71
    assert calls == [ProviderName.colab]


def test_min_memory_20gb_rejects_t4x2_even_with_two_gpus() -> None:
    decision, _calls = _decide(min_memory=20, timeout=3600)
    kaggle = _one(decision, "kaggle:T4x2")
    assert kaggle.accepted is False
    text = _joined(kaggle.reasons)
    assert "never summed" in text
    assert "across 2 GPUs" in text
    assert "16" in text
    assert "20" in text
    assert decision.candidate_id == "colab:L4"
    assert _one(decision, "colab:T4").accepted is False


def test_gpu_count_two_only_matches_t4x2() -> None:
    chosen, _calls = _decide(gpu_count=2, kaggle_scenario="ok")
    assert chosen.candidate_id == "kaggle:T4x2"
    assert chosen.resource is not None
    assert chosen.resource.gpu_count == 2
    assert [item.candidate_id for item in chosen.evaluations] == ["kaggle:T4x2"]

    blocked, _calls = _decide(gpu_count=2, kaggle_scenario="quota_insufficient")
    assert blocked.provider is None
    assert "exhausted" in _joined(_one(blocked, "kaggle:T4x2").reasons)
    colab = [item for item in blocked.evaluations if item.provider is ProviderName.colab]
    assert colab
    for evaluation in colab:
        assert evaluation.accepted is False
        assert "1 GPU(s), requested 2" in evaluation.reasons


def test_unknown_kaggle_quota_warn_selects_with_warning() -> None:
    unknown = QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.gpu_hours,
        is_fixture=True,
        source="mock-fixture",
        message="TEST FIXTURE: remaining unknown",
    )
    decision, calls = _decide(kaggle_quota_info=unknown, timeout=3600)
    assert decision.candidate_id == "kaggle:T4x2"
    assert calls == [ProviderName.kaggle]
    warning = "kaggle quota unknown; Kaggle may queue or reject the job"
    assert warning in decision.warnings
    assert warning in _one(decision, "kaggle:T4x2").reasons


def test_unknown_kaggle_quota_skip_rejects() -> None:
    unknown = QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.gpu_hours,
        is_fixture=True,
        source="mock-fixture",
        message="TEST FIXTURE: remaining unknown",
    )
    decision, _calls = _decide(
        kaggle_quota_info=unknown,
        colab_scenario="ai_pro",
        timeout=3600,
        settings=SchedulerSettings(unknown_kaggle_quota="skip"),
    )
    assert decision.candidate_id == "colab:T4"
    assert decision.fallback_used is True
    reason = "kaggle quota unknown; skipped by unknown_kaggle_quota=skip"
    assert reason in _one(decision, "kaggle:T4x2").reasons
    assert _one(decision, "kaggle:T4x2").accepted is False


def test_unknown_colab_balance_is_skipped() -> None:
    decision, _calls = _decide(
        provider=ProviderName.colab,
        colab_quota_info=colab_quota("free"),
        colab_resources_scenario="ai_pro",
        timeout=3600,
    )
    assert decision.provider is None
    assert "no backend satisfies the request" in decision.reasons
    text = _joined(_one(decision, "colab:T4").reasons)
    assert "colab CCU balance unknown" in text
    assert "never assume unlimited" in text
    assert colab_quota("free").remaining is None


def test_quota_unit_mismatch_is_treated_as_unknown() -> None:
    mismatched = QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.compute_units,
        total=0,
        used=0,
        remaining=0,
        is_fixture=True,
        source="mock-fixture",
        message="TEST FIXTURE: compute units are not GPU-hours",
    )
    decision, _calls = _decide(kaggle_quota_info=mismatched, timeout=100 * 3600)
    assert decision.candidate_id == "kaggle:T4x2"
    assert "exhausted" not in _joined(_one(decision, "kaggle:T4x2").reasons)
    assert "GPU-hours" not in _joined(_one(decision, "kaggle:T4x2").reasons)
    assert "kaggle quota unknown" in _joined(decision.warnings)
    quota = _one(decision, "kaggle:T4x2").quota
    assert quota is not None
    assert quota["unit"] == "compute_units"
    assert quota["remaining"] == 0


def test_missing_timeout_refuses_colab() -> None:
    decision, _calls = _decide(
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
        timeout=None,
    )
    assert decision.provider is None
    assert "timeout_seconds is required" in _joined(decision.reasons)
    assert "no backend satisfies the request" in decision.reasons
    assert _one(decision, "colab:T4").accepted is False
    assert _one(decision, "colab:L4").accepted is False


def test_kaggle_without_timeout_is_selected_with_warning() -> None:
    decision, _calls = _decide(kaggle_scenario="ok", timeout=None)
    assert decision.candidate_id == "kaggle:T4x2"
    assert decision.estimated_ccu is None
    warning = "no timeout_seconds; Kaggle GPU-hour use is not bounded by this request"
    assert warning in decision.warnings


def test_budget_guard_max_ccu_per_job() -> None:
    decision, _calls = _decide(gpu_type=GpuType.L4, timeout=20 * 3600, colab_scenario="ai_pro")
    assert decision.provider is None
    text = _joined(_one(decision, "colab:L4").reasons)
    assert "34.2" in text
    assert "exceeds max_ccu_per_job" in text
    assert "20" in text


def test_budget_guard_reserve() -> None:
    short, _calls = _decide(
        kaggle_scenario="quota_insufficient",
        colab_scenario="ccu_insufficient",
        timeout=3600,
    )
    assert short.provider is None
    assert "min_colab_ccu_reserve" in _joined(_one(short, "colab:T4").reasons)

    tight, _calls = _decide(
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
        timeout=3600,
        settings=SchedulerSettings(min_colab_ccu_reserve=199.5),
    )
    assert tight.provider is None
    text = _joined(_one(tight, "colab:T4").reasons)
    assert "min_colab_ccu_reserve" in text
    assert "199.5" in text
    assert "198.81" in text


def test_premium_blocked_by_default() -> None:
    decision, calls = _decide(
        gpu_type=GpuType.A100,
        premium_confirmed=True,
        timeout=3600,
        colab_scenario="ai_pro",
    )
    assert decision.provider is None
    assert calls == []
    reason = "premium GPU blocked: scheduler.allow_premium_gpu is false"
    assert reason in _one(decision, "colab:A100-40GB").reasons
    assert "allow_premium_gpu is false" in _joined(decision.reasons)


def test_premium_requires_confirmation() -> None:
    decision, _calls = _decide(
        gpu_type=GpuType.A100,
        premium_confirmed=False,
        timeout=3600,
        colab_scenario="ai_pro",
        settings=SchedulerSettings(allow_premium_gpu=True),
    )
    assert decision.provider is None
    text = _joined(_one(decision, "colab:A100-40GB").reasons)
    assert "confirmation" in text
    assert "allow_premium" in text


def test_premium_allowed_with_config_and_confirmation() -> None:
    decision, _calls = _decide(
        gpu_type=GpuType.A100,
        premium_confirmed=True,
        timeout=3600,
        colab_scenario="ai_pro",
        settings=SchedulerSettings(allow_premium_gpu=True),
    )
    assert decision.candidate_id == "colab:A100-40GB"
    assert decision.provider is ProviderName.colab
    assert decision.gpu_type is GpuType.A100
    assert decision.estimated_ccu == 5.4
    assert decision.resource is not None
    assert decision.resource.gpu_type is GpuType.A100
    assert decision.resource.premium is True
    assert any("unverified estimate" in warning for warning in decision.warnings)
    assert any("colab:A100-40GB" in warning for warning in decision.warnings)


def test_a100_80gb_variant_not_selectable() -> None:
    decision, calls = _decide(
        min_memory=60,
        premium_confirmed=True,
        timeout=3600,
        colab_scenario="ai_pro",
        settings=SchedulerSettings(allow_premium_gpu=True),
    )
    assert decision.provider is None
    assert calls == []
    assert "GB per GPU" in _joined(_one(decision, "colab:A100-40GB").reasons)
    assert "variant not selectable" in _joined(_one(decision, "colab:A100-80GB").reasons)
    assert "no backend satisfies the request" in decision.reasons


def test_premium_unavailable_fixture() -> None:
    offered = {item.gpu_type for item in colab_resources("premium_unavailable")}
    assert offered == {GpuType.T4, GpuType.L4, GpuType.NONE}
    decision, _calls = _decide(
        gpu_type=GpuType.A100,
        premium_confirmed=True,
        timeout=3600,
        colab_scenario="ai_pro",
        colab_resources_scenario="premium_unavailable",
        settings=SchedulerSettings(allow_premium_gpu=True),
    )
    assert decision.provider is None
    assert "does not currently offer" in _joined(_one(decision, "colab:A100-40GB").reasons)
    assert "A100x1" in _joined(_one(decision, "colab:A100-40GB").reasons)


def test_h100_has_no_rate_and_is_refused() -> None:
    decision, _calls = _decide(
        provider=ProviderName.colab,
        gpu_type=GpuType.H100,
        premium_confirmed=True,
        timeout=3600,
        colab_scenario="ai_pro",
        settings=SchedulerSettings(allow_premium_gpu=True),
    )
    assert decision.provider is None
    assert "no CCU rate" in _joined(_one(decision, "colab:H100").reasons)
    assert _one(decision, "colab:H100").accepted is False


def test_auth_failure_rejects_provider() -> None:
    failing = (
        AuthState.unauthenticated,
        AuthState.expired,
        AuthState.invalid,
        AuthState.error,
    )
    for state in failing:
        decision, _calls = _decide(kaggle_auth=state, colab_scenario="ai_pro", timeout=3600)
        assert decision.provider is ProviderName.colab
        assert f"kaggle auth is {state.value}" in _one(decision, "kaggle:T4x2").reasons

    errored, _calls = _decide(kaggle_error="auth_missing: token not found")
    errored_text = _joined(_one(errored, "kaggle:T4x2").reasons)
    assert "kaggle unavailable: auth_missing: token not found" in errored_text
    assert errored.provider is ProviderName.colab

    disabled, _calls = _decide(kaggle_enabled=False)
    assert "kaggle is disabled" in _one(disabled, "kaggle:T4x2").reasons
    assert disabled.provider is ProviderName.colab


def test_snapshot_called_lazily_once_per_provider() -> None:
    chosen, calls = _decide(kaggle_scenario="ok", timeout=3600)
    assert chosen.candidate_id == "kaggle:T4x2"
    assert calls == [ProviderName.kaggle]

    blocked, again = _decide(
        provider=ProviderName.kaggle,
        kaggle_scenario="quota_insufficient",
        timeout=3600,
    )
    assert blocked.provider is None
    assert again == [ProviderName.kaggle]
    assert _one(blocked, "kaggle:T4x2").accepted is False
    assert _one(blocked, "kaggle:P100").accepted is False


def test_decision_reasons_cover_every_rejected_candidate() -> None:
    decision, _calls = _decide(min_memory=24, timeout=3600)
    assert decision.candidate_id == "colab:L4"
    evaluated = [item.candidate_id for item in decision.evaluations]
    assert "colab:A100-40GB" not in evaluated
    blob = "\n".join(decision.reasons)
    for candidate_id in evaluated:
        assert candidate_id in blob
    for evaluation in decision.evaluations:
        assert evaluation.reasons
        if not evaluation.accepted:
            assert f"{evaluation.candidate_id}: rejected:" in blob
    payload = decision.to_dict()
    json.dumps(payload)
    assert set(payload) >= {"provider", "resource", "reasons", "warnings"}
    assert payload["provider"] == "colab"
    assert isinstance(payload["resource"], dict)
    assert payload["resource"]["gpu_type"] == "L4"
    assert isinstance(payload["reasons"], list)
    assert isinstance(payload["warnings"], list)


def test_fixture_reasons_are_labelled() -> None:
    fixture, _calls = _decide(kaggle_scenario="ok", timeout=3600)
    assert fixture.is_fixture is True
    assert "TEST FIXTURE" in _joined(_one(fixture, "kaggle:T4x2").reasons)
    assert "TEST FIXTURE" in _joined(fixture.reasons)

    live = QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.gpu_hours,
        total=30,
        used=12.5,
        remaining=17.5,
        is_fixture=False,
        source="account",
        message="17.5 GPU-hours remaining",
    )
    plain, _calls = _decide(kaggle_quota_info=live, timeout=3600)
    assert plain.is_fixture is False
    assert "TEST FIXTURE" not in _joined(plain.reasons)
    assert "17.5" in _joined(_one(plain, "kaggle:T4x2").reasons)


def test_fake_provider_pin_is_rejected() -> None:
    with pytest.raises(ValueError):
        ScheduleRequest(provider=ProviderName.fake)

    request = ScheduleRequest.model_construct(provider=ProviderName.fake)

    def snapshots(_provider: ProviderName) -> ProviderSnapshot:
        raise AssertionError("snapshot should not be called")

    with pytest.raises(ValueError, match="fake"):
        select_backend(request, SchedulerSettings(), snapshots)


def test_exhausted_fixture_reason_is_labelled() -> None:
    decision, _calls = _decide(kaggle_scenario="quota_insufficient", timeout=3600)
    text = _joined(_one(decision, "kaggle:T4x2").reasons)
    assert text.startswith("TEST FIXTURE:") or "TEST FIXTURE:" in text
    assert "kaggle GPU quota exhausted (0 GPU-hours remaining)" in text

def test_explicit_tpu_selects_colab_tpu_candidate() -> None:
    decision, calls = _decide(
        gpu_type=GpuType.TPU_V5E1,
        gpu_count=1,
        timeout=3600,
        premium_confirmed=True,
        provider=ProviderName.colab,
        settings=SchedulerSettings(allow_premium_gpu=True),
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
    )
    assert decision.provider is ProviderName.colab
    assert decision.candidate_id == "colab:TPU-v5e1"
    assert decision.gpu_type is GpuType.TPU_V5E1
    assert decision.estimated_ccu == 4.5
    assert ProviderName.colab in calls


def test_high_mem_rejects_l4_and_uses_t4_rate() -> None:
    blocked, _ = _decide(
        gpu_type=GpuType.L4,
        gpu_count=1,
        timeout=3600,
        high_mem=True,
        provider=ProviderName.colab,
        colab_scenario="ai_pro",
        kaggle_scenario="quota_insufficient",
    )
    assert blocked.provider is None
    assert any("high_mem" in reason for reason in blocked.reasons)

    decision, _ = _decide(
        gpu_type=GpuType.T4,
        gpu_count=1,
        timeout=3600,
        high_mem=True,
        provider=ProviderName.colab,
        colab_scenario="ai_pro",
        kaggle_scenario="quota_insufficient",
    )
    assert decision.candidate_id == "colab:T4"
    assert decision.estimated_ccu == 1.45


def test_tpu_not_auto_selected_without_explicit_gpu_type() -> None:
    decision, _ = _decide(
        gpu_type=None,
        gpu_count=1,
        timeout=3600,
        premium_confirmed=True,
        settings=SchedulerSettings(allow_premium_gpu=True),
        kaggle_scenario="quota_insufficient",
        colab_scenario="ai_pro",
    )
    assert decision.candidate_id in {"colab:T4", "colab:L4", "colab:A100-40GB"}
    assert decision.candidate_id is not None
    assert "TPU" not in decision.candidate_id



def test_colab_t4_accepted_at_zero_ccu_when_eligible() -> None:
    """CCU=0 + T4 in eligible_gpus accepts; membership must not be required."""
    zero = QuotaInfo(
        provider=ProviderName.colab,
        unit=QuotaUnit.compute_units,
        total=0.0,
        used=0.0,
        remaining=0.0,
        is_fixture=True,
        source="mock-fixture",
        message="TEST FIXTURE: 0 CCU, eligible T4",
        details={"membership": "free_pro"},
        eligible_gpus=["T4"],
    )
    decision, _calls = _decide(
        provider=ProviderName.colab,
        gpu_type=GpuType.T4,
        colab_quota_info=zero,
        colab_resources_scenario="ai_pro",
        timeout=3600,
    )
    assert decision.candidate_id == "colab:T4"
    assert decision.provider is ProviderName.colab
    text = _joined(_one(decision, "colab:T4").reasons)
    assert "account-eligible" in text
    assert "min_colab_ccu_reserve" not in text


def test_colab_l4_rejected_at_zero_ccu_when_not_eligible() -> None:
    zero = QuotaInfo(
        provider=ProviderName.colab,
        unit=QuotaUnit.compute_units,
        total=0.0,
        used=0.0,
        remaining=0.0,
        is_fixture=True,
        source="mock-fixture",
        message="TEST FIXTURE: 0 CCU, eligible T4 only",
        details={"membership": "paid_pro"},  # must not exempt L4
        eligible_gpus=["T4"],
    )
    decision, _calls = _decide(
        provider=ProviderName.colab,
        gpu_type=GpuType.L4,
        colab_quota_info=zero,
        colab_resources_scenario="ai_pro",
        timeout=3600,
    )
    assert decision.provider is None
    text = _joined(_one(decision, "colab:L4").reasons)
    assert "not in eligible_gpus" in text
    assert _one(decision, "colab:L4").accepted is False


def test_colab_reserve_still_enforced_when_remaining_positive() -> None:
    """remaining>0 keeps paid CCU path even if L4 is eligible."""
    short = QuotaInfo(
        provider=ProviderName.colab,
        unit=QuotaUnit.compute_units,
        total=2.0,
        used=0.0,
        remaining=2.0,
        is_fixture=True,
        source="mock-fixture",
        message="TEST FIXTURE: 2 CCU, eligible includes L4",
        details={"membership": "paid_pro"},
        eligible_gpus=["T4", "L4", "A100"],
    )
    decision, _calls = _decide(
        provider=ProviderName.colab,
        gpu_type=GpuType.L4,
        colab_quota_info=short,
        colab_resources_scenario="ai_pro",
        timeout=3600,
        settings=SchedulerSettings(min_colab_ccu_reserve=50),
    )
    assert decision.provider is None
    text = _joined(_one(decision, "colab:L4").reasons)
    assert "min_colab_ccu_reserve" in text


def test_zero_ccu_helper_ignores_membership_field() -> None:
    import inspect

    from link2kaggle_colab.scheduler import selector

    source = inspect.getsource(selector._zero_ccu_accelerator_allowed)
    assert "membership" not in source
    assert "free_pro" not in source
    assert "paid_pro" not in source
    assert "details[" not in source
    budget = inspect.getsource(selector._colab_budget)
    assert "membership" not in budget
    assert "free_pro" not in budget
    assert "details[" not in budget
    # Behaviour: paid_pro label cannot exempt a non-eligible GPU.
    allowed, _detail, _warn = selector._zero_ccu_accelerator_allowed(
        gpu_type="L4",
        candidate_id="colab:L4",
        premium=False,
        eligible_gpus=["T4"],
    )
    assert allowed is False
