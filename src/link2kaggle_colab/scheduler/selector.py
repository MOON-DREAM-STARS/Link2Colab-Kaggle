"""Kaggle-first placement.

``select_backend`` does not submit and does no I/O except calling ``snapshots``.
Compute units and GPU-hours are never compared.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from link2kaggle_colab.core.config import SchedulerSettings
from link2kaggle_colab.core.models import (
    TPU_TYPES,
    AuthState,
    GpuType,
    ProviderName,
    QuotaInfo,
    QuotaUnit,
    ResourceSpec,
)
from link2kaggle_colab.quota.cost import CostEstimate, estimate_ccu, kaggle_hours_needed
from link2kaggle_colab.scheduler.candidates import CANDIDATES, Candidate
from link2kaggle_colab.scheduler.decision import (
    CandidateEvaluation,
    SchedulerDecision,
    ScheduleRequest,
)
from link2kaggle_colab.scheduler.policy import premium_gate
from link2kaggle_colab.scheduler.snapshot import ProviderSnapshot, SnapshotSource

_FAILING_AUTH = frozenset(
    {
        AuthState.unauthenticated,
        AuthState.expired,
        AuthState.invalid,
        AuthState.error,
    }
)
_FIXTURE_PREFIX = "TEST FIXTURE: "
_KEEP_RUNTIME_REJECT = (
    "colab.keep_runtime is true; runtime is not stopped after the job, "
    "so CCU spend is not bounded by timeout_seconds; pin provider colab to accept"
)
_KEEP_RUNTIME_WARN = (
    "colab.keep_runtime is true; CCU spend is not bounded by timeout_seconds"
)


class _Outcome(NamedTuple):
    evaluation: CandidateEvaluation
    warnings: list[str]
    resource: ResourceSpec | None
    cost: CostEstimate | None


def select_backend(
    request: ScheduleRequest,
    settings: SchedulerSettings,
    snapshots: SnapshotSource,
) -> SchedulerDecision:
    """Pick the first candidate that passes static and quota checks.

    ``snapshots`` is called at most once per provider, and only after a
    candidate of that provider passes the static checks. Later candidates are
    not evaluated once one is accepted.
    """
    if request.provider is ProviderName.fake:
        raise ValueError("provider 'fake' cannot be scheduled")

    loaded: dict[ProviderName, ProviderSnapshot] = {}
    evaluations: list[CandidateEvaluation] = []
    warnings: list[str] = []
    selected: _Outcome | None = None
    selected_gpu: GpuType | None = None

    for candidate in _candidate_order(request, settings):
        static = _static_reasons(candidate, request, settings)
        if static:
            evaluations.append(_evaluation(candidate, accepted=False, reasons=static))
            continue
        snapshot = _load_snapshot(candidate.provider, snapshots, loaded)
        outcome = _evaluate_dynamic(candidate, request, settings, snapshot)
        evaluations.append(outcome.evaluation)
        warnings.extend(outcome.warnings)
        if outcome.evaluation.accepted:
            selected = outcome
            selected_gpu = candidate.gpu_type
            break

    summaries = [_summary(item) for item in evaluations]
    fixture = any(bool(item.quota and item.quota.get("is_fixture")) for item in evaluations)
    if selected is None:
        summaries.append("no backend satisfies the request")
        return SchedulerDecision(
            strategy=settings.strategy,
            provider=None,
            candidate_id=None,
            gpu_type=None,
            resource=None,
            fallback_used=False,
            estimated_ccu=None,
            reasons=summaries,
            warnings=warnings,
            evaluations=evaluations,
            is_fixture=fixture,
        )

    estimated = None if selected.cost is None else selected.cost.estimated_ccu
    return SchedulerDecision(
        strategy=settings.strategy,
        provider=selected.evaluation.provider,
        candidate_id=selected.evaluation.candidate_id,
        gpu_type=selected_gpu,
        resource=selected.resource,
        fallback_used=_fallback_used(evaluations, selected.evaluation.provider),
        estimated_ccu=estimated,
        reasons=summaries,
        warnings=warnings,
        evaluations=evaluations,
        is_fixture=fixture,
    )


def _candidate_order(request: ScheduleRequest, settings: SchedulerSettings) -> list[Candidate]:
    """Cost order, then remaining catalog entries for a pin or explicit type."""
    pinned = request.provider
    ordered: list[Candidate] = []
    seen: set[str] = set()
    for candidate_id in settings.cost_priority:
        candidate = CANDIDATES[candidate_id]
        if pinned is not None and candidate.provider is not pinned:
            continue
        ordered.append(candidate)
        seen.add(candidate.id)
    if pinned is not None:
        for candidate in CANDIDATES.values():
            if candidate.provider is pinned and candidate.id not in seen:
                ordered.append(candidate)
                seen.add(candidate.id)
        return ordered
    # Auto: also consider catalog entries matching an explicit accelerator
    # (TPU / H100) that sit outside default cost_priority.
    if request.gpu_type is not None:
        for candidate in CANDIDATES.values():
            if candidate.gpu_type is request.gpu_type and candidate.id not in seen:
                ordered.append(candidate)
                seen.add(candidate.id)
    return ordered


def _static_reasons(
    candidate: Candidate,
    request: ScheduleRequest,
    settings: SchedulerSettings,
) -> list[str]:
    """Checks that do not need a provider snapshot. All failures are kept."""
    reasons: list[str] = []
    if (
        request.provider is None
        and not settings.allow_colab_fallback
        and candidate.provider is ProviderName.colab
    ):
        reasons.append("allow_colab_fallback is false")
    if request.gpu_type is not None and request.gpu_type is not candidate.gpu_type:
        requested = request.gpu_type.value
        reasons.append(
            f"gpu_type {candidate.gpu_type.value} does not match requested {requested}"
        )
    if candidate.gpu_type in TPU_TYPES and request.gpu_type is None:
        reasons.append("TPU must be requested explicitly (gpu_type)")
    if candidate.gpu_count < request.gpu_count:
        reasons.append(f"{candidate.gpu_count} GPU(s), requested {request.gpu_count}")
    if not candidate.variant_selectable:
        reasons.append(f"variant not selectable via the official CLI: {candidate.note}")
    if request.high_mem and not candidate.supports_high_mem:
        reasons.append(
            f"{candidate.id} does not support high_mem "
            "(Colab --high-mem ignored for L4/TPU)"
        )
    minimum = request.min_gpu_memory_gb_per_gpu
    effective = candidate.gpu_memory_gb_per_gpu
    if minimum is not None and effective is not None and minimum > effective:
        reasons.append(_memory_reason(effective, minimum, candidate.gpu_count))
    if candidate.premium:
        blocked = premium_gate(settings, request.premium_confirmed)
        if blocked is not None:
            reasons.append(blocked)
    return reasons


def _evaluate_dynamic(
    candidate: Candidate,
    request: ScheduleRequest,
    settings: SchedulerSettings,
    snapshot: ProviderSnapshot,
) -> _Outcome:
    quota = _quota_view(snapshot)
    availability = _availability_reasons(candidate, snapshot)
    if availability:
        return _Outcome(
            _evaluation(candidate, accepted=False, reasons=availability, quota=quota),
            [],
            None,
            None,
        )

    offer_reasons, resource = _offering(candidate, request, settings, snapshot)
    if offer_reasons:
        return _Outcome(
            _evaluation(candidate, accepted=False, reasons=offer_reasons, quota=quota),
            [],
            None,
            None,
        )

    # Kaggle spends GPU-hours. Colab spends CCU. The branches do not share a balance.
    if candidate.paid:
        reasons, notes, warnings, cost = _colab_budget(candidate, request, settings, snapshot)
    else:
        reasons, notes, warnings = _kaggle_quota(request, settings, snapshot)
        cost = None
    accepted = not reasons
    return _Outcome(
        _evaluation(
            candidate,
            accepted=accepted,
            reasons=notes if accepted else reasons,
            cost=cost,
            quota=quota,
        ),
        warnings,
        resource if accepted else None,
        cost,
    )


def _availability_reasons(candidate: Candidate, snapshot: ProviderSnapshot) -> list[str]:
    provider = candidate.provider.value
    reasons: list[str] = []
    if not snapshot.enabled:
        reasons.append(f"{provider} is disabled")
    if snapshot.error:
        reasons.append(f"{provider} unavailable: {snapshot.error}")
    auth = snapshot.auth
    if auth is not None and auth.state in _FAILING_AUTH:
        reasons.append(f"{provider} auth is {auth.state.value}")
    return reasons


def _offering(
    candidate: Candidate,
    request: ScheduleRequest,
    settings: SchedulerSettings,
    snapshot: ProviderSnapshot,
) -> tuple[list[str], ResourceSpec | None]:
    match = _find_resource(snapshot.resources, candidate)
    if match is None:
        provider = candidate.provider.value
        offered = f"{candidate.gpu_type.value}x{candidate.gpu_count}"
        return [f"{provider} does not currently offer {offered}"], None

    reasons: list[str] = []
    minimum = request.min_gpu_memory_gb_per_gpu
    reported = match.gpu_memory_gb_per_gpu
    if minimum is not None and reported is not None and minimum > reported:
        reasons.append(_memory_reason(reported, minimum, match.gpu_count))
    if request.high_mem and not match.supports_high_mem:
        reasons.append(
            f"{match.provider_resource_id or match.gpu_type.value} "
            "does not support high_mem"
        )
    if match.premium:
        blocked = premium_gate(settings, request.premium_confirmed)
        if blocked is not None:
            reasons.append(blocked)
    return reasons, match


def _find_resource(
    resources: list[ResourceSpec] | None,
    candidate: Candidate,
) -> ResourceSpec | None:
    if not resources:
        return None
    for resource in resources:
        if resource.cpu_only:
            continue
        if resource.gpu_type is candidate.gpu_type and resource.gpu_count == candidate.gpu_count:
            return resource
    return None


def _kaggle_quota(
    request: ScheduleRequest,
    settings: SchedulerSettings,
    snapshot: ProviderSnapshot,
) -> tuple[list[str], list[str], list[str]]:
    """Rejection reasons, acceptance notes, and warnings. GPU-hours only."""
    quota, remaining = _quota_remaining(snapshot)
    if (
        quota is None
        or remaining is None
        or not quota.known
        or quota.unit is not QuotaUnit.gpu_hours
    ):
        if settings.unknown_kaggle_quota == "skip":
            return ["kaggle quota unknown; skipped by unknown_kaggle_quota=skip"], [], []
        note = "kaggle quota unknown; Kaggle may queue or reject the job"
        return [], [note], [note]

    reasons: list[str] = []
    if remaining <= 0:
        reasons.append(
            _quota_text(quota.is_fixture, "kaggle GPU quota exhausted (0 GPU-hours remaining)")
        )
    needed = kaggle_hours_needed(request.timeout_seconds)
    if needed is not None and needed > remaining:
        reasons.append(
            _quota_text(
                quota.is_fixture,
                f"needs {_fmt(needed)} GPU-hours, {_fmt(remaining)} GPU-hours remaining",
            )
        )
    if reasons:
        return reasons, [], []

    notes = [_quota_text(quota.is_fixture, f"{_fmt(remaining)} GPU-hours remaining")]
    warnings: list[str] = []
    if request.timeout_seconds is None:
        warning = "no timeout_seconds; Kaggle GPU-hour use is not bounded by this request"
        notes.append(warning)
        warnings.append(warning)
    return [], notes, warnings


def _zero_ccu_accelerator_allowed(
    *,
    gpu_type: str,
    candidate_id: str,
    premium: bool,
    eligible_gpus: list[str] | None,
) -> tuple[bool, str, str | None]:
    """Decide whether a Colab accelerator may run when CCU remaining is 0.

    Gates solely on ``eligible_gpus`` from usage parse (or opt-in experimental ccu-info).
    Never consults account tier / free / pro labels.
    Returns ``(allowed, note_or_reason, warning_or_none)``.
    """
    if eligible_gpus is not None:
        if gpu_type in eligible_gpus:
            note = (
                f"{gpu_type} is account-eligible; CCU budget was not required "
                "(availability still not guaranteed)"
            )
            return True, note, None
        reason = (
            f"{gpu_type} is not in eligible_gpus while CCU remaining is 0"
        )
        return False, reason, None
    # Eligibility unpublished: only non-premium colab:T4.
    if candidate_id == "colab:T4" and not premium and gpu_type == "T4":
        warning = (
            "Colab eligible_gpus unpublished; allowing non-premium colab:T4 at 0 CCU"
        )
        return True, warning, warning
    reason = (
        "colab CCU remaining is 0 and eligible_gpus unpublished; "
        f"refusing {candidate_id}"
    )
    return False, reason, None


def _colab_budget(
    candidate: Candidate,
    request: ScheduleRequest,
    settings: SchedulerSettings,
    snapshot: ProviderSnapshot,
) -> tuple[list[str], list[str], list[str], CostEstimate]:
    """Rejection reasons, acceptance notes, warnings, and the CCU estimate.

    When remaining CCU is 0, accept only if the requested accelerator is listed
    in ``quota.eligible_gpus`` (or unpublished eligibility with colab:T4).
    Membership / free / pro labels are never consulted.
    When remaining > 0, keep the paid CCU path (reserve + max_ccu_per_job).
    """
    cost = estimate_ccu(
        candidate.id,
        settings.ccu_rates,
        request.timeout_seconds,
        high_mem=request.high_mem,
    )
    # Auto fallback must not leave a runtime running. A pinned Colab job can.
    if snapshot.keeps_runtime and request.provider is None:
        return [_KEEP_RUNTIME_REJECT], [], [], cost
    reasons: list[str] = []
    estimated = cost.estimated_ccu
    if estimated is None:
        reasons.append(cost.reason)
    elif estimated > settings.max_ccu_per_job:
        reasons.append(
            f"estimated {_fmt(estimated)} CCU exceeds max_ccu_per_job "
            f"{_fmt(settings.max_ccu_per_job)}"
        )

    quota, remaining = _quota_remaining(snapshot)
    usable = (
        quota is not None
        and remaining is not None
        and quota.known
        and quota.unit is QuotaUnit.compute_units
    )
    eligibility_notes: list[str] = []
    eligibility_warnings: list[str] = []
    skip_ccu_reserve = False
    if not usable or quota is None or remaining is None:
        reasons.append(
            "colab CCU balance unknown; paid backend skipped (never assume unlimited)"
        )
    elif remaining == 0:
        # Zero-CCU path: eligible_gpus only. Do not read account tier labels.
        allowed, detail, warn = _zero_ccu_accelerator_allowed(
            gpu_type=candidate.gpu_type.value,
            candidate_id=candidate.id,
            premium=candidate.premium,
            eligible_gpus=quota.eligible_gpus,
        )
        if allowed:
            skip_ccu_reserve = True
            eligibility_notes.append(_quota_text(quota.is_fixture, detail))
            if warn is not None:
                eligibility_warnings.append(warn)
        else:
            reasons.append(_quota_text(quota.is_fixture, detail))
    elif (
        not skip_ccu_reserve
        and estimated is not None
        and remaining - estimated < settings.min_colab_ccu_reserve
    ):
        # remaining > 0: always enforce reserve, even when GPU is eligible.
        left = round(remaining - estimated, 4)
        reasons.append(
            _quota_text(
                quota.is_fixture,
                (
                    f"estimated {_fmt(estimated)} CCU would leave {_fmt(left)} CCU, "
                    f"below min_colab_ccu_reserve {_fmt(settings.min_colab_ccu_reserve)}"
                ),
            )
        )

    notes: list[str] = []
    if not reasons:
        notes.extend(eligibility_notes)
        if estimated is not None and not skip_ccu_reserve:
            notes.append(f"estimated {_fmt(estimated)} CCU")
            if quota is not None and remaining is not None:
                notes.append(
                    _quota_text(quota.is_fixture, f"{_fmt(remaining)} CCU remaining")
                )
        elif estimated is not None and skip_ccu_reserve:
            notes.append(f"estimated {_fmt(estimated)} CCU (not budgeted; eligible path)")

    warnings: list[str] = []
    warnings.extend(eligibility_warnings)
    if estimated is not None and not cost.rate_verified and cost.rate_source:
        warnings.append(
            f"CCU rate for {candidate.id} is an unverified estimate ({cost.rate_source})"
        )
    if snapshot.keeps_runtime and request.provider is ProviderName.colab:
        warnings.append(_KEEP_RUNTIME_WARN)
    return reasons, notes, warnings, cost


def _quota_remaining(snapshot: ProviderSnapshot) -> tuple[QuotaInfo | None, float | None]:
    cached = snapshot.quota
    if cached is None:
        return None, None
    quota = cached.quota
    remaining = quota.remaining
    if remaining is None:
        return quota, None
    return quota, remaining


def _quota_view(snapshot: ProviderSnapshot) -> dict[str, Any] | None:
    cached = snapshot.quota
    if cached is None:
        return None
    quota = cached.quota
    return {
        "unit": quota.unit.value,
        "remaining": quota.remaining,
        "known": quota.known,
        "is_fixture": quota.is_fixture,
        "from_cache": cached.from_cache,
        "age_seconds": cached.age_seconds,
    }


def _load_snapshot(
    provider: ProviderName,
    snapshots: SnapshotSource,
    loaded: dict[ProviderName, ProviderSnapshot],
) -> ProviderSnapshot:
    cached = loaded.get(provider)
    if cached is None:
        cached = snapshots(provider)
        loaded[provider] = cached
    return cached


def _evaluation(
    candidate: Candidate,
    *,
    accepted: bool,
    reasons: list[str],
    cost: CostEstimate | None = None,
    quota: dict[str, Any] | None = None,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id=candidate.id,
        provider=candidate.provider,
        accepted=accepted,
        reasons=reasons,
        premium=candidate.premium,
        paid=candidate.paid,
        cost=cost,
        quota=quota,
    )


def _summary(evaluation: CandidateEvaluation) -> str:
    detail = "; ".join(evaluation.reasons)
    verb = "selected" if evaluation.accepted else "rejected"
    if detail:
        return f"{evaluation.candidate_id}: {verb}: {detail}"
    return f"{evaluation.candidate_id}: {verb}"


def _fallback_used(evaluations: list[CandidateEvaluation], provider: ProviderName) -> bool:
    if provider is not ProviderName.colab:
        return False
    return any(item.provider is ProviderName.kaggle and not item.accepted for item in evaluations)


def _memory_reason(effective: float, requested: float, count: int) -> str:
    return (
        f"{_fmt(effective)} GB per GPU is below requested {_fmt(requested)} GB per GPU; "
        f"memory is never summed across {count} GPUs"
    )


def _quota_text(is_fixture: bool, text: str) -> str:
    if is_fixture:
        return f"{_FIXTURE_PREFIX}{text}"
    return text


def _fmt(value: float) -> str:
    text = f"{round(value, 4):.4f}".rstrip("0").rstrip(".")
    return text or "0"
