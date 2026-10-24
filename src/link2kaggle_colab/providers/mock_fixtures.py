"""TEST FIXTURE catalogs for mock providers.

Nothing in this module is a live Kaggle quota, Colab balance, or membership.
Do not present these numbers as an account.
"""

from __future__ import annotations

from link2kaggle_colab.core.models import GpuType, ProviderName, QuotaInfo, QuotaUnit, ResourceSpec

# Fixture values for tests; real rates must come from `colab usage`/official sources.
COLAB_CCU_RATES_PER_HOUR_FIXTURE: dict[str, float] = {
    "T4": 1.8,
    "L4": 4.8,
    "A100": 11.8,
    "H100": 18.0,
    # High-mem shape and TPU rates are TEST FIXTURE estimates only.
    "T4+high_mem": 2.2,
    "A100+high_mem": 14.0,
    "TPU_V5E1": 6.0,
    "TPU_V6E1": 8.0,
}

_FIXTURE_SOURCE = "mock-fixture"


def _cpu(provider_resource_id: str = "cpu", *, supports_high_mem: bool = False) -> ResourceSpec:
    return ResourceSpec(
        gpu_type=GpuType.NONE,
        gpu_count=0,
        gpu_memory_gb_per_gpu=None,
        provider_resource_id=provider_resource_id,
        premium=False,
        supports_high_mem=supports_high_mem,
        label="CPU" + (" (--high-mem ok)" if supports_high_mem else ""),
    )


def kaggle_resources() -> list[ResourceSpec]:
    """Kaggle accelerators used by the mock. T4 is two 16 GB GPUs, not 32 GB."""
    return [
        ResourceSpec(
            gpu_type=GpuType.T4,
            gpu_count=2,
            gpu_memory_gb_per_gpu=16,
            provider_resource_id="NvidiaTeslaT4",
            premium=False,
            label="NvidiaTeslaT4 (T4 x2, 16 GB per GPU)",
        ),
        ResourceSpec(
            gpu_type=GpuType.P100,
            gpu_count=1,
            gpu_memory_gb_per_gpu=16,
            provider_resource_id="NvidiaTeslaP100",
            premium=False,
            label="NvidiaTeslaP100 (P100, 16 GB per GPU)",
        ),
        _cpu(),
    ]


def colab_resources(scenario: str) -> list[ResourceSpec]:
    """Colab accelerators. ``free`` is T4 only. A100 memory is unknown.

    ``premium_unavailable`` is a TEST FIXTURE that offers only T4, L4, and CPU
    (no A100, H100, or TPU). Paid scenarios include TPU v5e1/v6e1.
    """
    t4 = ResourceSpec(
        gpu_type=GpuType.T4,
        gpu_count=1,
        gpu_memory_gb_per_gpu=16,
        provider_resource_id="T4",
        premium=False,
        supports_high_mem=True,
        label="T4 (16 GB nominal per GPU; about 15 GB usable; --high-mem ok)",
    )
    l4 = ResourceSpec(
        gpu_type=GpuType.L4,
        gpu_count=1,
        gpu_memory_gb_per_gpu=24,
        provider_resource_id="L4",
        premium=False,
        label="L4 (24 GB per GPU)",
    )
    tpu_v5 = ResourceSpec(
        gpu_type=GpuType.TPU_V5E1,
        gpu_count=1,
        gpu_memory_gb_per_gpu=None,
        provider_resource_id="TPU_V5E1",
        premium=True,
        supports_high_mem=False,
        label="TPU v5e1 (colab new --tpu v5e1; high-mem N/A)",
    )
    tpu_v6 = ResourceSpec(
        gpu_type=GpuType.TPU_V6E1,
        gpu_count=1,
        gpu_memory_gb_per_gpu=None,
        provider_resource_id="TPU_V6E1",
        premium=True,
        supports_high_mem=False,
        label="TPU v6e1 (colab new --tpu v6e1; high-mem N/A)",
    )
    a100 = ResourceSpec(
        gpu_type=GpuType.A100,
        gpu_count=1,
        gpu_memory_gb_per_gpu=None,
        provider_resource_id="A100",
        premium=True,
        supports_high_mem=True,
        label="A100 (per-GPU memory unknown: 40 GB or 80 GB; --high-mem ok)",
    )
    h100 = ResourceSpec(
        gpu_type=GpuType.H100,
        gpu_count=1,
        gpu_memory_gb_per_gpu=80,
        provider_resource_id="H100",
        premium=True,
        supports_high_mem=True,
        label="H100 (80 GB per GPU; --high-mem ok)",
    )
    cpu = _cpu(supports_high_mem=True)
    if scenario in {"free", "free_pro_zero"}:
        return [t4, cpu]
    if scenario == "premium_unavailable":
        return [t4, l4, cpu]
    return [t4, l4, a100, h100, tpu_v5, tpu_v6, cpu]


def _hours(provider: ProviderName, total: float, used: float, message: str) -> QuotaInfo:
    return QuotaInfo(
        provider=provider,
        unit=QuotaUnit.gpu_hours,
        total=total,
        used=used,
        remaining=total - used,
        is_fixture=True,
        source=_FIXTURE_SOURCE,
        message=message,
    )


def kaggle_quota(scenario: str) -> QuotaInfo:
    """Weekly GPU-hour fixtures. ``quota_45h`` is 45h total; insufficient is 30/30."""
    if scenario == "quota_45h":
        return _hours(
            ProviderName.kaggle,
            45.0,
            0.0,
            "TEST FIXTURE: 45h total GPU quota. Not a live Kaggle account.",
        )
    if scenario == "quota_insufficient":
        return _hours(
            ProviderName.kaggle,
            30.0,
            30.0,
            "TEST FIXTURE: 30h total, 30h used. Not a live Kaggle account.",
        )
    return _hours(
        ProviderName.kaggle,
        30.0,
        12.5,
        "TEST FIXTURE: 30h total / 12.5h used. Not a live Kaggle account.",
    )


_COLAB_FREE_GPUS = ["T4"]
_COLAB_PAID_GPUS = ["T4", "L4", "A100", "G4", "TPU_V5E1", "TPU_V6E1"]


def colab_quota(scenario: str) -> QuotaInfo:
    """Compute-unit fixtures. ``free`` leaves remaining unknown."""
    if scenario == "free":
        return QuotaInfo(
            provider=ProviderName.colab,
            unit=QuotaUnit.compute_units,
            is_fixture=True,
            source=_FIXTURE_SOURCE,
            message=(
                "TEST FIXTURE: free account, compute-unit balance unknown. "
                "Not a live Colab account."
            ),
            eligible_gpus=list(_COLAB_FREE_GPUS),
        )
    if scenario == "ai_pro":
        return QuotaInfo(
            provider=ProviderName.colab,
            unit=QuotaUnit.compute_units,
            total=200.0,
            used=0.0,
            remaining=200.0,
            is_fixture=True,
            source=_FIXTURE_SOURCE,
            message=(
                "TEST FIXTURE: Google AI Pro (MOCK) membership, 200 CCU balance. "
                "Not a live Colab account."
            ),
            eligible_gpus=list(_COLAB_PAID_GPUS),
        )
    if scenario == "free_pro_zero":
        # TEST FIXTURE: free Pro shape (0 CCU, T4-only eligible). free_pro ≠ paid.
        return QuotaInfo(
            provider=ProviderName.colab,
            unit=QuotaUnit.compute_units,
            total=0.0,
            used=0.0,
            remaining=0.0,
            is_fixture=True,
            source=_FIXTURE_SOURCE,
            message=(
                "TEST FIXTURE: free Pro (MOCK) 0 CCU, eligible T4 only. "
                "free_pro ≠ paid. Not a live Colab account."
            ),
            details={"membership": "free_pro"},
            eligible_gpus=list(_COLAB_FREE_GPUS),
        )
    if scenario == "ccu_insufficient":
        return QuotaInfo(
            provider=ProviderName.colab,
            unit=QuotaUnit.compute_units,
            total=2.0,
            used=0.0,
            remaining=2.0,
            is_fixture=True,
            source=_FIXTURE_SOURCE,
            message="TEST FIXTURE: 2 CCU remaining. Not a live Colab account.",
            eligible_gpus=list(_COLAB_PAID_GPUS),
        )
    return QuotaInfo(
        provider=ProviderName.colab,
        unit=QuotaUnit.compute_units,
        total=100.0,
        used=0.0,
        remaining=100.0,
        is_fixture=True,
        source=_FIXTURE_SOURCE,
        message="TEST FIXTURE: placeholder CCU balance. Not a live Colab account.",
        eligible_gpus=list(_COLAB_PAID_GPUS),
    )


MOCK_RESULT_TEXT = "MOCK TEST FIXTURE\nstatus=ok\n"


def mock_log(provider: str, job_id: str, state: str, gpu: str | None = None) -> str:
    """Log text that is obviously a fixture."""
    lines = [
        "[MOCK] TEST FIXTURE",
        f"[MOCK] provider={provider} job={job_id} state={state}",
    ]
    if gpu:
        lines.append(f"[MOCK] gpu={gpu}")
    lines.append("[MOCK] this output is not from a real account")
    return "\n".join(lines) + "\n"
