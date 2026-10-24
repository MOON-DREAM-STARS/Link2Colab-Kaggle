"""Accelerator catalog for placement.

``DEFAULT_COST_PRIORITY`` is a COST preference (free Kaggle quota first, then
cheaper Colab GPUs), not a performance ranking. Positive CCU balances stay
CCU-metered (reserve + max_ccu_per_job). At CCU=0 the scheduler may accept
accelerators listed in ``quota.eligible_gpus`` (usage parse and/or opt-in ccu-info). Display
tier labels are not a scheduler input (free Pro is not paid Pro).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from link2kaggle_colab.core.errors import ConfigError
from link2kaggle_colab.core.models import GpuType, ProviderName


class Candidate(BaseModel):
    """One placeable accelerator variant.

    ``gpu_memory_gb_per_gpu`` is the nominal memory of a single GPU. It is
    never a sum across devices.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    provider: ProviderName
    gpu_type: GpuType
    gpu_count: int = Field(ge=1)
    gpu_memory_gb_per_gpu: float | None = Field(default=None, gt=0)
    paid: bool
    premium: bool
    # False when the official CLI cannot select this memory variant.
    variant_selectable: bool = True
    # Colab ``--high-mem`` shape. False for L4 and TPU (single shape).
    supports_high_mem: bool = False
    note: str = ""


CANDIDATES: dict[str, Candidate] = {
    "kaggle:T4x2": Candidate(
        id="kaggle:T4x2",
        provider=ProviderName.kaggle,
        gpu_type=GpuType.T4,
        gpu_count=2,
        gpu_memory_gb_per_gpu=16,
        paid=False,
        premium=False,
        note="two separate 16 GB GPUs, not 32 GB",
    ),
    "kaggle:P100": Candidate(
        id="kaggle:P100",
        provider=ProviderName.kaggle,
        gpu_type=GpuType.P100,
        gpu_count=1,
        gpu_memory_gb_per_gpu=16,
        paid=False,
        premium=False,
    ),
    "colab:T4": Candidate(
        id="colab:T4",
        provider=ProviderName.colab,
        gpu_type=GpuType.T4,
        gpu_count=1,
        gpu_memory_gb_per_gpu=16,
        paid=True,
        premium=False,
        supports_high_mem=True,
    ),
    "colab:L4": Candidate(
        id="colab:L4",
        provider=ProviderName.colab,
        gpu_type=GpuType.L4,
        gpu_count=1,
        gpu_memory_gb_per_gpu=24,
        paid=True,
        premium=False,
    ),
    "colab:A100-40GB": Candidate(
        id="colab:A100-40GB",
        provider=ProviderName.colab,
        gpu_type=GpuType.A100,
        gpu_count=1,
        gpu_memory_gb_per_gpu=40,
        paid=True,
        premium=True,
        supports_high_mem=True,
    ),
    "colab:A100-80GB": Candidate(
        id="colab:A100-80GB",
        provider=ProviderName.colab,
        gpu_type=GpuType.A100,
        gpu_count=1,
        gpu_memory_gb_per_gpu=80,
        paid=True,
        premium=True,
        variant_selectable=False,
        supports_high_mem=True,
        note=(
            "official colab CLI has no 40GB/80GB A100 selector "
            "(see docs/api-verification.md); needs real verification"
        ),
    ),
    "colab:H100": Candidate(
        id="colab:H100",
        provider=ProviderName.colab,
        gpu_type=GpuType.H100,
        gpu_count=1,
        gpu_memory_gb_per_gpu=80,
        paid=True,
        premium=True,
        supports_high_mem=True,
    ),
    "colab:TPU-v5e1": Candidate(
        id="colab:TPU-v5e1",
        provider=ProviderName.colab,
        gpu_type=GpuType.TPU_V5E1,
        gpu_count=1,
        gpu_memory_gb_per_gpu=None,
        paid=True,
        premium=True,
        supports_high_mem=False,
        note="colab new --tpu v5e1; Colab-only; not in default cost priority",
    ),
    "colab:TPU-v6e1": Candidate(
        id="colab:TPU-v6e1",
        provider=ProviderName.colab,
        gpu_type=GpuType.TPU_V6E1,
        gpu_count=1,
        gpu_memory_gb_per_gpu=None,
        paid=True,
        premium=True,
        supports_high_mem=False,
        note="colab new --tpu v6e1; Colab-only; not in default cost priority",
    ),
}

# Cost order, not speed. Kaggle P100, Colab H100, and Colab TPU are catalogued
# but omitted: P100 is not the default free choice; H100/TPU need an explicit
# request (and a configured CCU rate for H100). Pin provider=colab or set
# gpu_type to reach them.
DEFAULT_COST_PRIORITY: tuple[str, ...] = (
    "kaggle:T4x2",
    "colab:T4",
    "colab:L4",
    "colab:A100-40GB",
    "colab:A100-80GB",
)


def get_candidate(candidate_id: str) -> Candidate:
    """Return the catalog entry for ``candidate_id``.

    Raises:
        ConfigError: ``candidate_id`` is not in the catalog.
    """
    try:
        return CANDIDATES[candidate_id]
    except KeyError as exc:
        raise ConfigError(
            f"unknown candidate id: {candidate_id}",
            details={"candidate_id": candidate_id},
        ) from exc
