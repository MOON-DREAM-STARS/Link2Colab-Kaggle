"""Candidate catalog ids, per-GPU memory, and cost priority."""

import pytest

from link2kaggle_colab.core.errors import ConfigError
from link2kaggle_colab.core.models import GpuType, ProviderName
from link2kaggle_colab.scheduler import candidates

# provider, gpu, count, memory per GPU, paid, premium, variant_selectable
# provider, gpu, count, memory per GPU, paid, premium, variant_selectable, supports_high_mem
_CATALOG: dict[str, tuple[ProviderName, GpuType, int, float | None, bool, bool, bool, bool]] = {
    "kaggle:T4x2": (ProviderName.kaggle, GpuType.T4, 2, 16, False, False, True, False),
    "kaggle:P100": (ProviderName.kaggle, GpuType.P100, 1, 16, False, False, True, False),
    "colab:T4": (ProviderName.colab, GpuType.T4, 1, 16, True, False, True, True),
    "colab:L4": (ProviderName.colab, GpuType.L4, 1, 24, True, False, True, False),
    "colab:A100-40GB": (ProviderName.colab, GpuType.A100, 1, 40, True, True, True, True),
    "colab:A100-80GB": (ProviderName.colab, GpuType.A100, 1, 80, True, True, False, True),
    "colab:H100": (ProviderName.colab, GpuType.H100, 1, 80, True, True, True, True),
    "colab:TPU-v5e1": (ProviderName.colab, GpuType.TPU_V5E1, 1, None, True, True, True, False),
    "colab:TPU-v6e1": (ProviderName.colab, GpuType.TPU_V6E1, 1, None, True, True, True, False),
}


def test_catalog_ids_match_the_listed_variants() -> None:
    assert set(candidates.CANDIDATES) == set(_CATALOG)
    for candidate_id, expected in _CATALOG.items():
        item = candidates.CANDIDATES[candidate_id]
        provider, gpu_type, count, memory, paid, premium, selectable, high_mem = expected
        assert item.id == candidate_id
        assert item.provider is provider
        assert item.gpu_type is gpu_type
        assert item.gpu_count == count
        assert item.gpu_memory_gb_per_gpu == memory
        assert item.paid is paid
        assert item.premium is premium
        assert item.variant_selectable is selectable
        assert item.supports_high_mem is high_mem
        assert candidates.get_candidate(candidate_id) is item


def test_kaggle_t4x2_is_two_16gb_gpus_not_32() -> None:
    item = candidates.CANDIDATES["kaggle:T4x2"]
    assert item.gpu_count == 2
    assert item.gpu_memory_gb_per_gpu == 16
    assert item.gpu_memory_gb_per_gpu != 32
    assert item.note == "two separate 16 GB GPUs, not 32 GB"


def test_a100_80gb_variant_is_not_selectable() -> None:
    item = candidates.get_candidate("colab:A100-80GB")
    assert item.variant_selectable is False
    assert item.gpu_memory_gb_per_gpu == 80
    assert "40GB/80GB" in item.note
    assert candidates.get_candidate("colab:A100-40GB").variant_selectable is True


def test_unknown_candidate_raises_config_error() -> None:
    with pytest.raises(ConfigError) as caught:
        candidates.get_candidate("colab:nope")
    assert caught.value.code == "config_error"


def test_default_cost_priority_order() -> None:
    assert candidates.DEFAULT_COST_PRIORITY == (
        "kaggle:T4x2",
        "colab:T4",
        "colab:L4",
        "colab:A100-40GB",
        "colab:A100-80GB",
    )


def test_module_docstring_describes_cost_preference() -> None:
    text = candidates.__doc__ or ""
    assert "COST preference" in text
    assert "not a performance ranking" in text
    assert "CCU-metered" in text
