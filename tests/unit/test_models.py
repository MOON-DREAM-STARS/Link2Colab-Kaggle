"""Model contracts: per-GPU memory, job specs, and terminal states."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from link2kaggle_colab.core.models import (
    GpuType,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
    QuotaUnit,
    ResourceSpec,
)
from link2kaggle_colab.providers.mock_fixtures import (
    COLAB_CCU_RATES_PER_HOUR_FIXTURE,
    colab_resources,
    kaggle_resources,
)


def test_kaggle_t4_is_two_gpus_of_16gb_not_32() -> None:
    t4 = next(item for item in kaggle_resources() if item.gpu_type is GpuType.T4)
    assert t4.gpu_count == 2
    assert t4.gpu_memory_gb_per_gpu == 16
    assert t4.gpu_memory_gb_per_gpu != 32
    assert t4.provider_resource_id == "NvidiaTeslaT4"
    assert t4.cpu_only is False
    assert t4.fits(16, 2) is True
    assert t4.fits(16, 3) is False
    assert t4.fits(32, 1) is False
    assert t4.fits(None, 2) is True
    p100 = next(item for item in kaggle_resources() if item.gpu_type is GpuType.P100)
    assert p100.gpu_count == 1
    assert p100.gpu_memory_gb_per_gpu == 16
    assert p100.provider_resource_id == "NvidiaTeslaP100"
    cpu = next(item for item in kaggle_resources() if item.cpu_only)
    assert cpu.gpu_type is GpuType.NONE
    assert cpu.fits(None, 1) is False


def test_colab_catalog_memory_and_fixture_rates() -> None:
    resources = colab_resources("ai_pro")
    t4 = next(item for item in resources if item.gpu_type is GpuType.T4)
    assert t4.gpu_memory_gb_per_gpu == 16
    assert "15" in t4.label
    a100 = next(item for item in resources if item.gpu_type is GpuType.A100)
    assert a100.gpu_memory_gb_per_gpu is None
    assert a100.premium is True
    assert a100.fits(40, 1) is None
    assert a100.provider_resource_id == "A100"
    h100 = next(item for item in resources if item.gpu_type is GpuType.H100)
    assert h100.gpu_memory_gb_per_gpu == 80
    assert h100.premium is True
    l4 = next(item for item in resources if item.gpu_type is GpuType.L4)
    assert l4.gpu_memory_gb_per_gpu == 24
    free = colab_resources("free")
    assert {item.gpu_type for item in free} == {GpuType.T4, GpuType.NONE}
    assert a100.supports_high_mem is True
    assert l4.supports_high_mem is False
    tpu = {item.gpu_type for item in resources} & {GpuType.TPU_V5E1, GpuType.TPU_V6E1}
    assert tpu == {GpuType.TPU_V5E1, GpuType.TPU_V6E1}
    assert COLAB_CCU_RATES_PER_HOUR_FIXTURE["T4"] == 1.8
    assert COLAB_CCU_RATES_PER_HOUR_FIXTURE["TPU_V5E1"] == 6.0
    assert COLAB_CCU_RATES_PER_HOUR_FIXTURE["T4+high_mem"] == 2.2


def test_resource_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ResourceSpec(gpu_type=GpuType.T4, gpu_count=1, unexpected=True)


def test_quota_known_and_unknown_factory() -> None:
    known = QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.gpu_hours,
        total=30,
        used=30,
        remaining=0,
        is_fixture=True,
        source="mock-fixture",
    )
    assert known.known is True
    missing = QuotaInfo.unknown(ProviderName.colab, "format not recognized")
    assert missing.known is False
    assert missing.unit is QuotaUnit.unknown
    assert missing.is_fixture is False
    assert missing.source == "unknown"
    assert missing.remaining is None


def test_job_state_terminal_helper() -> None:
    assert JobState.SUCCEEDED.is_terminal() is True
    assert JobState.FAILED.is_terminal() is True
    assert JobState.CANCELLED.is_terminal() is True
    assert JobState.RUNNING.is_terminal() is False
    assert JobState.QUEUED.is_terminal() is False
    assert JobState.UNKNOWN.is_terminal() is False
    assert JobState.CANCELLING.is_terminal() is False


def test_job_spec_rejects_missing_file_bad_suffix_and_bad_name(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        JobSpec(script_path=tmp_path / "missing.py")
    text = tmp_path / "notes.txt"
    text.write_text("nope", encoding="utf-8")
    with pytest.raises(ValidationError):
        JobSpec(script_path=text)
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, name="Bad_Name")
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, name="a" * 51)
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, name="-ab")
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, timeout_seconds=0)
    good = JobSpec(script_path=script, name="a", timeout_seconds=None, gpu_count=1)
    assert good.script_path == script.resolve()
    assert good.name == "a"
    long_name = "a" * 50
    assert JobSpec(script_path=script, name=long_name).name == long_name


def test_gpu_count_zero_rejects_concrete_gpu_type(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="gpu_count 0 requests CPU"):
        JobSpec(script_path=script, gpu_type=GpuType.T4, gpu_count=0)
    none = JobSpec(script_path=script, gpu_type=GpuType.NONE, gpu_count=0)
    assert none.gpu_type is GpuType.NONE
    assert none.gpu_count == 0
    omitted = JobSpec(script_path=script, gpu_count=0)
    assert omitted.gpu_type is None
    assert omitted.gpu_count == 0


def test_job_spec_resolves_relative_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path("rel.py")
    path.write_text("print(1)\n", encoding="utf-8")
    spec = JobSpec(script_path=Path("rel.py"))
    assert spec.script_path == path.resolve()
    assert spec.script_path.is_absolute()

def test_job_spec_high_mem_and_tpu_rules(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    ok = JobSpec(script_path=script, gpu_type=GpuType.A100, allow_premium=True, high_mem=True)
    assert ok.high_mem is True
    tpu = JobSpec(script_path=script, gpu_type=GpuType.TPU_V5E1, allow_premium=True)
    assert tpu.gpu_type is GpuType.TPU_V5E1
    with pytest.raises(ValidationError, match="high_mem"):
        JobSpec(script_path=script, gpu_type=GpuType.L4, high_mem=True)
    with pytest.raises(ValidationError, match="high_mem"):
        JobSpec(script_path=script, gpu_type=GpuType.TPU_V5E1, high_mem=True)
    with pytest.raises(ValidationError, match="gpu_count"):
        JobSpec(script_path=script, gpu_type=GpuType.TPU_V5E1, gpu_count=2)

