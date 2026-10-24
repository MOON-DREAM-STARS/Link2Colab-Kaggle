"""CCU estimates stay in compute units. GPU-hours are a separate Kaggle unit."""

from link2kaggle_colab.core.config import SchedulerSettings
from link2kaggle_colab.quota import cost


def test_t4_hour_is_1_19_compute_units() -> None:
    estimate = cost.estimate_ccu("colab:T4", SchedulerSettings().ccu_rates, 3600)
    assert estimate.estimated_ccu == 1.19
    assert estimate.estimated_hours == 1
    assert estimate.ccu_per_hour == 1.19
    assert estimate.rate_verified is False
    assert estimate.reason == ""
    assert estimate.unit == "compute_units"


def test_a100_40gb_two_hours_is_10_8_compute_units() -> None:
    estimate = cost.estimate_ccu("colab:A100-40GB", SchedulerSettings().ccu_rates, 7200)
    assert estimate.estimated_ccu == 10.8
    assert estimate.estimated_hours == 2
    assert estimate.unit == "compute_units"


def test_missing_timeout_has_no_ccu_estimate() -> None:
    estimate = cost.estimate_ccu("colab:T4", SchedulerSettings().ccu_rates, None)
    assert estimate.estimated_ccu is None
    assert estimate.estimated_hours is None
    assert estimate.reason == "timeout_seconds is required to bound CCU spend"
    assert estimate.unit == "compute_units"


def test_h100_without_a_rate_is_not_priced() -> None:
    rates = SchedulerSettings().ccu_rates
    assert "colab:H100" not in rates
    estimate = cost.estimate_ccu("colab:H100", rates, 3600)
    assert estimate.estimated_ccu is None
    assert estimate.estimated_hours == 1
    assert estimate.reason == "no CCU rate configured for colab:H100"
    assert estimate.unit == "compute_units"


def test_kaggle_hours_needed_is_gpu_hours() -> None:
    assert cost.kaggle_hours_needed(5400) == 1.5
    assert cost.kaggle_hours_needed(None) is None


def test_cost_estimate_unit_defaults_to_compute_units() -> None:
    estimate = cost.CostEstimate(
        candidate_id="colab:T4",
        ccu_per_hour=1.19,
        rate_source="test",
        rate_verified=False,
        estimated_hours=1,
        estimated_ccu=1.19,
    )
    assert estimate.unit == "compute_units"


def test_module_never_converts_ccu_and_gpu_hours() -> None:
    text = cost.__doc__ or ""
    assert "different units" in text
    assert "never converts" in text

def test_estimate_ccu_prefers_high_mem_rate() -> None:
    from link2kaggle_colab.core.config import SchedulerSettings
    from link2kaggle_colab.quota import cost

    rates = SchedulerSettings().ccu_rates
    base = cost.estimate_ccu("colab:T4", rates, 3600, high_mem=False)
    high = cost.estimate_ccu("colab:T4", rates, 3600, high_mem=True)
    assert base.estimated_ccu == 1.19
    assert high.estimated_ccu == 1.45
    tpu = cost.estimate_ccu("colab:TPU-v5e1", rates, 3600)
    assert tpu.estimated_ccu == 4.5

