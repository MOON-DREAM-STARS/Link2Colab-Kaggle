"""Scheduler and quota settings, including the example YAML block."""

from pathlib import Path

import pytest
import yaml

from link2kaggle_colab.core.config import QuotaSettings, SchedulerSettings, load_config
from link2kaggle_colab.core.errors import ConfigError

_EXAMPLE = Path(__file__).resolve().parents[2] / "config.example.yaml"
_SOURCE = (
    "unverified estimate: third-party report "
    "https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/ "
    "(checked 2026-03-10); Google publishes no fixed consumer Colab CCU rate; "
    "confirm with `colab usage`"
)
_HIGH_MEM_SOURCE = (
    "unverified fixture estimate: high-mem shape uplift over the base "
    "accelerator rate; Google publishes no fixed consumer Colab CCU rate; "
    "confirm with `colab usage`"
)
_TPU_SOURCE = (
    "unverified fixture estimate: Colab TPU CCU rate; "
    "Google publishes no fixed consumer Colab CCU rate; confirm with `colab usage`"
)
_BASE_RATE_KEYS = ("colab:T4", "colab:L4", "colab:A100-40GB", "colab:A100-80GB")
_HIGH_MEM_RATE_KEYS = (
    "colab:T4+high_mem",
    "colab:A100-40GB+high_mem",
    "colab:A100-80GB+high_mem",
)
_TPU_RATE_KEYS = ("colab:TPU-v5e1", "colab:TPU-v6e1")


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "scheduler.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_scheduler_defaults_match_the_documented_block() -> None:
    settings = SchedulerSettings()
    assert settings.strategy == "kaggle_first"
    assert settings.allow_colab_fallback is True
    assert settings.allow_premium_gpu is False
    assert settings.min_colab_ccu_reserve == 50
    assert settings.max_ccu_per_job == 20
    assert settings.require_premium_confirmation is True
    assert settings.unknown_kaggle_quota == "warn"
    assert settings.cost_priority == [
        "kaggle:T4x2",
        "colab:T4",
        "colab:L4",
        "colab:A100-40GB",
        "colab:A100-80GB",
    ]
    assert "colab:H100" not in settings.ccu_rates
    assert settings.ccu_rates["colab:T4"].ccu_per_hour == 1.19
    assert settings.ccu_rates["colab:L4"].ccu_per_hour == 1.71
    assert settings.ccu_rates["colab:A100-40GB"].ccu_per_hour == 5.40
    assert settings.ccu_rates["colab:A100-80GB"].ccu_per_hour == 7.52
    assert settings.ccu_rates["colab:T4+high_mem"].ccu_per_hour == 1.45
    assert settings.ccu_rates["colab:A100-40GB+high_mem"].ccu_per_hour == 6.50
    assert settings.ccu_rates["colab:A100-80GB+high_mem"].ccu_per_hour == 9.00
    assert settings.ccu_rates["colab:TPU-v5e1"].ccu_per_hour == 4.50
    assert settings.ccu_rates["colab:TPU-v6e1"].ccu_per_hour == 6.00
    for key in _BASE_RATE_KEYS:
        assert settings.ccu_rates[key].source == _SOURCE
        assert settings.ccu_rates[key].verified is False
    for key in _HIGH_MEM_RATE_KEYS:
        assert settings.ccu_rates[key].source == _HIGH_MEM_SOURCE
        assert settings.ccu_rates[key].verified is False
    for key in _TPU_RATE_KEYS:
        assert settings.ccu_rates[key].source == _TPU_SOURCE
        assert settings.ccu_rates[key].verified is False


def test_example_yaml_loads_and_matches_defaults() -> None:
    raw = yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))
    scheduler = raw["scheduler"]
    assert scheduler["strategy"] == "kaggle_first"
    assert scheduler["allow_colab_fallback"] is True
    assert scheduler["allow_premium_gpu"] is False
    assert scheduler["require_premium_confirmation"] is True
    assert scheduler["min_colab_ccu_reserve"] == 50
    assert scheduler["max_ccu_per_job"] == 20
    assert scheduler["unknown_kaggle_quota"] == "warn"
    assert scheduler["cost_priority"] == [
        "kaggle:T4x2",
        "colab:T4",
        "colab:L4",
        "colab:A100-40GB",
        "colab:A100-80GB",
    ]
    assert set(scheduler["ccu_rates"]) == {
        "colab:T4",
        "colab:L4",
        "colab:A100-40GB",
        "colab:A100-80GB",
        "colab:T4+high_mem",
        "colab:A100-40GB+high_mem",
        "colab:A100-80GB+high_mem",
        "colab:TPU-v5e1",
        "colab:TPU-v6e1",
    }
    assert raw["quota"]["cache_ttl_seconds"] == 300
    text = _EXAMPLE.read_text(encoding="utf-8")
    assert "not a performance ranking" in text
    assert "unverified" in text
    assert "never compared" in text
    loaded = load_config(_EXAMPLE)
    assert loaded.scheduler == SchedulerSettings()
    assert loaded.quota == QuotaSettings()


def test_unknown_cost_priority_id_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "scheduler:\n  cost_priority:\n    - not-a-candidate\n",
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_duplicate_cost_priority_id_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        'scheduler:\n  cost_priority:\n    - "colab:T4"\n    - "colab:T4"\n',
    )
    with pytest.raises(ConfigError):
        load_config(path)


def test_empty_cost_priority_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "scheduler:\n  cost_priority: []\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_kaggle_ccu_rate_key_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "scheduler:\n"
        "  ccu_rates:\n"
        '    "kaggle:T4x2":\n'
        "      ccu_per_hour: 1\n"
        "      source: not-colab\n"
        "      verified: false\n",
    )
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("rate", [0, -1])
def test_non_positive_ccu_rate_is_rejected(tmp_path: Path, rate: int) -> None:
    path = _write(
        tmp_path,
        "scheduler:\n"
        "  ccu_rates:\n"
        '    "colab:T4":\n'
        f"      ccu_per_hour: {rate}\n"
        "      source: bad\n"
        "      verified: false\n",
    )
    with pytest.raises(ConfigError):
        load_config(path)
