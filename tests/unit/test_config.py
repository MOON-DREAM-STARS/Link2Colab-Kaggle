"""Settings load from YAML, default to mock, and refuse secret-looking keys."""

from pathlib import Path

import pytest

from link2kaggle_colab.core.config import JobsSettings, Settings, load_config
from link2kaggle_colab.core.errors import ConfigError, RealModeDisabledError


def test_defaults_are_mock_and_real_jobs_need_both_flags(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path)
    assert settings.mode == "mock"
    assert settings.allow_real_jobs is False
    assert settings.real_enabled is False
    assert settings.providers.kaggle.cli_path == "kaggle"
    assert settings.providers.colab.cli_path == "colab"
    assert settings.providers.colab.auth == "adc"
    assert settings.mock.kaggle_scenario == "ok"
    assert settings.mock.colab_scenario == "ok"
    with pytest.raises(RealModeDisabledError) as caught:
        settings.require_real_jobs_allowed("kaggle")
    assert caught.value.code == "real_mode_disabled"

    mock_with_flag = Settings(mode="mock", allow_real_jobs=True, state_dir=tmp_path)
    assert mock_with_flag.real_enabled is False
    with pytest.raises(RealModeDisabledError):
        mock_with_flag.require_real_jobs_allowed("colab")

    real_without_flag = Settings(mode="real", allow_real_jobs=False, state_dir=tmp_path)
    assert real_without_flag.real_enabled is True
    with pytest.raises(RealModeDisabledError):
        real_without_flag.require_real_jobs_allowed("kaggle")

    enabled = Settings(mode="real", allow_real_jobs=True, state_dir=tmp_path)
    enabled.require_real_jobs_allowed("fake")


def test_missing_file_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    loaded = load_config(None)
    assert loaded.mode == "mock"
    assert loaded.allow_real_jobs is False


def test_search_order_and_state_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    home_dir = home / ".config" / "l2kc"
    home_dir.mkdir(parents=True)
    (home_dir / "config.yaml").write_text(
        "mock:\n  kaggle_scenario: quota_45h\n",
        encoding="utf-8",
    )
    (cwd / "l2kc.yaml").write_text(
        "mock:\n  kaggle_scenario: gpu_unavailable\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    assert load_config(None).mock.kaggle_scenario == "gpu_unavailable"

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert load_config(None).mock.kaggle_scenario == "quota_45h"

    env_file = tmp_path / "from-env.yaml"
    env_file.write_text("mock:\n  kaggle_scenario: quota_insufficient\n", encoding="utf-8")
    monkeypatch.setenv("L2KC_CONFIG", str(env_file))
    assert load_config(None).mock.kaggle_scenario == "quota_insufficient"

    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        f"mode: mock\nstate_dir: {tmp_path / 'from-file'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("L2KC_STATE_DIR", str(tmp_path / "from-env-state"))
    loaded = load_config(explicit)
    assert loaded.mock.kaggle_scenario == "ok"
    assert loaded.state_dir == tmp_path / "from-env-state"


def test_explicit_and_env_paths_must_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError):
        load_config(missing)
    monkeypatch.setenv("L2KC_CONFIG", str(missing))
    with pytest.raises(ConfigError):
        load_config(None)


def test_secret_looking_keys_are_rejected_and_refs_are_allowed(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret.yaml"
    secret_file.write_text(
        "providers:\n  kaggle:\n    api_key: example\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="credential"):
        load_config(secret_file)

    nested = tmp_path / "nested.yaml"
    nested.write_text("client_secret: example\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="credential"):
        load_config(nested)

    allowed = tmp_path / "allowed.yaml"
    allowed.write_text(
        "providers:\n"
        "  kaggle:\n"
        "    credential_ref: env:KAGGLE_API_TOKEN\n"
        "    api_key_env: KAGGLE_API_TOKEN\n"
        "  colab:\n"
        "    auth: oauth2\n",
        encoding="utf-8",
    )
    loaded = load_config(allowed)
    assert loaded.providers.kaggle.credential_ref == "env:KAGGLE_API_TOKEN"
    assert loaded.providers.kaggle.api_key_env == "KAGGLE_API_TOKEN"
    assert loaded.providers.colab.auth == "oauth2"


def test_invalid_yaml_mode_and_example_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(":\n  - [\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(broken)
    listing = tmp_path / "list.yaml"
    listing.write_text("- just-a-list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(listing)
    bad_mode = tmp_path / "mode.yaml"
    bad_mode.write_text("mode: sometimes\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_mode)
    omitted = tmp_path / "omit.yaml"
    omitted.write_text("allow_real_jobs: false\n", encoding="utf-8")
    assert load_config(omitted).mode == "mock"

    example = Path("config.example.yaml")
    loaded = load_config(example)
    assert loaded.mode == "mock"
    assert loaded.allow_real_jobs is False
    assert loaded.providers.kaggle.cli_path == "kaggle"
    assert loaded.providers.colab.cli_path == "colab"
    assert loaded.providers.colab.auth == "adc"
    assert loaded.providers.colab.keep_runtime is False
    assert loaded.providers.colab.default_exec_timeout_seconds == 21600
    blob = example.read_text(encoding="utf-8")
    assert "KGAT_" not in blob
    assert "ya29." not in blob
    assert "password:" not in blob
    assert loaded.jobs == JobsSettings()


def test_jobs_settings_reject_non_positive_bounds(tmp_path: Path) -> None:
    attempts = tmp_path / "attempts.yaml"
    attempts.write_text("jobs:\n  retry_max_attempts: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(attempts)
    timeout = tmp_path / "timeout.yaml"
    timeout.write_text("jobs:\n  submit_timeout_seconds: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(timeout)
    stale = tmp_path / "stale.yaml"
    stale.write_text("jobs:\n  stale_submitting_seconds: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(stale)
