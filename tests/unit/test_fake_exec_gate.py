"""FakeProvider refuses local exec when allow_exec is false."""

from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import FakeProviderSettings, ProvidersSettings, Settings
from link2kaggle_colab.core.errors import InvalidInputError
from link2kaggle_colab.core.models import JobSpec, ProviderName
from link2kaggle_colab.mcp.server import _env_flag, _mcp_settings


def test_fake_submit_blocked_when_allow_exec_false(tmp_path: Path, py_script: Path) -> None:
    settings = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        artifact_root=tmp_path,
        providers=ProvidersSettings(fake=FakeProviderSettings(allow_exec=False)),
    )
    client = Client(settings)
    with pytest.raises(InvalidInputError, match="fake local exec is disabled"):
        client.submit(
            JobSpec(script_path=py_script, provider=ProviderName.fake, gpu_type=None, gpu_count=0)
        )


def test_fake_submit_allowed_when_allow_exec_true(tmp_path: Path, py_script: Path) -> None:
    settings = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        artifact_root=tmp_path,
        providers=ProvidersSettings(fake=FakeProviderSettings(allow_exec=True)),
    )
    client = Client(settings)
    handle = client.submit(
        JobSpec(script_path=py_script, provider=ProviderName.fake, gpu_type=None, gpu_count=0)
    )
    assert handle.provider is ProviderName.fake


def test_mcp_settings_disables_fake_exec_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("L2KC_ALLOW_FAKE_EXEC", raising=False)
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = _mcp_settings()
    assert settings.providers.fake.allow_exec is False


def test_mcp_settings_honors_allow_fake_exec_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("L2KC_ALLOW_FAKE_EXEC", "1")
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = _mcp_settings()
    assert settings.providers.fake.allow_exec is True
    assert _env_flag("L2KC_ALLOW_FAKE_EXEC") is True
