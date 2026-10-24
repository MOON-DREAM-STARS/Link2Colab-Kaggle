"""Provider registry selects mocks by default and CLI adapters in real mode."""

from pathlib import Path

import pytest

from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import ConfigError
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.core.process import SubprocessRunner
from link2kaggle_colab.providers.colab.cli_provider import ColabCliProvider
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.fake.provider import FakeProvider
from link2kaggle_colab.providers.kaggle.cli_provider import KaggleCliProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.providers.registry import build_provider
from tests.mocks.runner import RecordingRunner


def test_registry_builds_mock_providers_by_default(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path, allow_real_jobs=True)
    assert settings.mode == "mock"
    kaggle = build_provider("kaggle", settings)
    colab = build_provider(ProviderName.colab, settings)
    fake = build_provider("fake", settings, runner=object())  # type: ignore[arg-type]
    assert isinstance(kaggle, MockKaggleProvider)
    assert isinstance(colab, MockColabProvider)
    assert isinstance(fake, FakeProvider)
    assert kaggle.is_mock is True
    assert colab.is_mock is True
    assert fake.is_mock is True


def test_real_mode_builds_cli_adapters_and_keeps_fake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(mode="real", allow_real_jobs=True, state_dir=tmp_path)
    bare = build_provider("kaggle", settings)
    assert isinstance(bare, KaggleCliProvider)
    assert isinstance(bare._runner, SubprocessRunner)
    assert isinstance(build_provider(ProviderName.fake, settings), FakeProvider)

    runner = RecordingRunner()
    monkeypatch.setattr(
        "link2kaggle_colab.providers.kaggle.cli_provider.shutil.which",
        lambda cmd: "/usr/bin/kaggle" if cmd == "kaggle" else None,
    )
    runner.script(["kaggle", "--version"], stdout="Kaggle CLI 2.2.5\n")
    kaggle = build_provider("kaggle", settings, runner=runner)
    colab = build_provider("colab", settings, runner=runner)
    assert isinstance(kaggle, KaggleCliProvider)
    assert isinstance(colab, ColabCliProvider)
    assert kaggle.is_mock is False
    assert colab.is_mock is False
    assert kaggle.detect_cli()["version"] == "Kaggle CLI 2.2.5"
    assert runner.calls == [["kaggle", "--version"]]
    with pytest.raises(ConfigError):
        build_provider("other", settings)


def test_builtin_factory_is_ignored_and_plugin_is_explicit(tmp_path: Path) -> None:
    import link2kaggle_colab.providers.registry as registry

    def builtin_factory(
        settings: Settings,
        account: object,
        runner: object,
    ) -> FakeProvider:
        raise AssertionError("built-in factory must not run")

    seen: dict[str, object] = {}

    def custom_factory(
        settings: Settings,
        account: object,
        runner: object,
    ) -> FakeProvider:
        seen["settings"] = settings
        seen["account"] = account
        seen["runner"] = runner
        return FakeProvider(settings)

    registry._FACTORIES.clear()
    try:
        registry.register_provider_factory("kaggle", builtin_factory)
        assert "kaggle" not in registry._FACTORIES
        provider = build_provider("kaggle", Settings(state_dir=tmp_path))
        assert isinstance(provider, MockKaggleProvider)

        runner = RecordingRunner()
        registry.register_provider_factory("custom", custom_factory)
        registry.register_provider_factory("custom", builtin_factory)
        built = build_provider("custom", Settings(state_dir=tmp_path), runner=runner)
        assert isinstance(built, FakeProvider)
        assert seen["account"] is None
        assert seen["runner"] is runner
    finally:
        registry._FACTORIES.clear()


def test_entry_point_factories_load_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as metadata

    import link2kaggle_colab.providers.registry as registry

    calls = {"n": 0}

    def fake_entry_points(*, group: str) -> list[object]:
        calls["n"] += 1
        assert group == "link2kaggle_colab.providers"
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    assert calls["n"] == 0
    registry.load_entry_point_factories()
    assert calls["n"] == 1
