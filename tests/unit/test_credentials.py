"""Credential resolution builds a per-account environment and does not log values."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from link2kaggle_colab.accounts.credentials import credential_health, resolve_credential
from link2kaggle_colab.accounts.models import (
    AdcFileCredential,
    ColabHomeCredential,
    EnvCredential,
    KaggleConfigDirCredential,
    KeyringCredential,
)
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.envspec import apply_account_env
from link2kaggle_colab.core.errors import CredentialRefError
from link2kaggle_colab.core.models import ProviderName

_KAGGLE_TOKEN_VARS = frozenset({"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"})
_KAGGLE_ENV_UNSET = frozenset({"KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY"})


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    payload: dict[str, object] = {"state_dir": tmp_path}
    payload.update(overrides)
    return Settings.model_validate(payload)


def test_kaggle_config_dir_sets_config_dir_and_unsets_token_vars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "parent-token")
    monkeypatch.setenv("KAGGLE_USERNAME", "parent-user")
    monkeypatch.setenv("KAGGLE_KEY", "parent-key")
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", "/parent/kaggle")
    before = dict(os.environ)
    path = "~/.l2kc/credentials/kaggle-main"
    resolved = resolve_credential(
        "kaggle-main",
        KaggleConfigDirCredential(source="kaggle_config_dir", path=path),
        ProviderName.kaggle,
        _settings(tmp_path),
    )
    assert dict(os.environ) == before
    expected = Path(path).expanduser()
    assert resolved.env.set == {"KAGGLE_CONFIG_DIR": str(expected)}
    assert resolved.env.unset == _KAGGLE_TOKEN_VARS
    assert resolved.home is None
    assert resolved.state_root == expected
    directory = tmp_path / "kaggle-home"
    directory.mkdir()
    secret = "KGAT_" + ("z" * 20)
    (directory / "kaggle.json").write_text(
        json.dumps({"username": "u", "key": secret}),
        encoding="utf-8",
    )
    declared = KaggleConfigDirCredential(source="kaggle_config_dir", path=str(directory))
    health = credential_health("kaggle-main", declared, ProviderName.kaggle, _settings(tmp_path))
    assert health["state"] == "present"
    assert secret not in json.dumps(health)
    assert health["source"].endswith("kaggle.json")


def test_env_credential_is_read_at_call_time_and_repr_masks_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "KAGGLE_ALT_API_TOKEN"
    first_secret = "KGAT_" + ("a" * 16)
    second_secret = "KGAT_" + ("b" * 16)
    monkeypatch.setenv(name, first_secret)
    ref = EnvCredential(source="env", env=name)
    settings = _settings(tmp_path)
    first = resolve_credential("kaggle-alt", ref, ProviderName.kaggle, settings)
    assert first.env.set["KAGGLE_API_TOKEN"] == first_secret
    assert first.env.unset == _KAGGLE_ENV_UNSET
    assert first_secret not in repr(first.env)
    assert first_secret not in str(first.env)
    assert first_secret not in repr(first)
    monkeypatch.setenv(name, second_secret)
    second = resolve_credential("kaggle-alt", ref, ProviderName.kaggle, settings)
    assert second.env.set["KAGGLE_API_TOKEN"] == second_secret
    assert first.env.set["KAGGLE_API_TOKEN"] == first_secret
    assert second_secret not in repr(second.env)
    assert second_secret not in str(second)
    health = credential_health("kaggle-alt", ref, ProviderName.kaggle, settings)
    assert health == {"state": "present", "source": f"env:{name}"}
    assert first_secret not in json.dumps(health)
    assert second_secret not in json.dumps(health)


def test_adc_file_sets_google_application_credentials(tmp_path: Path) -> None:
    path = "~/creds/colab-main-adc.json"
    resolved = resolve_credential(
        "colab-main",
        AdcFileCredential(source="adc_file", path=path),
        ProviderName.colab,
        _settings(tmp_path),
    )
    expected = Path(path).expanduser()
    assert resolved.env.set == {"GOOGLE_APPLICATION_CREDENTIALS": str(expected)}
    assert resolved.env.unset == frozenset()
    assert resolved.home is None
    assert resolved.state_root == expected.parent
    health = credential_health(
        "colab-main",
        AdcFileCredential(source="adc_file", path=path),
        ProviderName.colab,
        _settings(tmp_path),
    )
    assert health["state"] == "missing"
    assert health["source"] == f"file:{expected}"


def test_colab_home_sets_home_and_unsets_adc(tmp_path: Path) -> None:
    path = "~/.l2kc/credentials/colab-alt"
    resolved = resolve_credential(
        "colab-alt",
        ColabHomeCredential(source="colab_home", path=path),
        ProviderName.colab,
        _settings(tmp_path),
    )
    expected = Path(path).expanduser()
    assert resolved.env.set == {"HOME": str(expected)}
    assert resolved.env.unset == frozenset({"GOOGLE_APPLICATION_CREDENTIALS"})
    assert resolved.home == expected
    assert resolved.state_root == expected


def test_undeclared_ref_uses_convention_path(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    kaggle = resolve_credential("custom-kaggle", None, ProviderName.kaggle, settings)
    kaggle_root = (tmp_path / "credentials" / "custom-kaggle").resolve()
    assert Path(kaggle.env.set["KAGGLE_CONFIG_DIR"]) == kaggle_root
    assert kaggle.env.unset == _KAGGLE_TOKEN_VARS

    colab_adc = resolve_credential("custom-colab", None, ProviderName.colab, settings)
    adc = kaggle_root.with_name("custom-colab") / "application_default_credentials.json"
    assert Path(colab_adc.env.set["GOOGLE_APPLICATION_CREDENTIALS"]) == adc.resolve()

    oauth = _settings(tmp_path, providers={"colab": {"auth": "oauth2"}})
    colab_home = resolve_credential("custom-colab", None, ProviderName.colab, oauth)
    home = (tmp_path / "credentials" / "custom-colab").resolve()
    assert colab_home.env.set == {"HOME": str(home)}
    assert colab_home.home == home

    fake = resolve_credential("custom-fake", None, ProviderName.fake, settings)
    assert fake.env.set == {}
    assert fake.env.unset == frozenset()
    assert fake.source == "none"


def test_ambient_credential_is_empty(tmp_path: Path) -> None:
    resolved = resolve_credential(None, None, ProviderName.kaggle, _settings(tmp_path))
    assert resolved.env.set == {}
    assert resolved.env.unset == frozenset()
    assert resolved.home is None
    assert resolved.state_root is None
    assert resolved.source == "ambient"
    health = credential_health(None, None, ProviderName.colab, _settings(tmp_path))
    assert health == {"state": "present", "source": "ambient"}


def test_keyring_without_package_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "keyring", None)
    ref = KeyringCredential(source="keyring", service="l2kc", username="kaggle-x")
    settings = _settings(tmp_path)
    with pytest.raises(CredentialRefError) as caught:
        resolve_credential("vault-x", ref, ProviderName.kaggle, settings)
    assert caught.value.code == "credential_ref_error"
    assert caught.value.details["reason"] == "unsupported"
    assert "account_id" in caught.value.details
    health = credential_health("vault-x", ref, ProviderName.kaggle, settings)
    assert health == {"state": "unsupported", "source": "keyring:l2kc"}


def test_two_accounts_have_disjoint_env_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "KGAT_" + ("c" * 16)
    monkeypatch.setenv("KAGGLE_ALT_API_TOKEN", secret)
    settings = _settings(tmp_path)
    kaggle_dir = resolve_credential(
        "kaggle-main",
        KaggleConfigDirCredential(
            source="kaggle_config_dir",
            path=str(tmp_path / "kaggle-main"),
        ),
        ProviderName.kaggle,
        settings,
    )
    kaggle_env = resolve_credential(
        "kaggle-alt",
        EnvCredential(source="env", env="KAGGLE_ALT_API_TOKEN"),
        ProviderName.kaggle,
        settings,
    )
    assert set(kaggle_dir.env.set).isdisjoint(kaggle_env.env.set)
    assert set(kaggle_dir.env.set.values()).isdisjoint(kaggle_env.env.set.values())
    assert secret not in repr(kaggle_env.env)

    colab_adc = resolve_credential(
        "colab-main",
        AdcFileCredential(source="adc_file", path=str(tmp_path / "adc.json")),
        ProviderName.colab,
        settings,
    )
    colab_home = resolve_credential(
        "colab-alt",
        ColabHomeCredential(source="colab_home", path=str(tmp_path / "colab-alt")),
        ProviderName.colab,
        settings,
    )
    assert set(colab_adc.env.set).isdisjoint(colab_home.env.set)
    assert set(colab_adc.env.set.values()).isdisjoint(colab_home.env.set.values())

    base = {
        "PATH": "/usr/bin",
        "KAGGLE_API_TOKEN": "base-token",
        "KAGGLE_USERNAME": "base-user",
        "KAGGLE_KEY": "base-key",
        "KAGGLE_CONFIG_DIR": "/base/kaggle",
        "GOOGLE_APPLICATION_CREDENTIALS": "/base/adc.json",
        "HOME": "/base/home",
    }
    child_dir = apply_account_env(kaggle_dir.env, base)
    child_env = apply_account_env(kaggle_env.env, base)
    assert child_dir["KAGGLE_CONFIG_DIR"] == str(tmp_path / "kaggle-main")
    assert "KAGGLE_API_TOKEN" not in child_dir
    assert "KAGGLE_USERNAME" not in child_dir
    assert "KAGGLE_KEY" not in child_dir
    assert child_env["KAGGLE_API_TOKEN"] == secret
    assert "KAGGLE_CONFIG_DIR" not in child_env
    assert child_dir["PATH"] == base["PATH"]
    assert child_env["HOME"] == base["HOME"]
    assert base == {
        "PATH": "/usr/bin",
        "KAGGLE_API_TOKEN": "base-token",
        "KAGGLE_USERNAME": "base-user",
        "KAGGLE_KEY": "base-key",
        "KAGGLE_CONFIG_DIR": "/base/kaggle",
        "GOOGLE_APPLICATION_CREDENTIALS": "/base/adc.json",
        "HOME": "/base/home",
    }
