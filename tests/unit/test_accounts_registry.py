"""Registry defaults, legacy ids, and persisted enable/disable overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import link2kaggle_colab.accounts.selection as selection
from link2kaggle_colab.accounts.health import account_health
from link2kaggle_colab.accounts.models import AccountConfig
from link2kaggle_colab.accounts.registry import AccountRegistry
from link2kaggle_colab.accounts.selection import select_account
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import AccountDisabledError, AccountNotFoundError
from link2kaggle_colab.core.models import AuthMethod, AuthState, AuthStatus, ProviderName


def _settings(
    state_dir: Path,
    accounts: list[AccountConfig],
    credentials: dict[str, object],
) -> Settings:
    return Settings.model_validate(
        {
            "state_dir": state_dir,
            "accounts": accounts,
            "credentials": credentials,
        }
    )


def _two_kaggle(
    state_dir: Path,
    *,
    first_default: bool = False,
    second_default: bool = False,
    first_enabled: bool = True,
    second_enabled: bool = True,
) -> Settings:
    return _settings(
        state_dir,
        [
            AccountConfig(
                id="kaggle-primary",
                provider=ProviderName.kaggle,
                credential_ref="ref-primary",
                enabled=first_enabled,
                default=first_default,
            ),
            AccountConfig(
                id="kaggle-alt",
                provider=ProviderName.kaggle,
                credential_ref="ref-alt",
                enabled=second_enabled,
                default=second_default,
            ),
        ],
        {
            "ref-primary": {"source": "env", "env": "KAGGLE_PRIMARY_API_TOKEN"},
            "ref-alt": {"source": "env", "env": "KAGGLE_ALT_API_TOKEN"},
        },
    )


def test_explicit_default_wins(tmp_path: Path) -> None:
    registry = AccountRegistry.from_settings(_two_kaggle(tmp_path, second_default=True))
    chosen = registry.default_for("kaggle")
    assert chosen.id == "kaggle-alt"
    assert chosen.is_default is True
    assert registry.get("kaggle-primary").is_default is False
    assert select_account(registry, ProviderName.kaggle).id == "kaggle-alt"


def test_first_listed_is_default_when_none_flagged(tmp_path: Path) -> None:
    registry = AccountRegistry.from_settings(_two_kaggle(tmp_path))
    chosen = registry.default_for(ProviderName.kaggle)
    assert chosen.id == "kaggle-primary"
    assert chosen.is_default is True
    assert registry.get("kaggle-alt").is_default is False


def test_disabled_default_does_not_fall_through(tmp_path: Path) -> None:
    flagged_dir = tmp_path / "flagged"
    flagged_dir.mkdir()
    flagged = AccountRegistry.from_settings(
        _two_kaggle(flagged_dir, first_default=True, first_enabled=False),
    )
    with pytest.raises(AccountDisabledError) as flagged_error:
        flagged.default_for("kaggle")
    assert flagged_error.value.details["account_id"] == "kaggle-primary"
    with pytest.raises(AccountDisabledError) as selected:
        select_account(flagged, "kaggle", None)
    assert selected.value.details["account_id"] == "kaggle-primary"
    assert flagged.get("kaggle-alt").enabled is True

    listed_dir = tmp_path / "listed"
    listed_dir.mkdir()
    listed = AccountRegistry.from_settings(_two_kaggle(listed_dir, first_enabled=False))
    with pytest.raises(AccountDisabledError) as listed_error:
        listed.default_for(ProviderName.kaggle)
    assert listed_error.value.details["account_id"] == "kaggle-primary"
    assert listed.get("kaggle-alt").enabled is True


def test_select_account_explicit_unknown_wrong_provider_and_disabled(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        [
            AccountConfig(
                id="kaggle-primary",
                provider=ProviderName.kaggle,
                credential_ref="ref-primary",
                enabled=False,
                default=True,
            ),
            AccountConfig(
                id="colab-primary",
                provider=ProviderName.colab,
                credential_ref="ref-colab",
                enabled=True,
            ),
        ],
        {
            "ref-primary": {"source": "env", "env": "KAGGLE_PRIMARY_API_TOKEN"},
            "ref-colab": {"source": "adc_file", "path": "~/creds/colab.json"},
        },
    )
    registry = AccountRegistry.from_settings(settings)

    with pytest.raises(AccountNotFoundError) as missing:
        select_account(registry, "kaggle", "missing-account")
    assert missing.value.code == "account_not_found"
    assert missing.value.details["account_id"] == "missing-account"

    with pytest.raises(AccountNotFoundError) as mismatch:
        select_account(registry, "colab", "kaggle-primary")
    assert mismatch.value.code == "account_not_found"
    assert mismatch.value.details["reason"] == "provider_mismatch"
    assert mismatch.value.details["account_id"] == "kaggle-primary"

    with pytest.raises(AccountDisabledError) as disabled:
        select_account(registry, "kaggle", "kaggle-primary")
    assert disabled.value.code == "account_disabled"
    assert disabled.value.details["account_id"] == "kaggle-primary"
    assert registry.get("colab-primary").enabled is True


def test_resolve_for_record_legacy_alias(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    registry = AccountRegistry.from_settings(_two_kaggle(plain, first_default=True))
    assert registry.resolve_for_record("kaggle-primary", "kaggle").id == "kaggle-primary"
    assert registry.resolve_for_record("kaggle-default", ProviderName.kaggle).id == "kaggle-primary"
    with pytest.raises(AccountNotFoundError) as mismatch:
        registry.resolve_for_record("kaggle-alt", "colab")
    assert mismatch.value.details["reason"] == "provider_mismatch"
    with pytest.raises(AccountNotFoundError):
        registry.resolve_for_record("missing-account", "kaggle")

    disabled_dir = tmp_path / "disabled"
    disabled_dir.mkdir()
    disabled = AccountRegistry.from_settings(
        _two_kaggle(disabled_dir, first_default=True, first_enabled=False),
    )
    with pytest.raises(AccountDisabledError) as caught:
        disabled.resolve_for_record("kaggle-default", "kaggle")
    assert caught.value.details["account_id"] == "kaggle-primary"

    named = tmp_path / "named"
    named.mkdir()
    explicit = AccountRegistry.from_settings(
        _settings(
            named,
            [
                AccountConfig(
                    id="kaggle-default",
                    provider=ProviderName.kaggle,
                    credential_ref="ref-legacy",
                    default=False,
                ),
                AccountConfig(
                    id="kaggle-alt",
                    provider=ProviderName.kaggle,
                    credential_ref="ref-alt",
                    default=True,
                ),
            ],
            {
                "ref-legacy": {"source": "env", "env": "KAGGLE_LEGACY_API_TOKEN"},
                "ref-alt": {"source": "env", "env": "KAGGLE_ALT_API_TOKEN"},
            },
        )
    )
    assert explicit.resolve_for_record("kaggle-default", "kaggle").id == "kaggle-default"
    assert explicit.default_for("kaggle").id == "kaggle-alt"


def test_enable_disable_persists_across_a_fresh_registry(tmp_path: Path) -> None:
    settings = _two_kaggle(tmp_path, second_default=True)
    registry = AccountRegistry.from_settings(settings)
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    registry.set_enabled("kaggle-alt", False)
    after = {path for path in tmp_path.rglob("*") if path.is_file()}
    assert after - before == {tmp_path / "accounts_state.json"}
    stored = json.loads((tmp_path / "accounts_state.json").read_text(encoding="utf-8"))
    assert stored == {"disabled": ["kaggle-alt"], "enabled": []}
    fresh = AccountRegistry.from_settings(settings)
    assert fresh.get("kaggle-alt").enabled is False
    assert fresh.get("kaggle-primary").enabled is True
    assert settings.accounts is not None
    assert settings.accounts[1].enabled is True

    registry.set_enabled("kaggle-alt", True)
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == after
    stored = json.loads((tmp_path / "accounts_state.json").read_text(encoding="utf-8"))
    assert stored == {"disabled": [], "enabled": ["kaggle-alt"]}
    again = AccountRegistry.from_settings(settings)
    assert again.get("kaggle-alt").enabled is True


def test_selection_docstring_states_the_no_rotation_rule() -> None:
    text = selection.__doc__ or ""
    lowered = text.lower()
    assert "no automatic account rotation" in lowered
    assert "quota" in lowered
    assert "never" in lowered


def test_unconfigured_providers_keep_implicit_defaults(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        [
            AccountConfig(
                id="kaggle-a",
                provider=ProviderName.kaggle,
                credential_ref="ref-a",
            )
        ],
        {"ref-a": {"source": "env", "env": "KAGGLE_A_TOKEN"}},
    )
    registry = AccountRegistry.from_settings(settings)
    assert [account.id for account in registry.list()] == [
        "kaggle-a",
        "colab-default",
        "fake-default",
    ]
    colab = registry.get("colab-default")
    fake = registry.get("fake-default")
    assert colab.implicit is True and colab.is_default is True and colab.enabled is True
    assert colab.credential_ref is None
    assert fake.implicit is True and fake.is_default is True and fake.enabled is True
    assert registry.get("kaggle-a").implicit is False
    assert registry.get("kaggle-a").is_default is True


def test_configured_account_follows_disabled_provider(tmp_path: Path) -> None:
    settings = Settings.model_validate(
        {
            "state_dir": tmp_path,
            "providers": {"kaggle": {"enabled": False}, "fake": {"enabled": False}},
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "credential_ref": "ref-a",
                    "enabled": True,
                }
            ],
            "credentials": {"ref-a": {"source": "env", "env": "KAGGLE_A_TOKEN"}},
        }
    )
    registry = AccountRegistry.from_settings(settings)
    assert registry.get("kaggle-a").enabled is False
    with pytest.raises(AccountDisabledError) as caught:
        registry.default_for("kaggle")
    assert caught.value.details["account_id"] == "kaggle-a"
    assert registry.get("colab-default").enabled is True
    assert registry.get("colab-default").implicit is True
    assert registry.get("fake-default").enabled is False
    assert registry.get("fake-default").implicit is True


def test_account_health_skips_real_cli_and_reports_mock_auth(tmp_path: Path) -> None:
    settings = _two_kaggle(tmp_path, second_default=True)
    account = AccountRegistry.from_settings(settings).get("kaggle-alt")

    class _Real:
        is_mock = False

        def auth_status(self) -> AuthStatus:
            raise AssertionError("real auth_status must not run")

    class _Mock:
        is_mock = True

        def auth_status(self) -> AuthStatus:
            return AuthStatus(
                provider=ProviderName.kaggle,
                state=AuthState.authenticated,
                method=AuthMethod.mock,
                is_mock=True,
                message="TEST FIXTURE: token accepted",
                details={"fixture": True},
            )

    real = account_health(account, _Real(), settings=settings)
    assert real["enabled"] is True
    assert real["credential"]["source"] == "env:KAGGLE_ALT_API_TOKEN"
    assert real["credential"]["state"] == "missing"
    assert "auth_status" not in real

    mock = account_health(account, _Mock(), settings=settings)
    assert mock["auth_status"]["is_mock"] is True
    assert mock["auth_status"]["state"] == "authenticated"
    assert "TEST FIXTURE" in mock["auth_status"]["message"]
