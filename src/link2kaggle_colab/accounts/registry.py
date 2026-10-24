"""Account registry. Defaults do not fall through to another account."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from link2kaggle_colab.accounts.models import Account, AccountConfig
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import (
    AccountDisabledError,
    AccountNotFoundError,
    ConfigError,
    InvalidInputError,
)
from link2kaggle_colab.core.models import ProviderName

_STATE_NAME = "accounts_state.json"
_IMPLICIT_ORDER = (ProviderName.kaggle, ProviderName.colab, ProviderName.fake)


def _provider(value: ProviderName | str) -> ProviderName:
    if isinstance(value, ProviderName):
        return value
    try:
        return ProviderName(str(value))
    except ValueError:
        raise InvalidInputError(
            "unknown provider",
            details={"field": "provider"},
        ) from None


def _implicit_accounts(settings: Settings) -> list[Account]:
    accounts: list[Account] = []
    for provider in _IMPLICIT_ORDER:
        enabled = bool(getattr(settings.providers, provider.value).enabled)
        accounts.append(
            Account(
                id=f"{provider.value}-default",
                provider=provider,
                credential_ref=None,
                enabled=enabled,
                is_default=True,
                labels={},
                mock_scenario=None,
                implicit=True,
            )
        )
    return accounts


def _provider_enabled(settings: Settings, provider: ProviderName) -> bool:
    return bool(getattr(settings.providers, provider.value).enabled)


def _mask_disabled_providers(settings: Settings, accounts: list[Account]) -> list[Account]:
    """A disabled provider keeps every one of its accounts disabled."""
    masked: list[Account] = []
    for account in accounts:
        if account.enabled and not _provider_enabled(settings, account.provider):
            masked.append(account.model_copy(update={"enabled": False}))
        else:
            masked.append(account)
    return masked


def _configured_accounts(configs: list[AccountConfig], settings: Settings) -> list[Account]:
    grouped: dict[ProviderName, list[AccountConfig]] = {}
    for config in configs:
        grouped.setdefault(config.provider, []).append(config)
    chosen: set[str] = set()
    for rows in grouped.values():
        flagged = [row for row in rows if row.default]
        chosen.add((flagged[0] if flagged else rows[0]).id)
    accounts: list[Account] = []
    for config in configs:
        provider_on = _provider_enabled(settings, config.provider)
        accounts.append(
            Account(
                id=config.id,
                provider=config.provider,
                credential_ref=config.credential_ref,
                enabled=config.enabled and provider_on,
                is_default=config.id in chosen,
                labels=dict(config.labels),
                mock_scenario=config.mock_scenario,
                implicit=False,
            )
        )
    present = {account.provider for account in accounts}
    for provider in _IMPLICIT_ORDER:
        if provider in present:
            continue
        accounts.append(
            Account(
                id=f"{provider.value}-default",
                provider=provider,
                credential_ref=None,
                enabled=_provider_enabled(settings, provider),
                is_default=True,
                labels={},
                mock_scenario=None,
                implicit=True,
            )
        )
    return accounts


def _string_list(value: object, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(
            "accounts state file is invalid",
            details={"field": field},
        )
    return list(value)


class AccountRegistry:
    """Accounts from settings, with runtime enable and disable overrides.

    When ``settings.accounts`` is None, the registry synthesizes
    ``kaggle-default``, ``colab-default``, and ``fake-default``. Each is
    ambient, implicit, and enabled only when that provider is enabled.

    When ``accounts`` is set, a provider with no row still gets that implicit
    ambient account, enabled only when the provider is enabled. A configured
    account whose provider is disabled is disabled too.

    ``default_for`` returns the account flagged ``default: true``, or the
    first listed account for that provider. A disabled result is an error.
    Another enabled account of the same provider is not substituted.
    """

    def __init__(self, settings: Settings, accounts: list[Account]) -> None:
        self._settings = settings
        self._accounts = list(accounts)
        self._state_path = Path(settings.state_dir) / _STATE_NAME
        self._reindex()

    @classmethod
    def from_settings(cls, settings: Settings) -> AccountRegistry:
        """Build a registry and apply ``state_dir/accounts_state.json`` when present."""
        if settings.accounts is None:
            accounts = _implicit_accounts(settings)
        else:
            accounts = _configured_accounts(settings.accounts, settings)
        registry = cls(settings, accounts)
        registry._apply_saved_overrides()
        registry._accounts = _mask_disabled_providers(settings, registry._accounts)
        registry._reindex()
        return registry

    def _reindex(self) -> None:
        self._by_id = {account.id: account for account in self._accounts}

    def list(self, provider: ProviderName | str | None = None) -> list[Account]:
        """Return accounts in config order, optionally filtered by provider."""
        if provider is None:
            return list(self._accounts)
        name = _provider(provider)
        return [account for account in self._accounts if account.provider is name]

    def get(self, account_id: str) -> Account:
        """Return one account or raise :class:`AccountNotFoundError`."""
        found = self._by_id.get(account_id)
        if found is None:
            raise AccountNotFoundError(
                "account not found",
                account_id=account_id,
            )
        return found

    def default_for(self, provider: ProviderName | str) -> Account:
        """Return the provider default. Never falls through to another account."""
        name = _provider(provider)
        chosen = next(
            (
                account
                for account in self._accounts
                if account.provider is name and account.is_default
            ),
            None,
        )
        if chosen is None:
            raise AccountNotFoundError(
                "no account is configured for this provider",
                account_id=f"{name.value}-default",
                provider=name.value,
            )
        if not chosen.enabled:
            raise AccountDisabledError(
                "account is disabled",
                account_id=chosen.id,
                provider=name.value,
            )
        return chosen

    def resolve_for_record(self, account_id: str, provider: ProviderName | str) -> Account:
        """Resolve a stored job's account id.

        An exact id wins. A legacy ``<provider>-default`` id that is not
        configured maps to :meth:`default_for`. Any other id is missing.
        The account's provider must match ``provider``.
        """
        name = _provider(provider)
        found = self._by_id.get(account_id)
        if found is not None:
            if found.provider is not name:
                raise AccountNotFoundError(
                    "account provider does not match",
                    account_id=account_id,
                    provider=name.value,
                    details={"reason": "provider_mismatch"},
                )
            return found
        if account_id == f"{name.value}-default":
            return self.default_for(name)
        raise AccountNotFoundError(
            "account not found",
            account_id=account_id,
            provider=name.value,
        )

    def override_disabled(self, account_id: str) -> bool:
        """True when ``accounts_state.json`` lists ``account_id`` as disabled."""
        disabled, _enabled_ids = self._read_override_sets()
        return account_id in disabled

    def set_enabled(self, account_id: str, enabled: bool) -> None:
        """Persist an enable or disable override and update this registry.

        The only file written is ``state_dir/accounts_state.json``.
        """
        current = self.get(account_id)
        disabled, enabled_ids = self._read_override_sets()
        if enabled:
            enabled_ids.add(account_id)
            disabled.discard(account_id)
        else:
            disabled.add(account_id)
            enabled_ids.discard(account_id)
        self._write_state(disabled, enabled_ids)
        updated = current.model_copy(update={"enabled": enabled})
        self._accounts = [
            updated if account.id == account_id else account for account in self._accounts
        ]
        self._accounts = _mask_disabled_providers(self._settings, self._accounts)
        self._reindex()

    def _apply_saved_overrides(self) -> None:
        disabled, enabled_ids = self._read_override_sets()
        if not disabled and not enabled_ids:
            return
        updated: list[Account] = []
        for account in self._accounts:
            flag = account.enabled
            if account.id in enabled_ids:
                flag = True
            if account.id in disabled:
                flag = False
            if flag == account.enabled:
                updated.append(account)
            else:
                updated.append(account.model_copy(update={"enabled": flag}))
        self._accounts = updated
        self._reindex()

    def _read_override_sets(self) -> tuple[set[str], set[str]]:
        path = self._state_path
        if not path.is_file():
            return set(), set()
        try:
            text = path.read_text(encoding="utf-8")
            loaded = json.loads(text)
        except (OSError, json.JSONDecodeError):
            raise ConfigError(
                "accounts state file is invalid",
                details={"field": "accounts_state"},
            ) from None
        if not isinstance(loaded, dict):
            raise ConfigError(
                "accounts state file is invalid",
                details={"field": "accounts_state"},
            )
        disabled = set(_string_list(loaded.get("disabled"), "disabled"))
        enabled_ids = set(_string_list(loaded.get("enabled"), "enabled"))
        return disabled, enabled_ids

    def _write_state(self, disabled: set[str], enabled_ids: set[str]) -> None:
        path = self._state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "disabled": sorted(disabled),
            "enabled": sorted(enabled_ids),
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=".accounts_state.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
