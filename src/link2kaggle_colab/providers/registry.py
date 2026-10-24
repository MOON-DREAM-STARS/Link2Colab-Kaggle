"""Build a provider from settings.

Built-in names (``kaggle``, ``colab``, ``fake``) always use the built-in
classes. :func:`register_provider_factory` ignores those names. Entry-point
factories load only when :func:`load_entry_point_factories` is called; importing
this module does not scan entry points. ``ProviderName`` stays closed, so a
plugin name is not accepted by config.
"""

from __future__ import annotations

from typing import Protocol

from link2kaggle_colab.accounts.credentials import resolve_credential
from link2kaggle_colab.accounts.models import Account
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.envspec import AccountEnv
from link2kaggle_colab.core.errors import ConfigError
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.core.paths import safe_join
from link2kaggle_colab.core.process import CommandRunner
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.providers.colab.cli_provider import ColabCliProvider
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.fake.provider import FakeProvider
from link2kaggle_colab.providers.kaggle.cli_provider import KaggleCliProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider

_BUILTIN = frozenset(item.value for item in ProviderName)
_ENTRY_GROUP = "link2kaggle_colab.providers"


class ProviderFactory(Protocol):
    """Build one provider. ``account`` and ``runner`` may be None."""

    def __call__(
        self,
        settings: Settings,
        account: Account | None,
        runner: CommandRunner | None,
    ) -> Provider: ...


_FACTORIES: dict[str, ProviderFactory] = {}


def register_provider_factory(name: str, factory: ProviderFactory) -> None:
    """Register ``factory`` for ``name``.

    Built-in names are ignored so a plugin cannot replace Kaggle, Colab, or
    fake. A name that is already registered is left as it is.
    """
    if name in _BUILTIN or name in _FACTORIES:
        return
    _FACTORIES[name] = factory


def load_entry_point_factories(group: str = _ENTRY_GROUP) -> None:
    """Load provider factories from ``group``. Not called on import.

    Built-in names and names already registered are skipped. A failing entry
    point is not swallowed: the caller asked to load plugins.
    """
    from importlib.metadata import entry_points

    for entry in entry_points(group=group):
        register_provider_factory(entry.name, entry.load())


def build_provider(
    name: ProviderName | str,
    settings: Settings,
    *,
    runner: CommandRunner | None = None,
    account: Account | None = None,
) -> Provider:
    """Return the provider for ``name``.

    Mock mode selects the in-memory Kaggle and Colab providers. Real mode
    selects the official CLI adapters. ``fake`` works in both modes. ``runner``
    is forwarded to the real adapters and ignored otherwise.

    ``account`` selects that account's credential overlay and, when the account
    is not an implicit default, a state directory under
    ``state_dir/accounts/<account_id>/``. Implicit defaults keep today's paths.
    ``account.mock_scenario`` overrides ``settings.mock`` for that account.
    Real CLI adapters receive the account environment on every runner call.
    Mock mode does not resolve credentials.
    """
    provider_name = _coerce_name(name)
    if provider_name is None:
        factory = _FACTORIES.get(str(name))
        if factory is None:
            raise ConfigError(
                f"unknown provider {name!s}",
                details={"provider": str(name)},
            )
        return factory(settings, account, runner)
    if account is not None and account.provider is not provider_name:
        raise ConfigError(
            "account provider does not match",
            details={
                "field": "account",
                "account_id": account.id,
                "provider": provider_name.value,
            },
        )
    state_dir = None
    account_id = None
    scenario = None
    if account is not None:
        account_id = account.id
        if account.mock_scenario:
            scenario = account.mock_scenario
        if not account.implicit:
            state_dir = safe_join(settings.state_dir, "accounts", account.id)
    if provider_name is ProviderName.fake:
        return FakeProvider(settings, state_dir=state_dir, account_id=account_id)
    if settings.mode == "mock":
        if provider_name is ProviderName.kaggle:
            return MockKaggleProvider(
                settings,
                scenario=scenario,
                state_dir=state_dir,
                account_id=account_id,
            )
        return MockColabProvider(
            settings,
            scenario=scenario,
            state_dir=state_dir,
            account_id=account_id,
        )
    account_env = _real_account_env(settings, account, provider_name)
    if provider_name is ProviderName.kaggle:
        return KaggleCliProvider(
            settings,
            runner=runner,
            account_env=account_env,
            state_dir=state_dir,
            account_id=account_id,
        )
    return ColabCliProvider(
        settings,
        runner=runner,
        account_env=account_env,
        state_dir=state_dir,
        account_id=account_id,
    )


def _coerce_name(name: ProviderName | str) -> ProviderName | None:
    if isinstance(name, ProviderName):
        return name
    try:
        return ProviderName(str(name))
    except ValueError:
        return None


def _real_account_env(
    settings: Settings,
    account: Account | None,
    provider: ProviderName,
) -> AccountEnv | None:
    """Credential overlay for a real CLI adapter. None when no account is bound."""
    if account is None or provider is ProviderName.fake:
        return None
    ref_name = account.credential_ref
    ref_config = settings.credentials.get(ref_name) if ref_name else None
    return resolve_credential(ref_name, ref_config, provider, settings).env
