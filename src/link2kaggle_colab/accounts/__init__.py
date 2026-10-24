"""Account records and credential references.

Imports are lazy so :mod:`link2kaggle_colab.core.config` can load
:mod:`link2kaggle_colab.accounts.models` without cycling through the registry.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Account",
    "AccountConfig",
    "AccountEnv",
    "AccountRegistry",
    "AdcFileCredential",
    "ColabHomeCredential",
    "CredentialRefConfig",
    "EnvCredential",
    "KaggleConfigDirCredential",
    "KeyringCredential",
    "ResolvedCredential",
    "account_health",
    "credential_health",
    "resolve_credential",
    "select_account",
]

_LAZY: dict[str, tuple[str, str]] = {
    "Account": ("link2kaggle_colab.accounts.models", "Account"),
    "AccountConfig": ("link2kaggle_colab.accounts.models", "AccountConfig"),
    "AdcFileCredential": ("link2kaggle_colab.accounts.models", "AdcFileCredential"),
    "ColabHomeCredential": ("link2kaggle_colab.accounts.models", "ColabHomeCredential"),
    "CredentialRefConfig": ("link2kaggle_colab.accounts.models", "CredentialRefConfig"),
    "EnvCredential": ("link2kaggle_colab.accounts.models", "EnvCredential"),
    "KaggleConfigDirCredential": ("link2kaggle_colab.accounts.models", "KaggleConfigDirCredential"),
    "KeyringCredential": ("link2kaggle_colab.accounts.models", "KeyringCredential"),
    "AccountEnv": ("link2kaggle_colab.core.envspec", "AccountEnv"),
    "ResolvedCredential": ("link2kaggle_colab.accounts.credentials", "ResolvedCredential"),
    "credential_health": ("link2kaggle_colab.accounts.credentials", "credential_health"),
    "resolve_credential": ("link2kaggle_colab.accounts.credentials", "resolve_credential"),
    "AccountRegistry": ("link2kaggle_colab.accounts.registry", "AccountRegistry"),
    "select_account": ("link2kaggle_colab.accounts.selection", "select_account"),
    "account_health": ("link2kaggle_colab.accounts.health", "account_health"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
