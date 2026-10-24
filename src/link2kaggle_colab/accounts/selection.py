"""Account selection.

Selection is deterministic. There is no automatic account rotation: never
switch to another account of the same provider because of quota exhaustion,
rate limits, auth errors, or provider errors. :func:`select_account` never
consults quota.
"""

from __future__ import annotations

from link2kaggle_colab.accounts.models import Account
from link2kaggle_colab.accounts.registry import AccountRegistry
from link2kaggle_colab.core.errors import (
    AccountDisabledError,
    AccountNotFoundError,
    InvalidInputError,
)
from link2kaggle_colab.core.models import ProviderName


def select_account(
    registry: AccountRegistry,
    provider: ProviderName | str | None = None,
    explicit: str | None = None,
) -> Account:
    """Pick the explicit account or the provider default.

    An explicit id must exist, match ``provider`` when one is given, and be
    enabled. Otherwise the provider default is used. A disabled default raises
    :class:`AccountDisabledError` and is not replaced by another account.
    Quota, rate limits, and provider errors are not inputs.
    """
    if explicit is not None:
        account = registry.get(explicit)
        if provider is not None and account.provider is not _as_provider(provider):
            raise AccountNotFoundError(
                "account provider does not match",
                account_id=account.id,
                provider=account.provider.value,
                details={"reason": "provider_mismatch"},
            )
        if not account.enabled:
            raise AccountDisabledError(
                "account is disabled",
                account_id=account.id,
                provider=account.provider.value,
            )
        return account
    if provider is None:
        raise InvalidInputError(
            "provider is required to select an account",
            details={"field": "provider"},
        )
    return registry.default_for(provider)


def _as_provider(value: ProviderName | str) -> ProviderName:
    if isinstance(value, ProviderName):
        return value
    try:
        return ProviderName(str(value))
    except ValueError:
        raise InvalidInputError(
            "unknown provider",
            details={"field": "provider"},
        ) from None
