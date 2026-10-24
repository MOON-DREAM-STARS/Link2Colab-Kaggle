"""Account health. Real mode checks credential presence and does not run a CLI."""

from __future__ import annotations

from typing import Any

from link2kaggle_colab.accounts.credentials import credential_health
from link2kaggle_colab.accounts.models import Account
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.models import AuthStatus


def account_health(
    account: Account,
    provider_obj: object | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Return enabled state, credential presence, and mock auth when applicable.

    A mock provider (``is_mock`` is true) contributes ``auth_status``. A real
    provider is not called, so this never runs ``kaggle`` or ``colab``.
    Credential health is presence only.
    """
    if settings is None and provider_obj is not None:
        candidate = getattr(provider_obj, "_settings", None)
        if isinstance(candidate, Settings):
            settings = candidate
    if settings is None:
        credential: dict[str, str] = {"state": "missing", "source": "unconfigured"}
    else:
        ref_name = account.credential_ref
        ref_config = settings.credentials.get(ref_name) if ref_name is not None else None
        credential = credential_health(ref_name, ref_config, account.provider, settings)
    body: dict[str, Any] = {
        "account_id": account.id,
        "provider": account.provider.value,
        "enabled": account.enabled,
        "credential": credential,
    }
    if provider_obj is not None and bool(getattr(provider_obj, "is_mock", False)):
        status = provider_obj.auth_status()  # type: ignore[attr-defined]
        if isinstance(status, AuthStatus):
            body["auth_status"] = status.model_dump(mode="json")
        else:
            body["auth_status"] = status
    return body
