"""Non-secret account identity cache under ``state_dir/accounts/<id>/``.

``membership_tier`` is for display only. Schedulers must never read it to
accept or reject accelerators — use live ``eligible_gpus`` instead.
``free_pro`` is not the same as paid Pro.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Display/cache labels only. free_pro ≠ paid_pro.
MEMBERSHIP_TIERS = frozenset(
    {"free_pro", "paid_pro", "paid_exhausted", "free", "unknown"}
)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with a ``Z`` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def identity_path(state_dir: Path, account_id: str) -> Path:
    """``state_dir/accounts/<account_id>/identity.json``."""
    return Path(state_dir).expanduser().resolve() / "accounts" / account_id / "identity.json"


def write_identity(path: Path, payload: dict[str, Any]) -> None:
    """Write a non-secret identity document. Never stores tokens or keys."""
    forbidden = {
        "token",
        "refresh_token",
        "access_token",
        "client_secret",
        "api_token",
        "key",
        "password",
    }
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in forbidden or "token" in key.lower() or "secret" in key.lower():
            continue
        cleaned[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def infer_colab_membership_tier(
    balance: float,
    eligible_gpus: list[str] | None,
) -> str:
    """Infer a display-only Colab membership label from balance + eligibility.

    This is not a scheduler input. ``free_pro`` means a free Pro-shaped account
    (typically 0 CCU with only T4 eligible), not a paid subscription.
    """
    if balance > 0:
        return "paid_pro"
    if eligible_gpus is None:
        return "unknown"
    names = [str(item).strip().upper() for item in eligible_gpus if str(item).strip()]
    if names == ["T4"]:
        return "free_pro"
    if names:
        return "paid_exhausted"
    return "unknown"
