"""Parse Colab ``ccu-info`` JSONP and resolve the OAuth token path.

``ccu-info`` is an **undocumented** Colab endpoint. It is experimental and
opt-in only (see ``providers.colab.experimental_ccu_info`` /
``L2KC_COLAB_EXPERIMENTAL_CCU_INFO``). The default quota path is ``colab usage``.

Membership inferred here is for identity/display only. Schedulers must use
``eligible_gpus``, never ``membership_tier``. ``free_pro`` is not paid Pro.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CCU_INFO_URL = "https://colab.research.google.com/tun/m/ccu-info?authuser=0"
_JSONP_PREFIX = ")]}'"
# Short accelerator tokens seen in verified samples (T4, L4, A100, H100, G4,
# V5E1, V6E1, …). Reject long or non-token garbage from a malformed body.
_GPU_LIKE = re.compile(r"^[A-Za-z][A-Za-z0-9_+-]{0,11}$")


def strip_jsonp_prefix(body: str) -> str:
    """Remove the XSSI ``)]}'`` prefix Colab returns before the JSON array."""
    text = body.lstrip("\ufeff").strip()
    if text.startswith(_JSONP_PREFIX):
        text = text[len(_JSONP_PREFIX) :].lstrip()
    return text


def parse_ccu_info(body: str) -> tuple[float, list[str] | None]:
    """Return ``(current_balance, eligible_accelerators)`` from a ccu-info body.

    Layout (verified): index 0 = currentBalance, index 3 = eligible list,
    index 4 = other/premium list. Only index 3 is returned as eligible.
    Never invents an eligible list when the field is missing or null.

    Rejects a non-numeric balance and eligible items that are not short
    GPU-like tokens (``T4``, ``L4``, ``A100``, ``H100``, ``G4``, ``V5E1``, …).
    """
    data = json.loads(strip_jsonp_prefix(body))
    if not isinstance(data, list) or len(data) < 4:
        raise ValueError("ccu-info payload is not a list with balance + eligible")
    raw_balance = data[0]
    # bool is a subclass of int; reject it so True/False cannot become 1.0/0.0.
    if isinstance(raw_balance, bool) or not isinstance(raw_balance, (int, float)):
        raise ValueError("ccu-info balance is not numeric")
    balance = float(raw_balance)
    raw = data[3]
    if raw is None:
        return balance, None
    if not isinstance(raw, list):
        raise ValueError("ccu-info eligible field is not a list")
    names: list[str] = []
    for item in raw:
        text_item = str(item).strip()
        if not text_item:
            continue
        if _GPU_LIKE.fullmatch(text_item) is None:
            raise ValueError("ccu-info eligible item is not a GPU-like token")
        names.append(text_item)
    return balance, names or None


def read_token_account(token_path: Path) -> tuple[str | None, str | None]:
    """Return ``(bearer_token, email)`` from ``token.json``.

    Only the ``token`` and ``account`` fields are read. Secrets are never
    written to identity files by callers of this helper.
    """
    try:
        data = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    token = data.get("token")
    account = data.get("account")
    bearer = token.strip() if isinstance(token, str) and token.strip() else None
    email = account.strip() if isinstance(account, str) and account.strip() else None
    return bearer, email


def fetch_ccu_info_body(bearer_token: str, *, timeout: float = 30.0) -> str:
    """GET ccu-info with the Colab OAuth bearer. Raises on HTTP/network errors."""
    request = urllib.request.Request(
        CCU_INFO_URL,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json, text/plain, */*",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise ValueError(f"ccu-info HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"ccu-info network error: {exc.reason}") from exc
    return raw.decode("utf-8", errors="replace")


def token_json_path(home: Path) -> Path:
    """``<home>/.config/colab-cli/token.json``."""
    return Path(home).expanduser() / ".config" / "colab-cli" / "token.json"


def ccu_info_snapshot(home: Path, *, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch and parse ccu-info for ``home``. Raises ``ValueError`` on failure."""
    path = token_json_path(home)
    if not path.is_file():
        raise ValueError("colab token.json missing")
    bearer, email = read_token_account(path)
    if bearer is None:
        raise ValueError("colab token.json has no bearer token")
    body = fetch_ccu_info_body(bearer, timeout=timeout)
    balance, eligible = parse_ccu_info(body)
    return {
        "balance": balance,
        "eligible_gpus": eligible,
        "email": email,
        "source": "ccu-info",
        "raw_body": body,
    }
