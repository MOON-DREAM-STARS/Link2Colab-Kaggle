"""Colab ccu-info JSONP parse and membership inference (display only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from link2kaggle_colab.accounts.identity import (
    infer_colab_membership_tier,
    write_identity,
)
from link2kaggle_colab.providers.colab.ccu_info import (
    parse_ccu_info,
    read_token_account,
    strip_jsonp_prefix,
)

# Verified live samples (balances/eligible only; no secrets).
_FREE_PRO_BODY = (
    ")]}'\n"
    '[0,0,0,["T4"],["H100","G4","A100","L4"],null,["V5E1"],["V6E1"]]'
)
_PAID_PRO_BODY = (
    ")]}'\n"
    '[199.99,0,0,["G4","A100","L4","T4"],["H100"],null,["V6E1","V5E1"]]'
)


def test_strip_jsonp_prefix() -> None:
    assert strip_jsonp_prefix(")]}'\n[1]") == "[1]"
    assert strip_jsonp_prefix("[1]") == "[1]"


def test_parse_ccu_info_free_pro_sample() -> None:
    balance, eligible = parse_ccu_info(_FREE_PRO_BODY)
    assert balance == 0.0
    assert eligible == ["T4"]
    assert infer_colab_membership_tier(balance, eligible) == "free_pro"


def test_parse_ccu_info_paid_pro_sample() -> None:
    balance, eligible = parse_ccu_info(_PAID_PRO_BODY)
    assert balance == pytest.approx(199.99)
    assert eligible == ["G4", "A100", "L4", "T4"]
    assert infer_colab_membership_tier(balance, eligible) == "paid_pro"


def test_infer_paid_exhausted_when_zero_with_broad_eligible() -> None:
    assert (
        infer_colab_membership_tier(0.0, ["G4", "A100", "L4", "T4"]) == "paid_exhausted"
    )


def test_infer_unknown_when_zero_without_eligible() -> None:
    assert infer_colab_membership_tier(0.0, None) == "unknown"


def test_read_token_account_email_only(tmp_path: Path) -> None:
    path = tmp_path / "token.json"
    path.write_text(
        json.dumps(
            {
                "account": "user@example.com",
                "token": "ya29.secret-should-not-leak",
                "refresh_token": "1//secret",
                "client_secret": "shhh",
            }
        ),
        encoding="utf-8",
    )
    bearer, email = read_token_account(path)
    assert bearer == "ya29.secret-should-not-leak"
    assert email == "user@example.com"


def test_identity_json_omits_secrets(tmp_path: Path) -> None:
    path = tmp_path / "accounts" / "colab-a" / "identity.json"
    write_identity(
        path,
        {
            "provider": "colab",
            "email": "user@example.com",
            "membership_tier": "free_pro",
            "eligible_gpus": ["T4"],
            "ccu_remaining": 0.0,
            "token": "ya29.must-not-appear",
            "refresh_token": "1//must-not-appear",
            "client_secret": "nope",
            "source": "ccu-info",
            "refreshed_at": "2026-10-24T00:00:00Z",
        },
    )
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["email"] == "user@example.com"
    assert data["membership_tier"] == "free_pro"
    assert data["eligible_gpus"] == ["T4"]
    assert "token" not in data
    assert "refresh_token" not in data
    assert "client_secret" not in data
    assert "ya29" not in raw
    assert "must-not-appear" not in raw


def test_parse_ccu_info_rejects_non_numeric_balance() -> None:
    with pytest.raises(ValueError, match="balance is not numeric"):
        parse_ccu_info(")]}'\n" + '["x",0,0,["T4"]]')
    with pytest.raises(ValueError, match="balance is not numeric"):
        parse_ccu_info(")]}'\n" + '[true,0,0,["T4"]]')


def test_parse_ccu_info_rejects_non_gpu_eligible_tokens() -> None:
    with pytest.raises(ValueError, match="GPU-like"):
        parse_ccu_info(
            ")]}'\n"
            + '[0,0,0,["T4","http://evil.example/long-token-that-is-not-a-gpu"]]'
        )
