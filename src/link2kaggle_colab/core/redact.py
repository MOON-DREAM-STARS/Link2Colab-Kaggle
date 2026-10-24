"""Mask credential material before it reaches errors or logs."""

from __future__ import annotations

import re
from collections.abc import Mapping

_REDACTED = "[REDACTED]"

# Kaggle API tokens are prefixed with KGAT_.
_KGAT = re.compile(r"KGAT_[A-Za-z0-9_-]+")

# Google OAuth access tokens.
_YA29 = re.compile(r"ya29\.[A-Za-z0-9._-]+")

# Google refresh tokens commonly start with 1//.
_REFRESH = re.compile(r"1//[A-Za-z0-9._-]{8,}")

# JSON string fields whose names carry secrets. Short labels such as "T4" stay.
_JSON_FIELD = re.compile(
    r"(?i)(\"(?:access_token|refresh_token|id_token|api_key|apikey|token|secret|password|key)\""
    r"\s*:\s*\")([^\"\\]{8,})(\")"
)

# key= / token: assignments. Only 32+ hex/base64-ish values are masked.
# Bare token/key/secret/password stay on this rule so a 31-character label is kept.
_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:access_token|refresh_token|api_key|apikey|token|secret|password|key)\b"
    r"\s*[=:]\s*)([A-Za-z0-9_+\-/=]{32,})"
)

# Env-style NAME=value / NAME: value. `_` is a word character, so the rule above
# misses KAGGLE_KEY, HF_TOKEN, DB_PASSWORD, and AWS_SECRET_ACCESS_KEY.
# Values shorter than 8 stay (`gpu: T4`, `key=short`). `idempotency_key` is an id.
_ENV_ASSIGNMENT = re.compile(
    r"(?i)"
    r"(?<![A-Za-z0-9_])"
    r"(?!idempotency_key(?![A-Za-z0-9_]))"
    r"([A-Za-z0-9_]*_(?:API_KEY|ACCESS_KEY|PASSWORD|PASSWD|TOKEN|SECRET|KEY)|apikey)"
    r"(?![A-Za-z0-9_])"
    r"(\s*[=:]\s*[\"']?)"
    r"([^\s\"']{8,})"
)

_BEARER = re.compile(r"(?i)((?:authorization|x-api-key|api-key)\s*:\s*(?:bearer\s+)?)(\S{8,})")

# Mapping keys whose values are secrets even when the value has no token shape.
# Bare `key` is handled separately and only when the string is at least 16 characters.
# `idempotency_key` is not a secret.
_SECRET_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "private_key",
        "kaggle_key",
        "credentials",
        "credential",
        "authorization",
    }
)
_SECRET_KEY_SUFFIXES = ("_token", "_secret", "_password", "_api_key", "_private_key")


def redact(text: str) -> str:
    """Return ``text`` with tokens, refresh tokens, and secret fields masked."""
    if not text:
        return text
    masked = _KGAT.sub(_REDACTED, text)
    masked = _YA29.sub(_REDACTED, masked)
    masked = _REFRESH.sub(_REDACTED, masked)
    masked = _JSON_FIELD.sub(rf"\1{_REDACTED}\3", masked)
    masked = _ASSIGNMENT.sub(rf"\1{_REDACTED}", masked)
    masked = _ENV_ASSIGNMENT.sub(rf"\1\2{_REDACTED}", masked)
    return _BEARER.sub(rf"\1{_REDACTED}", masked)


def _is_secret_key(key: str) -> bool:
    folded = key.casefold()
    if folded in {"idempotency_key", "key"}:
        return False
    if folded in _SECRET_KEYS:
        return True
    return folded.endswith(_SECRET_KEY_SUFFIXES)


def _redact_secret_tree(value: object) -> object:
    """Mask every string leaf. Numbers, booleans, and None stay."""
    if isinstance(value, str):
        if not value:
            return value
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(key): _redact_secret_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secret_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secret_tree(item) for item in value)
    return value


def redact_value(value: object) -> object:
    """Recursively redact strings inside mappings and sequences.

    A mapping key that names a secret replaces a non-empty string with
    ``[REDACTED]`` and masks every string leaf of a nested dict or list.
    Bare ``key`` is redacted only when its value is a string of length 16
    or more. ``idempotency_key`` is left to the value patterns.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if isinstance(key, str) and _is_secret_key(key):
                cleaned[name] = _redact_secret_tree(item)
            elif (
                isinstance(key, str)
                and key.casefold() == "key"
                and isinstance(item, str)
                and len(item) >= 16
            ):
                cleaned[name] = _REDACTED
            else:
                cleaned[name] = redact_value(item)
        return cleaned
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    return value
