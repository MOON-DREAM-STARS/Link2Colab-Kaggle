"""Account rows and credential references stored in config.

String fields are references or labels. A value that looks like a token, key,
or PEM block is rejected. Error text names the field and never includes the value.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from link2kaggle_colab.core.errors import ConfigError
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.core.redact import redact

_ACCOUNT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_REF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_HEX_RUN = re.compile(r"[0-9a-fA-F]{32,}")
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
_PEM = re.compile(r"-----BEGIN [A-Z0-9 ]+-----")
_GITHUB = re.compile(r"(?:^|[^A-Za-z0-9])ghp_[A-Za-z0-9]")
_SK = re.compile(r"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9]")
_KGAT = re.compile(r"KGAT_")
_YA29 = re.compile(r"ya29\.")
_JSON_USER = re.compile(r"""["']username["']\s*:""")
_JSON_KEY = re.compile(r"""["']key["']\s*:""")


def _json_username_and_key(value: str) -> bool:
    text = value.strip()
    if not text.startswith("{") or "username" not in text or "key" not in text:
        return False
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return _JSON_USER.search(text) is not None and _JSON_KEY.search(text) is not None
    return isinstance(loaded, dict) and "username" in loaded and "key" in loaded


def looks_like_literal_secret(value: str) -> bool:
    """True when ``value`` is token-shaped or :func:`redact` would change it."""
    if redact(value) != value:
        return True
    if _HEX_RUN.search(value) or _JWT.search(value) or _PEM.search(value):
        return True
    if _GITHUB.search(value) or _SK.search(value) or _KGAT.search(value):
        return True
    if _YA29.search(value):
        return True
    return _json_username_and_key(value)


def reject_literal_secret(value: str, field: str) -> str:
    """Return ``value`` or raise ``ConfigError`` naming ``field`` only."""
    if looks_like_literal_secret(value):
        raise ConfigError(
            f"{field} looks like a secret; store a reference, not the secret",
            details={"field": field},
        )
    return value


def _reject_controls(value: str, field: str) -> None:
    if any(char in value for char in "\n\r\x00"):
        raise ConfigError(
            f"{field} contains a forbidden character",
            details={"field": field},
        )


def reject_account_id(value: str) -> str:
    """Require ``^[a-z0-9][a-z0-9-]{0,62}$`` without echoing ``value``."""
    reject_literal_secret(value, "id")
    if _ACCOUNT_ID.fullmatch(value) is None:
        raise ConfigError(
            "id must be a lowercase account id",
            details={"field": "id"},
        )
    return value


def reject_ref_name(value: str, field: str) -> str:
    """Require one path segment that is not a literal secret."""
    reject_literal_secret(value, field)
    if _REF_NAME.fullmatch(value) is None:
        raise ConfigError(
            f"{field} must be a credential reference name",
            details={"field": field},
        )
    return value


def reject_env_name(value: str, field: str) -> str:
    """Require ``^[A-Z_][A-Z0-9_]*$`` after the literal-secret check."""
    reject_literal_secret(value, field)
    if _ENV_NAME.fullmatch(value) is None:
        raise ConfigError(
            f"{field} must be an environment variable name",
            details={"field": field},
        )
    return value


def reject_path_text(value: str, field: str) -> str:
    """Reject secrets and newlines. The path itself is not echoed."""
    reject_literal_secret(value, field)
    if value.strip() == "" or any(char in value for char in "\n\r\x00"):
        raise ConfigError(
            f"{field} must be a path",
            details={"field": field},
        )
    return value


def reject_plain_text(value: str, field: str) -> str:
    """Reject secrets, control characters, and empty strings."""
    reject_literal_secret(value, field)
    _reject_controls(value, field)
    if value.strip() == "":
        raise ConfigError(
            f"{field} must not be empty",
            details={"field": field},
        )
    return value


class AccountConfig(BaseModel):
    """One ``accounts:`` row. ``credential_ref`` is a name, never a secret."""

    model_config = ConfigDict(extra="forbid")

    id: str
    provider: ProviderName
    credential_ref: str | None = None
    enabled: bool = True
    default: bool = False
    labels: dict[str, str] = Field(default_factory=dict)
    mock_scenario: str | None = None

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        return reject_account_id(value)

    @field_validator("credential_ref")
    @classmethod
    def _ref_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return reject_ref_name(value, "credential_ref")

    @field_validator("labels")
    @classmethod
    def _labels_are_not_secrets(cls, value: dict[str, str]) -> dict[str, str]:
        for key, item in value.items():
            reject_plain_text(key, "labels")
            reject_plain_text(item, "labels")
        return value

    @field_validator("mock_scenario")
    @classmethod
    def _scenario_is_not_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return reject_plain_text(value, "mock_scenario")


class KaggleConfigDirCredential(BaseModel):
    """Kaggle config directory. The child receives ``KAGGLE_CONFIG_DIR``."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["kaggle_config_dir"]
    path: str

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        return reject_path_text(value, "path")


class EnvCredential(BaseModel):
    """Environment variable name. The value is read only when resolving."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["env"]
    env: str

    @field_validator("env")
    @classmethod
    def _env(cls, value: str) -> str:
        return reject_env_name(value, "env")


class AdcFileCredential(BaseModel):
    """Application Default Credentials file."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["adc_file"]
    path: str

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        return reject_path_text(value, "path")


class ColabHomeCredential(BaseModel):
    """Directory used as the child ``HOME``. The token file is not opened here."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["colab_home"]
    path: str

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        return reject_path_text(value, "path")


class KeyringCredential(BaseModel):
    """System keyring entry. Resolution needs the optional ``keyring`` package."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["keyring"]
    service: str
    username: str

    @field_validator("service")
    @classmethod
    def _service(cls, value: str) -> str:
        return reject_plain_text(value, "service")

    @field_validator("username")
    @classmethod
    def _username(cls, value: str) -> str:
        return reject_plain_text(value, "username")


CredentialRefConfig = Annotated[
    KaggleConfigDirCredential
    | EnvCredential
    | AdcFileCredential
    | ColabHomeCredential
    | KeyringCredential,
    Field(discriminator="source"),
]


class Account(BaseModel):
    """Resolved account. ``implicit`` marks a legacy default synthesized at runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: ProviderName
    credential_ref: str | None = None
    enabled: bool = True
    is_default: bool = False
    labels: dict[str, str] = Field(default_factory=dict)
    mock_scenario: str | None = None
    implicit: bool = False
