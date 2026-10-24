"""Structured errors redact secrets and expose stable codes."""

import json

import pytest
from pydantic import ValidationError

from link2kaggle_colab.core.errors import (
    AccountDisabledError,
    AccountNotFoundError,
    AuthError,
    AuthExpiredError,
    AuthInvalidError,
    AuthMissingError,
    ConcurrentUpdateError,
    ConfigError,
    CredentialRefError,
    InvalidInputError,
    InvalidTransitionError,
    JobFailedError,
    JobNotFoundError,
    L2KCError,
    NotSupportedError,
    PathSecurityError,
    PermissionDeniedError,
    PremiumNotAllowed,
    ProviderCLIError,
    QuotaExceededError,
    RealModeDisabledError,
    ResourceUnavailableError,
    RuntimeDisconnectedError,
    SubmitOutcomeUnknownError,
    TransientProviderError,
)

_CLASSES = [
    (L2KCError, "l2kc_error"),
    (ConfigError, "config_error"),
    (AuthError, "auth_error"),
    (AuthMissingError, "auth_missing"),
    (AuthInvalidError, "auth_invalid"),
    (AuthExpiredError, "auth_expired"),
    (PermissionDeniedError, "permission_denied"),
    (QuotaExceededError, "quota_exceeded"),
    (ResourceUnavailableError, "resource_unavailable"),
    (PremiumNotAllowed, "premium_not_allowed"),
    (NotSupportedError, "not_supported"),
    (RealModeDisabledError, "real_mode_disabled"),
    (JobNotFoundError, "job_not_found"),
    (JobFailedError, "job_failed"),
    (RuntimeDisconnectedError, "runtime_disconnected"),
    (InvalidInputError, "invalid_input"),
    (InvalidTransitionError, "invalid_transition"),
    (ConcurrentUpdateError, "concurrent_update"),
    (PathSecurityError, "path_security"),
    (TransientProviderError, "transient_error"),
    (SubmitOutcomeUnknownError, "submit_outcome_unknown"),
    (AccountNotFoundError, "account_not_found"),
    (AccountDisabledError, "account_disabled"),
    (CredentialRefError, "credential_ref_error"),
]


@pytest.mark.parametrize(("cls", "code"), _CLASSES)
def test_default_codes_and_redacted_dict(cls: type[L2KCError], code: str) -> None:
    marker = "KGAT_" + ("n" * 24)
    access = "ya29." + ("q" * 20)
    refresh = "1//" + ("r" * 16)
    error = cls(
        f"failed {marker}",
        provider="kaggle",
        details={"token": access, "nested": {"key": refresh}, "ok": "visible"},
    )
    assert error.code == code
    assert isinstance(error, L2KCError)
    payload = json.dumps(error.to_dict())
    assert marker not in payload
    assert marker not in str(error)
    assert access not in payload
    assert refresh not in payload
    assert "visible" in payload
    assert error.to_dict()["provider"] == "kaggle"
    assert error.to_dict()["code"] == code


def test_permission_denied_is_an_auth_error() -> None:
    assert issubclass(PermissionDeniedError, AuthError)
    assert not issubclass(InvalidInputError, ValidationError)


def test_provider_cli_error_redacts_stderr() -> None:
    raw = "token=" + ("a" * 40)
    error = ProviderCLIError("cli failed", returncode=2, stderr_excerpt=raw, provider="colab")
    assert error.code == "provider_cli_error"
    assert error.returncode == 2
    assert error.stderr_excerpt is not None
    assert "a" * 40 not in error.stderr_excerpt
    payload = json.dumps(error.to_dict())
    assert "a" * 40 not in payload
    assert error.to_dict()["returncode"] == 2
    assert error.to_dict()["stderr_excerpt"] == error.stderr_excerpt


def test_custom_code_overrides_the_default() -> None:
    error = ConfigError("bad", code="custom_code")
    assert error.code == "custom_code"
