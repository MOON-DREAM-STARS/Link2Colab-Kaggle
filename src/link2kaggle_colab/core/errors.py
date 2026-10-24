"""Structured errors. ``details`` and messages are redacted and never hold secrets."""

from __future__ import annotations

from typing import Any, ClassVar

from link2kaggle_colab.core.redact import redact, redact_value


class L2KCError(Exception):
    """Base error for Link2Kaggle&Colab.

    ``code`` is a stable machine identifier. ``details`` must not contain secrets;
    values are passed through :func:`redact_value` before they are stored.
    """

    default_code: ClassVar[str] = "l2kc_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_details = redact_value(dict(details or {}))
        if not isinstance(safe_details, dict):
            safe_details = {}
        self.code = code or self.default_code
        self.message = redact(message)
        self.provider = provider
        self.details: dict[str, Any] = safe_details
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready payload with secrets already removed."""
        details = redact_value(self.details)
        return {
            "code": self.code,
            "message": self.message,
            "provider": self.provider,
            "details": details if isinstance(details, dict) else {},
        }


class ConfigError(L2KCError):
    """Configuration is missing, invalid, or tries to store a secret."""

    default_code = "config_error"


class AuthError(L2KCError):
    """Authentication failed or cannot be used."""

    default_code = "auth_error"


class AuthMissingError(AuthError):
    """No credential source is configured."""

    default_code = "auth_missing"


class AuthInvalidError(AuthError):
    """A credential was present but rejected."""

    default_code = "auth_invalid"


class AuthExpiredError(AuthError):
    """A credential is expired."""

    default_code = "auth_expired"


class PermissionDeniedError(AuthError):
    """The account is authenticated but not allowed to perform the action."""

    default_code = "permission_denied"


class QuotaExceededError(L2KCError):
    """The account cannot spend more GPU time or compute units."""

    default_code = "quota_exceeded"


class ResourceUnavailableError(L2KCError):
    """The requested accelerator is not offered or not free right now."""

    default_code = "resource_unavailable"


class PremiumNotAllowed(L2KCError):
    """A premium accelerator was requested without ``allow_premium``."""

    default_code = "premium_not_allowed"


class NotSupportedError(L2KCError):
    """The provider or mode does not implement this operation."""

    default_code = "not_supported"


class RealModeDisabledError(L2KCError):
    """A real call was attempted without ``mode: real`` and ``allow_real_jobs``."""

    default_code = "real_mode_disabled"


class TransientProviderError(L2KCError):
    """A status, logs, or download call failed in a way that can be retried.

    Submit and cancel are never retried. A disconnected runtime is a job
    outcome, not this error.
    """

    default_code = "transient_error"


class SubmitOutcomeUnknownError(L2KCError):
    """Submit may or may not have reached the provider.

    The job is not resubmitted. Reconcile polls when a provider id was stored.
    """

    default_code = "submit_outcome_unknown"


class ProviderCLIError(L2KCError):
    """The official provider CLI is missing or exited non-zero.

    ``stderr_excerpt`` is redacted and truncated. It is an excerpt, not a log dump.
    """

    default_code = "provider_cli_error"

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr_excerpt: str | None = None,
        code: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        excerpt: str | None = None
        if stderr_excerpt:
            excerpt = redact(stderr_excerpt)[:500]
        self.returncode = returncode
        self.stderr_excerpt = excerpt
        merged = dict(details or {})
        merged["returncode"] = returncode
        if excerpt is not None:
            merged["stderr_excerpt"] = excerpt
        super().__init__(message, code=code, provider=provider, details=merged)

    def to_dict(self) -> dict[str, Any]:
        body = super().to_dict()
        body["returncode"] = self.returncode
        body["stderr_excerpt"] = self.stderr_excerpt
        return body


class JobNotFoundError(L2KCError):
    """No stored job matches the id."""

    default_code = "job_not_found"


class JobFailedError(L2KCError):
    """The job finished unsuccessfully."""

    default_code = "job_failed"


class RuntimeDisconnectedError(L2KCError):
    """The remote runtime went away before the job finished."""

    default_code = "runtime_disconnected"


class InvalidInputError(L2KCError):
    """Caller input is invalid.

    Named ``InvalidInputError`` so it does not clash with ``pydantic.ValidationError``.
    """

    default_code = "invalid_input"


class InvalidTransitionError(L2KCError):
    """The job state machine rejected this state change.

    ``details`` carries ``from_state`` and ``to_state``.
    """

    default_code = "invalid_transition"


class ConcurrentUpdateError(L2KCError):
    """A write lost an optimistic version check.

    The stored ``version`` is not the ``expected_version`` the caller read.
    """

    default_code = "concurrent_update"


class PathSecurityError(L2KCError):
    """A path escaped its base directory or used a forbidden component."""

    default_code = "path_security"


def _details_with_account_id(
    details: dict[str, Any] | None,
    account_id: str | None,
) -> dict[str, Any]:
    merged = dict(details or {})
    if account_id is not None:
        merged["account_id"] = account_id
    elif "account_id" not in merged:
        merged["account_id"] = None
    return merged


class AccountNotFoundError(L2KCError):
    """No account matches this id for the requested provider.

    Not transient. Callers must not substitute a different account.
    """

    default_code = "account_not_found"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        account_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            provider=provider,
            details=_details_with_account_id(details, account_id),
        )


class AccountDisabledError(L2KCError):
    """The selected account is disabled.

    Not transient. Another account of the same provider is not a fallback.
    """

    default_code = "account_disabled"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        account_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            provider=provider,
            details=_details_with_account_id(details, account_id),
        )


class CredentialRefError(L2KCError):
    """A credential reference cannot be resolved.

    Not transient. The reference is a name or a location, not a secret value.
    """

    default_code = "credential_ref_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        account_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            provider=provider,
            details=_details_with_account_id(details, account_id),
        )
