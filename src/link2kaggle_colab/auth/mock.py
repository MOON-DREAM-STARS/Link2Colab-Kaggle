"""Mock authentication scenarios. Every status is a TEST FIXTURE."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from link2kaggle_colab.auth.base import AuthProvider
from link2kaggle_colab.core.errors import AuthError, ConfigError
from link2kaggle_colab.core.models import AuthMethod, AuthState, AuthStatus, ProviderName

KAGGLE_MOCK_ACCOUNT = "mock-kaggle-user (TEST FIXTURE)"
COLAB_MOCK_ACCOUNT = "mock-colab-user (TEST FIXTURE)"

KAGGLE_AUTH_SCENARIOS = frozenset(
    {
        "token_valid",
        "token_invalid",
        "token_missing",
        "oauth_ok",
        "oauth_fail",
        "permission_error",
    }
)
COLAB_AUTH_SCENARIOS = frozenset(
    {
        "oauth_ok",
        "oauth_fail",
        "token_expired",
        "refresh_fail",
        "adc_valid",
        "adc_missing",
    }
)


def _status(
    *,
    provider: ProviderName,
    state: AuthState,
    method: AuthMethod,
    account_label: str,
    message: str,
    scenario: str,
    expires_at: datetime | None = None,
) -> AuthStatus:
    return AuthStatus(
        provider=provider,
        state=state,
        method=method,
        account_label=account_label,
        is_mock=True,
        expires_at=expires_at,
        message=message,
        details={"scenario": scenario, "fixture": True},
    )


class MockKaggleAuth(AuthProvider):
    """Kaggle auth fixture. Unknown scenario names raise ``ConfigError``."""

    provider = ProviderName.kaggle

    def __init__(self, scenario: str) -> None:
        if scenario not in KAGGLE_AUTH_SCENARIOS:
            raise ConfigError(
                f"unknown Kaggle mock auth scenario: {scenario}",
                provider=self.provider.value,
                details={"scenario": scenario},
            )
        self.scenario = scenario

    def status(self) -> AuthStatus:
        scenario = self.scenario
        future = datetime.now(UTC) + timedelta(hours=12)
        if scenario == "token_valid":
            return _status(
                provider=self.provider,
                state=AuthState.authenticated,
                method=AuthMethod.api_token,
                account_label=KAGGLE_MOCK_ACCOUNT,
                message="TEST FIXTURE: token accepted",
                scenario=scenario,
                expires_at=future,
            )
        if scenario == "token_invalid":
            return _status(
                provider=self.provider,
                state=AuthState.invalid,
                method=AuthMethod.api_token,
                account_label=KAGGLE_MOCK_ACCOUNT,
                message="TEST FIXTURE: token invalid",
                scenario=scenario,
            )
        if scenario == "token_missing":
            return _status(
                provider=self.provider,
                state=AuthState.unauthenticated,
                method=AuthMethod.api_token,
                account_label=KAGGLE_MOCK_ACCOUNT,
                message="TEST FIXTURE: token missing",
                scenario=scenario,
            )
        if scenario == "oauth_ok":
            return _status(
                provider=self.provider,
                state=AuthState.authenticated,
                method=AuthMethod.oauth,
                account_label=KAGGLE_MOCK_ACCOUNT,
                message="TEST FIXTURE: oauth accepted",
                scenario=scenario,
                expires_at=future,
            )
        if scenario == "oauth_fail":
            return _status(
                provider=self.provider,
                state=AuthState.error,
                method=AuthMethod.oauth,
                account_label=KAGGLE_MOCK_ACCOUNT,
                message="TEST FIXTURE: oauth failed",
                scenario=scenario,
            )
        return _status(
            provider=self.provider,
            state=AuthState.error,
            method=AuthMethod.api_token,
            account_label=KAGGLE_MOCK_ACCOUNT,
            message="TEST FIXTURE: permission denied",
            scenario=scenario,
        )


class MockColabAuth(AuthProvider):
    """Colab auth fixture.

    ``token_expired`` reports ``expired`` until :meth:`refresh`, which then
    reports ``authenticated`` with a new ``expires_at``. ``refresh_fail`` stays
    expired and ``refresh`` raises ``AuthError``.
    """

    provider = ProviderName.colab

    def __init__(self, scenario: str) -> None:
        if scenario not in COLAB_AUTH_SCENARIOS:
            raise ConfigError(
                f"unknown Colab mock auth scenario: {scenario}",
                provider=self.provider.value,
                details={"scenario": scenario},
            )
        self.scenario = scenario
        self._refreshed = False

    def status(self) -> AuthStatus:
        scenario = self.scenario
        if scenario == "token_expired" and self._refreshed:
            return _status(
                provider=self.provider,
                state=AuthState.authenticated,
                method=AuthMethod.oauth,
                account_label=COLAB_MOCK_ACCOUNT,
                message="TEST FIXTURE: refresh succeeded",
                scenario=scenario,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        if scenario == "oauth_ok":
            return _status(
                provider=self.provider,
                state=AuthState.authenticated,
                method=AuthMethod.oauth,
                account_label=COLAB_MOCK_ACCOUNT,
                message="TEST FIXTURE: oauth accepted",
                scenario=scenario,
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        if scenario == "oauth_fail":
            return _status(
                provider=self.provider,
                state=AuthState.error,
                method=AuthMethod.oauth,
                account_label=COLAB_MOCK_ACCOUNT,
                message="TEST FIXTURE: oauth failed",
                scenario=scenario,
            )
        if scenario in {"token_expired", "refresh_fail"}:
            return _status(
                provider=self.provider,
                state=AuthState.expired,
                method=AuthMethod.oauth,
                account_label=COLAB_MOCK_ACCOUNT,
                message="TEST FIXTURE: credential expired",
                scenario=scenario,
                expires_at=datetime.now(UTC) - timedelta(hours=1),
            )
        if scenario == "adc_valid":
            return _status(
                provider=self.provider,
                state=AuthState.authenticated,
                method=AuthMethod.adc,
                account_label=COLAB_MOCK_ACCOUNT,
                message="TEST FIXTURE: ADC accepted",
                scenario=scenario,
            )
        return _status(
            provider=self.provider,
            state=AuthState.unauthenticated,
            method=AuthMethod.adc,
            account_label=COLAB_MOCK_ACCOUNT,
            message="TEST FIXTURE: ADC missing",
            scenario=scenario,
        )

    def refresh(self) -> AuthStatus:
        if self.scenario == "token_expired":
            self._refreshed = True
            return self.status()
        if self.scenario == "refresh_fail":
            raise AuthError(
                "TEST FIXTURE: token refresh failed",
                provider=self.provider.value,
                details={"scenario": self.scenario, "fixture": True},
            )
        return super().refresh()
