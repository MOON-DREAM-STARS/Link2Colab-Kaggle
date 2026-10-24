"""In-memory Colab provider. Balances, rates, and membership are TEST FIXTURES."""

from __future__ import annotations

from link2kaggle_colab.auth.base import AuthProvider
from link2kaggle_colab.auth.mock import COLAB_AUTH_SCENARIOS, COLAB_MOCK_ACCOUNT, MockColabAuth
from link2kaggle_colab.core.errors import (
    ConfigError,
    QuotaExceededError,
    ResourceUnavailableError,
    RuntimeDisconnectedError,
)
from link2kaggle_colab.core.models import (
    TPU_TYPES,
    AuthState,
    AuthStatus,
    Capability,
    GpuType,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)
from link2kaggle_colab.providers.mock_fixtures import colab_quota, colab_resources
from link2kaggle_colab.providers.mock_runtime import ScenarioMockProvider, wants_gpu

COLAB_JOB_SCENARIOS = frozenset(
    {
        "ok",
        "free",
        "free_pro_zero",
        "ai_pro",
        "ccu_insufficient",
        "premium_unavailable",
        "runtime_disconnect",
    }
)
_PREMIUM = frozenset({GpuType.A100, GpuType.H100, *TPU_TYPES})


class MockColabProvider(ScenarioMockProvider):
    """Deterministic Colab mock. Cancel maps to ``colab stop``."""

    name = ProviderName.colab

    def _default_scenario(self) -> str:
        return self._settings.mock.colab_scenario

    def _split_scenario(self, scenario: str) -> tuple[str, str]:
        if scenario in COLAB_JOB_SCENARIOS:
            return scenario, "adc_valid"
        if scenario in COLAB_AUTH_SCENARIOS:
            return "ok", scenario
        raise ConfigError(
            f"unknown Colab mock scenario: {scenario}",
            provider=self.name.value,
            details={"scenario": scenario},
        )

    def _make_auth(self, auth_scenario: str) -> AuthProvider:
        return MockColabAuth(auth_scenario)

    def _resources(self) -> list[ResourceSpec]:
        return colab_resources(self.job_scenario)

    def _progression(self) -> tuple[list[JobState], str]:
        if self.job_scenario == "runtime_disconnect":
            error = RuntimeDisconnectedError(
                "TEST FIXTURE: Colab runtime disconnected",
                provider=self.name.value,
            )
            return (
                [JobState.QUEUED, JobState.RUNNING, JobState.FAILED],
                f"{error.code}: {error.message}",
            )
        return super()._progression()

    def capabilities(self) -> Capability:
        return Capability(
            provider=self.name,
            supports_submit=True,
            supports_status=True,
            supports_logs=True,
            supports_cancel=True,
            supports_download=True,
            supports_quota=True,
            supports_interactive_session=False,
            is_batch_only=True,
            notes=[
                "Mock cancel maps to `colab stop`.",
                "CCU balances, membership, and hourly rates are TEST FIXTURES.",
                "Fixture hourly rates are not official; real rates come from `colab usage`.",
            ],
        )

    def get_quota(self) -> QuotaInfo:
        self._require_auth()
        quota = self._with_account(colab_quota(self.job_scenario))
        if not isinstance(quota, QuotaInfo):
            raise TypeError("quota must be QuotaInfo")
        return quota

    def auth_status(self) -> AuthStatus:
        status = self.auth.status()
        if self.job_scenario == "ai_pro" and status.state is AuthState.authenticated:
            status = status.model_copy(update={"account_label": "Google AI Pro (MOCK)"})
        elif self.job_scenario == "free" and status.account_label == COLAB_MOCK_ACCOUNT:
            status = status.model_copy(update={"account_label": "mock-colab-free (TEST FIXTURE)"})
        stamped = self._with_account(status)
        if not isinstance(stamped, AuthStatus):
            raise TypeError("auth status must be AuthStatus")
        return stamped

    def _guard_resources(self, spec: JobSpec) -> None:
        gpu = spec.gpu_type if spec.gpu_type is not None else GpuType.T4
        if self.job_scenario == "ccu_insufficient" and wants_gpu(spec):
            raise QuotaExceededError(
                "TEST FIXTURE: 2 CCU remaining is not enough to start this job.",
                provider=self.name.value,
                details={"fixture": True, "remaining": 2},
            )
        if self.job_scenario == "premium_unavailable" and gpu in _PREMIUM and wants_gpu(spec):
            raise ResourceUnavailableError(
                "TEST FIXTURE: premium GPU is not available on this account.",
                provider=self.name.value,
                details={"fixture": True},
            )
        super()._guard_resources(spec)

    def _cancel_message(self) -> str:
        return "TEST FIXTURE: cancelled (maps to colab stop)"
