"""In-memory Kaggle provider. Quota and auth results are TEST FIXTURES."""

from __future__ import annotations

from link2kaggle_colab.auth.base import AuthProvider
from link2kaggle_colab.auth.mock import KAGGLE_AUTH_SCENARIOS, MockKaggleAuth
from link2kaggle_colab.core.errors import (
    ConfigError,
    NotSupportedError,
    QuotaExceededError,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import (
    Capability,
    JobResult,
    JobSpec,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)
from link2kaggle_colab.providers.mock_fixtures import kaggle_quota, kaggle_resources
from link2kaggle_colab.providers.mock_runtime import ScenarioMockProvider, wants_gpu

KAGGLE_JOB_SCENARIOS = frozenset(
    {"ok", "quota_45h", "quota_insufficient", "gpu_unavailable"}
)


class MockKaggleProvider(ScenarioMockProvider):
    """Deterministic Kaggle mock. Cancel matches the real CLI: not supported."""

    name = ProviderName.kaggle

    def _default_scenario(self) -> str:
        return self._settings.mock.kaggle_scenario

    def _split_scenario(self, scenario: str) -> tuple[str, str]:
        if scenario in KAGGLE_JOB_SCENARIOS:
            return scenario, "token_valid"
        if scenario in KAGGLE_AUTH_SCENARIOS:
            return "ok", scenario
        raise ConfigError(
            f"unknown Kaggle mock scenario: {scenario}",
            provider=self.name.value,
            details={"scenario": scenario},
        )

    def _make_auth(self, auth_scenario: str) -> AuthProvider:
        return MockKaggleAuth(auth_scenario)

    def _resources(self) -> list[ResourceSpec]:
        return kaggle_resources()

    def capabilities(self) -> Capability:
        return Capability(
            provider=self.name,
            supports_submit=True,
            supports_status=True,
            supports_logs=True,
            supports_cancel=False,
            supports_download=True,
            supports_quota=True,
            supports_interactive_session=False,
            is_batch_only=True,
            notes=[
                "Kaggle is a batch runner. The official CLI has no kernel-run cancel.",
                "Quota figures from this provider are TEST FIXTURES.",
            ],
        )

    def get_quota(self) -> QuotaInfo:
        self._require_auth()
        quota = self._with_account(kaggle_quota(self.job_scenario))
        if not isinstance(quota, QuotaInfo):
            raise TypeError("quota must be QuotaInfo")
        return quota

    def _guard_resources(self, spec: JobSpec) -> None:
        if self.job_scenario == "quota_insufficient" and wants_gpu(spec):
            raise QuotaExceededError(
                "TEST FIXTURE: Kaggle GPU quota is exhausted (30h used of 30h).",
                provider=self.name.value,
                details={"fixture": True, "remaining": 0},
            )
        if self.job_scenario == "gpu_unavailable" and wants_gpu(spec):
            raise ResourceUnavailableError(
                "TEST FIXTURE: no Kaggle GPU is available.",
                provider=self.name.value,
                details={"fixture": True},
            )
        super()._guard_resources(spec)

    def cancel(self, job_id: str) -> JobResult:
        self._require_auth()
        del job_id
        raise NotSupportedError(
            "Kaggle CLI cannot cancel a running kernel. Delete is not cancel.",
            provider=self.name.value,
        )
