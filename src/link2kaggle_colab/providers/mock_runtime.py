"""In-memory job progression persisted as JSON so later processes can read it."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from link2kaggle_colab.auth.base import AuthProvider
from link2kaggle_colab.core.config import SchedulerSettings, Settings
from link2kaggle_colab.core.errors import (
    AuthError,
    AuthExpiredError,
    AuthInvalidError,
    AuthMissingError,
    InvalidInputError,
    JobNotFoundError,
    PathSecurityError,
    PermissionDeniedError,
    PremiumNotAllowed,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import (
    Artifact,
    AuthState,
    AuthStatus,
    GpuType,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)
from link2kaggle_colab.core.paths import ensure_within, safe_join
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.providers.mock_fixtures import MOCK_RESULT_TEXT, mock_log
from link2kaggle_colab.scheduler.policy import premium_gate

_JOB_ID = re.compile(r"^[a-f0-9]{32}$")


class StoredJob(BaseModel):
    """On-disk job record. No credential material belongs here."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    provider: ProviderName
    remote_id: str | None = None
    state: JobState
    message: str = ""
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    artifacts: list[str] = Field(default_factory=list)
    is_mock: bool = True
    gpu_type: GpuType | None = None
    step_index: int = 0
    states: list[JobState] = Field(default_factory=list)
    failure_message: str = ""
    workdir: str | None = None


def raise_for_auth(status: AuthStatus, *, scenario: str) -> None:
    """Raise the auth error that matches a non-authenticated fixture."""
    if status.state is AuthState.authenticated:
        return
    provider = status.provider.value
    message = status.message or "authentication failed"
    if scenario == "permission_error":
        raise PermissionDeniedError(message, provider=provider, details={"fixture": True})
    mapping: dict[AuthState, type[AuthError]] = {
        AuthState.unauthenticated: AuthMissingError,
        AuthState.invalid: AuthInvalidError,
        AuthState.expired: AuthExpiredError,
    }
    exc_type = mapping.get(status.state, AuthError)
    raise exc_type(message, provider=provider, details={"fixture": True})


def wants_gpu(spec: JobSpec) -> bool:
    """True when a cloud mock should treat the spec as a GPU request.

    ``gpu_type is None`` means the provider default, which is a GPU for Kaggle
    and Colab. ``GpuType.NONE`` or ``gpu_count == 0`` asks for CPU.
    """
    if spec.gpu_count <= 0 or spec.gpu_type is GpuType.NONE:
        return False
    return True


def recorded_gpu(spec: JobSpec) -> GpuType:
    """Accelerator a cloud mock actually runs.

    ``gpu_type is None`` with a positive count is T4, the same default the
    real adapters use when Client has not already resolved a GPU. CPU requests
    are ``GpuType.NONE``.
    """
    if not wants_gpu(spec):
        return GpuType.NONE
    if spec.gpu_type is None:
        return GpuType.T4
    return spec.gpu_type


def ensure_resource(
    resources: list[ResourceSpec],
    spec: JobSpec,
    *,
    provider: str,
    scheduler: SchedulerSettings,
) -> None:
    """Raise when ``spec`` asks for an accelerator this catalog cannot give.

    A premium GPU also has to pass :func:`premium_gate`. ``allow_premium`` on
    the spec does not override ``scheduler.allow_premium_gpu``.
    """
    if not wants_gpu(spec):
        if spec.high_mem:
            cpu = next((item for item in resources if item.cpu_only), None)
            if cpu is None or not cpu.supports_high_mem:
                raise ResourceUnavailableError(
                    "TEST FIXTURE: high_mem is not supported for CPU on this provider.",
                    provider=provider,
                    details={"fixture": True, "high_mem": True},
                )
        return
    gpu = spec.gpu_type if spec.gpu_type is not None else GpuType.T4
    match = next((item for item in resources if item.gpu_type is gpu), None)
    if match is None:
        raise ResourceUnavailableError(
            f"TEST FIXTURE: {gpu.value} is not available.",
            provider=provider,
            details={"fixture": True},
        )
    if match.premium:
        blocked = premium_gate(scheduler, spec.allow_premium)
        if blocked is not None:
            raise PremiumNotAllowed(
                blocked,
                provider=provider,
                details={"fixture": True, "gpu_type": gpu.value},
            )
    if spec.high_mem and not match.supports_high_mem:
        raise ResourceUnavailableError(
            f"TEST FIXTURE: high_mem is not supported for {gpu.value}.",
            provider=provider,
            details={"fixture": True, "high_mem": True},
        )
    fit = match.fits(spec.min_gpu_memory_gb_per_gpu, spec.gpu_count)
    if fit is False:
        raise ResourceUnavailableError(
            "TEST FIXTURE: requested GPU count or per-GPU memory exceeds the resource.",
            provider=provider,
            details={"fixture": True},
        )


def project_status(job: StoredJob, now: datetime) -> StoredJob:
    """Return the next mock state. Empty ``states`` means the record is already final."""
    if not job.states:
        return job
    index = min(max(job.step_index, 0), len(job.states) - 1)
    state = job.states[index]
    next_index = index + 1 if index < len(job.states) - 1 else index
    message = job.message
    started = job.started_at
    finished = job.finished_at
    exit_code = job.exit_code
    artifacts = list(job.artifacts)
    if state is JobState.QUEUED and not message:
        message = "TEST FIXTURE: queued"
    elif state is JobState.RUNNING:
        started = started or now
        if not message or message == "TEST FIXTURE: queued":
            message = "TEST FIXTURE: running"
    elif state is JobState.SUCCEEDED:
        started = started or now
        finished = finished or now
        exit_code = 0 if exit_code is None else exit_code
        if not message or message.startswith("TEST FIXTURE: "):
            message = "TEST FIXTURE: job succeeded"
        if not artifacts:
            artifacts = ["result.txt"]
    elif state is JobState.FAILED:
        started = started or now
        finished = finished or now
        exit_code = 1 if exit_code is None else exit_code
        if not message or message.startswith("TEST FIXTURE: "):
            message = job.failure_message or "TEST FIXTURE: job failed"
    elif state is JobState.CANCELLED:
        finished = finished or now
        if not message:
            message = "TEST FIXTURE: cancelled"
    return job.model_copy(
        update={
            "state": state,
            "step_index": next_index,
            "message": message,
            "started_at": started,
            "finished_at": finished,
            "exit_code": exit_code,
            "artifacts": artifacts,
        }
    )


def to_result(job: StoredJob) -> JobResult:
    details: dict[str, str] = {}
    if job.gpu_type is not None:
        details["gpu_type"] = job.gpu_type.value
    return JobResult(
        job_id=job.job_id,
        provider=job.provider,
        state=job.state,
        remote_id=job.remote_id,
        message=job.message,
        started_at=job.started_at,
        finished_at=job.finished_at,
        exit_code=job.exit_code,
        artifacts=list(job.artifacts),
        is_mock=job.is_mock,
        details=details,
    )


class JobStore:
    """JSON job files under a state directory."""

    def __init__(self, root: Path, *, nested: bool) -> None:
        self.root = root
        self.nested = nested

    def path_for(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise JobNotFoundError("unknown job id", details={"reason": "job id is not a uuid hex"})
        if self.nested:
            return safe_join(self.root, job_id, "job.json")
        return safe_join(self.root, f"{job_id}.json")

    def save(self, job: StoredJob) -> None:
        path = self.path_for(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)

    def load(self, job_id: str) -> StoredJob:
        path = self.path_for(job_id)
        if not path.is_file():
            raise JobNotFoundError("job not found", details={"job_id": job_id})
        try:
            return StoredJob.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise InvalidInputError("stored job metadata is unreadable") from exc

    def workdir(self, job_id: str) -> Path:
        """Fake-provider work directory. Rejects a stored path that leaves the root."""
        expected = safe_join(self.root, job_id)
        job = self.load(job_id)
        if job.workdir is None:
            return expected
        resolved = ensure_within(self.root, Path(job.workdir))
        if resolved != expected:
            raise PathSecurityError("job workdir escapes the fake state directory")
        return resolved


class ScenarioMockProvider(Provider):
    """Shared submit/status/log/download loop for scenario-driven mocks."""

    is_mock = True
    name: ProviderName

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        scenario: str | None = None,
        state_dir: Path | None = None,
        account_id: str | None = None,
    ) -> None:
        self._settings = settings if settings is not None else Settings()
        self._state_dir = Path(state_dir) if state_dir is not None else self._settings.state_dir
        self._account_id = account_id
        raw = scenario if scenario is not None else self._default_scenario()
        self.scenario = raw
        self.job_scenario, self.auth_scenario = self._split_scenario(raw)
        self.auth: AuthProvider = self._make_auth(self.auth_scenario)
        self._jobs = JobStore(self._state_dir / "mock" / self.name.value, nested=False)

    def _with_account(self, model: AuthStatus | QuotaInfo) -> AuthStatus | QuotaInfo:
        """Stamp the account id. Fixture text on the model is left as it is."""
        if not self._account_id or model.account_id:
            return model
        return model.model_copy(update={"account_id": self._account_id})

    def _default_scenario(self) -> str:
        raise NotImplementedError

    def _split_scenario(self, scenario: str) -> tuple[str, str]:
        raise NotImplementedError

    def _make_auth(self, auth_scenario: str) -> AuthProvider:
        raise NotImplementedError

    def _resources(self) -> list[ResourceSpec]:
        raise NotImplementedError

    def _progression(self) -> tuple[list[JobState], str]:
        return ([JobState.QUEUED, JobState.RUNNING, JobState.SUCCEEDED], "")

    def _guard_resources(self, spec: JobSpec) -> None:
        ensure_resource(
            self.list_resources(),
            spec,
            provider=self.name.value,
            scheduler=self._settings.scheduler,
        )

    def _require_auth(self) -> None:
        raise_for_auth(self.auth.status(), scenario=self.auth_scenario)

    def list_resources(self) -> list[ResourceSpec]:
        return self._resources()

    def auth_status(self) -> AuthStatus:
        status = self._with_account(self.auth.status())
        if not isinstance(status, AuthStatus):
            raise TypeError("auth status must be AuthStatus")
        return status

    def submit(self, spec: JobSpec) -> JobHandle:
        self._require_auth()
        self._guard_resources(spec)
        job_id = uuid4().hex
        now = datetime.now(UTC)
        states, failure = self._progression()
        remote_id = f"mock-{self.name.value}-{job_id[:12]}"
        record = StoredJob(
            job_id=job_id,
            provider=self.name,
            remote_id=remote_id,
            state=JobState.QUEUED,
            message="TEST FIXTURE: queued",
            submitted_at=now,
            is_mock=True,
            gpu_type=recorded_gpu(spec),
            states=states,
            failure_message=failure,
            step_index=0,
        )
        self._jobs.save(record)
        return JobHandle(
            job_id=job_id,
            provider=self.name,
            remote_id=remote_id,
            submitted_at=now,
        )

    def status(self, job_id: str) -> JobResult:
        self._require_auth()
        job = project_status(self._jobs.load(job_id), datetime.now(UTC))
        self._jobs.save(job)
        return to_result(job)

    def logs(self, job_id: str) -> str:
        self._require_auth()
        job = self._jobs.load(job_id)
        gpu = job.gpu_type.value if job.gpu_type is not None else None
        return redact(mock_log(self.name.value, job.job_id, job.state.value, gpu))

    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        """Copy the fixture artifact. The stored state must already be SUCCEEDED."""
        self._require_auth()
        job = self._jobs.load(job_id)
        if job.state is not JobState.SUCCEEDED:
            raise InvalidInputError(
                f"download requires a succeeded job, current state is {job.state.value}",
                provider=self.name.value,
                details={"state": job.state.value},
            )
        root = Path(dest_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        target = safe_join(root, "result.txt")
        target.write_text(MOCK_RESULT_TEXT, encoding="utf-8")
        return [
            Artifact(
                name="result.txt",
                local_path=target,
                size_bytes=target.stat().st_size,
            )
        ]

    def cancel(self, job_id: str) -> JobResult:
        """Cancel a job that is still running.

        Status is projected first, the same way ``status`` advances a mock.
        SUCCEEDED, FAILED, and CANCELLED are returned unchanged, matching the
        real Colab adapter. Kaggle overrides this because its CLI cannot cancel.
        """
        self._require_auth()
        projected = project_status(self._jobs.load(job_id), datetime.now(UTC))
        if projected.state.is_terminal():
            self._jobs.save(projected)
            return to_result(projected)
        updated = projected.model_copy(
            update={
                "state": JobState.CANCELLED,
                "states": [JobState.CANCELLED],
                "step_index": 0,
                "finished_at": projected.finished_at or datetime.now(UTC),
                "message": self._cancel_message(),
                "failure_message": "",
            }
        )
        self._jobs.save(updated)
        return to_result(updated)

    def _cancel_message(self) -> str:
        return "TEST FIXTURE: cancelled"
