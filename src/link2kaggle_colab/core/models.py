"""Shared pydantic models for providers, jobs, quota, and auth."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from link2kaggle_colab.core.redact import redact, redact_value

_JOB_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,49}$")
_SCRIPT_SUFFIXES = {".py", ".ipynb"}


class ProviderName(StrEnum):
    kaggle = "kaggle"
    colab = "colab"
    fake = "fake"


class AuthMethod(StrEnum):
    api_token = "api_token"  # noqa: S105
    oauth = "oauth"
    adc = "adc"
    legacy_key = "legacy_key"  # noqa: S105
    mock = "mock"
    none = "none"


class AuthState(StrEnum):
    authenticated = "authenticated"
    unauthenticated = "unauthenticated"
    expired = "expired"
    invalid = "invalid"
    error = "error"
    unknown = "unknown"


class AuthStatus(BaseModel):
    """Authentication snapshot. ``account_label`` is a display name, never a secret."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    state: AuthState
    method: AuthMethod
    account_label: str | None = None
    account_id: str | None = None
    is_mock: bool = False
    expires_at: datetime | None = None
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("account_label")
    @classmethod
    def _label_is_not_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if redact(value) != value:
            raise ValueError("account_label must not contain a secret")
        return value

    @field_validator("message")
    @classmethod
    def _redact_message(cls, value: str) -> str:
        return redact(value)

    @field_validator("details")
    @classmethod
    def _redact_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        redacted = redact_value(value)
        if not isinstance(redacted, dict):
            raise ValueError("details must be a mapping")
        return redacted


class GpuType(StrEnum):
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    H100 = "H100"
    G4 = "G4"
    P100 = "P100"
    # Colab CLI ``--tpu`` values (v5e1 / v6e1). Not Kaggle TPU ids.
    TPU_V5E1 = "TPU_V5E1"
    TPU_V6E1 = "TPU_V6E1"
    NONE = "NONE"
    OTHER = "OTHER"


# Colab ``colab new --tpu`` accelerators. Separate from GPU ``--gpu`` flags.
TPU_TYPES = frozenset({GpuType.TPU_V5E1, GpuType.TPU_V6E1})

# Accelerators where Colab honors ``--high-mem`` (L4 and TPU ignore it).
HIGH_MEM_GPU_TYPES = frozenset({GpuType.NONE, GpuType.T4, GpuType.A100, GpuType.H100, GpuType.G4})


class ResourceSpec(BaseModel):
    """One accelerator offering.

    ``gpu_memory_gb_per_gpu`` is the memory of a single GPU. It is never a sum
    across devices. A Kaggle T4 accelerator is ``gpu_count=2`` with 16 GB on
    each GPU, not one 32 GB GPU. ``None`` means the per-GPU size is unknown
    (Colab A100 has no 40 GB vs 80 GB selector).
    """

    model_config = ConfigDict(extra="forbid")

    gpu_type: GpuType
    gpu_count: int = Field(ge=0)
    gpu_memory_gb_per_gpu: float | None = Field(default=None, gt=0)
    provider_resource_id: str = ""
    premium: bool = False
    label: str = ""
    # True when Colab ``--high-mem`` applies to this offering (not L4/TPU).
    supports_high_mem: bool = False

    @computed_field
    @property
    def cpu_only(self) -> bool:
        return self.gpu_type is GpuType.NONE or self.gpu_count == 0

    def fits(
        self,
        min_gpu_memory_gb_per_gpu: float | None,
        min_gpu_count: int,
    ) -> bool | None:
        """Whether this offering meets a per-GPU memory and count request.

        Returns False when the count is too small, or when memory is known and
        too small. Returns None when a minimum memory was requested but this
        offering's per-GPU memory is unknown and the count is otherwise enough.
        """
        if self.gpu_count < min_gpu_count:
            return False
        if min_gpu_memory_gb_per_gpu is None:
            return True
        if self.gpu_memory_gb_per_gpu is None:
            return None
        return self.gpu_memory_gb_per_gpu >= min_gpu_memory_gb_per_gpu


class QuotaUnit(StrEnum):
    gpu_hours = "gpu_hours"
    compute_units = "compute_units"
    unknown = "unknown"


class QuotaInfo(BaseModel):
    """Quota snapshot. Mock numbers are test fixtures, not a live account."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    unit: QuotaUnit
    account_id: str | None = None
    total: float | None = None
    used: float | None = None
    remaining: float | None = None
    refresh_at: datetime | None = None
    is_fixture: bool = False
    source: str = "unknown"
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    # Accelerator ids this account may start. None means unknown (do not invent).
    eligible_gpus: list[str] | None = None

    @computed_field
    @property
    def known(self) -> bool:
        return self.remaining is not None

    @field_validator("message")
    @classmethod
    def _redact_message(cls, value: str) -> str:
        return redact(value)

    @field_validator("details")
    @classmethod
    def _redact_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        redacted = redact_value(value)
        if not isinstance(redacted, dict):
            raise ValueError("details must be a mapping")
        return redacted

    @classmethod
    def unknown(cls, provider: ProviderName | str, reason: str) -> QuotaInfo:
        """Quota the provider could not determine. This is not a fixture balance."""
        return cls(
            provider=ProviderName(provider),
            unit=QuotaUnit.unknown,
            source="unknown",
            message=reason,
            is_fixture=False,
        )


class Capability(BaseModel):
    """Operations one provider implements. Each provider exposes one of these."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    supports_submit: bool
    supports_status: bool
    supports_logs: bool
    supports_cancel: bool
    supports_download: bool
    supports_quota: bool
    supports_interactive_session: bool
    is_batch_only: bool
    notes: list[str] = Field(default_factory=list)


class JobState(StrEnum):
    PENDING = "PENDING"
    SUBMITTING = "SUBMITTING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    def is_terminal(self) -> bool:
        return self in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobSpec(BaseModel):
    """Local job request. No environment secrets belong on this model."""

    model_config = ConfigDict(extra="forbid")

    script_path: Path
    args: list[str] = Field(default_factory=list)
    provider: ProviderName | None = None
    gpu_type: GpuType | None = None
    gpu_count: int = Field(default=1, ge=0)
    min_gpu_memory_gb_per_gpu: float | None = Field(default=None, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    name: str | None = Field(default=None, pattern=_JOB_NAME.pattern)
    input_files: list[Path] = Field(default_factory=list)
    output_dir: Path | None = None
    allow_premium: bool = False
    # Colab machine shape ``--high-mem``. Ignored by non-Colab providers.
    # Does not bypass premium GPU guards. Default false for compatibility.
    high_mem: bool = False
    # Client-supplied dedupe key. Absent from the spec hash. Providers ignore it.
    idempotency_key: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._:-]{1,128}$",
    )

    @field_validator("script_path")
    @classmethod
    def _validate_script(cls, value: Path) -> Path:
        """Resolve the script and require an existing ``.py`` or ``.ipynb`` file."""
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"script_path must be an existing file: {path.name}")
        if path.suffix not in _SCRIPT_SUFFIXES:
            raise ValueError("script_path suffix must be .py or .ipynb")
        return path

    @field_validator("input_files")
    @classmethod
    def _resolve_inputs(cls, value: list[Path]) -> list[Path]:
        return [Path(item).expanduser().resolve() for item in value]

    @field_validator("output_dir")
    @classmethod
    def _expand_output(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @model_validator(mode="after")
    def _cpu_count_has_no_accelerator(self) -> JobSpec:
        """``gpu_count`` 0 is CPU. A concrete accelerator would still be billed."""
        if self.gpu_count == 0 and self.gpu_type is not None and self.gpu_type is not GpuType.NONE:
            raise ValueError("gpu_count 0 requests CPU; gpu_type must be omitted or NONE")
        if self.gpu_type in TPU_TYPES and self.gpu_count != 1:
            raise ValueError("Colab TPU requests require gpu_count=1")
        if self.high_mem:
            effective = self.gpu_type if self.gpu_type is not None else (
                GpuType.NONE if self.gpu_count == 0 else GpuType.T4
            )
            if effective in TPU_TYPES or effective is GpuType.L4:
                raise ValueError(
                    "high_mem is not supported for L4 or TPU "
                    "(Colab offers only one machine shape)"
                )
            if effective not in HIGH_MEM_GPU_TYPES and effective is not GpuType.OTHER:
                raise ValueError(
                    f"high_mem is not supported for accelerator {effective.value}"
                )
        return self


class JobHandle(BaseModel):
    """Identifier returned by submit. ``job_id`` is our uuid4 hex, not a remote id.

    ``account_id`` is the account bound on the stored row. Provider adapters
    may omit it; :class:`JobManager` fills it from that row.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    provider: ProviderName
    account_id: str | None = None
    remote_id: str | None = None
    submitted_at: datetime


class JobResult(BaseModel):
    """Point-in-time job status. ``JobStatus`` is the same model.

    ``details`` is provider metadata. Colab sets ``runtime_stop_failed`` when
    ``colab stop`` could not release the runtime. ``account_id`` is the
    account bound on the stored row.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    provider: ProviderName
    account_id: str | None = None
    state: JobState
    remote_id: str | None = None
    message: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    artifacts: list[str] = Field(default_factory=list)
    is_mock: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _redact_message(cls, value: str) -> str:
        return redact(value)

    @field_validator("details")
    @classmethod
    def _redact_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        redacted = redact_value(value)
        if not isinstance(redacted, dict):
            raise ValueError("details must be a mapping")
        return redacted


JobStatus = JobResult


class Artifact(BaseModel):
    """A file copied out of a job. ``name`` is relative to the download directory."""

    model_config = ConfigDict(extra="forbid")

    name: str
    local_path: Path
    size_bytes: int = Field(ge=0)

    @field_validator("name")
    @classmethod
    def _relative_name(cls, value: str) -> str:
        if not value or value.startswith(("/", "\\")) or ".." in Path(value).parts:
            raise ValueError("artifact name must be a relative path")
        return value
