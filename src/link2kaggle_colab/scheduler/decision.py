"""Placement request and the decision returned by the selector.

These models do not submit a job. ``SchedulerDecision.to_dict`` keeps the V0.1
keys ``provider``, ``resource``, ``reasons``, and ``warnings``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from link2kaggle_colab.core.models import GpuType, ProviderName, ResourceSpec
from link2kaggle_colab.quota.cost import CostEstimate


class ScheduleRequest(BaseModel):
    """GPU placement request. CPU jobs are not scheduled here."""

    model_config = ConfigDict(extra="forbid")

    gpu_type: GpuType | None = None
    gpu_count: int = Field(default=1, ge=1)
    min_gpu_memory_gb_per_gpu: float | None = Field(default=None, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    # Per-job confirmation (JobSpec.allow_premium / --allow-premium).
    premium_confirmed: bool = False
    # Colab ``--high-mem`` machine shape. Does not bypass premium guards.
    high_mem: bool = False
    # None = auto. kaggle or colab pins the provider. fake is rejected.
    provider: ProviderName | None = None

    @field_validator("provider")
    @classmethod
    def _reject_fake(cls, value: ProviderName | None) -> ProviderName | None:
        if value is ProviderName.fake:
            raise ValueError("provider 'fake' cannot be scheduled")
        return value


class CandidateEvaluation(BaseModel):
    """One candidate after static checks and, when those pass, a snapshot."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    provider: ProviderName
    accepted: bool
    # Rejection reasons in check order, or acceptance notes.
    reasons: list[str]
    premium: bool
    paid: bool
    cost: CostEstimate | None = None
    quota: dict[str, Any] | None = None


class SchedulerDecision(BaseModel):
    """Result of one ``select_backend`` call. No job was submitted."""

    model_config = ConfigDict(extra="forbid")

    strategy: str
    provider: ProviderName | None
    candidate_id: str | None
    gpu_type: GpuType | None
    resource: ResourceSpec | None
    fallback_used: bool
    estimated_ccu: float | None
    reasons: list[str]
    warnings: list[str]
    evaluations: list[CandidateEvaluation]
    is_fixture: bool

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready decision, including the V0.1 recommendation keys."""
        payload = self.model_dump(mode="json")
        payload["provider"] = payload.get("provider")
        payload["resource"] = payload.get("resource")
        payload["reasons"] = list(payload.get("reasons") or [])
        payload["warnings"] = list(payload.get("warnings") or [])
        return payload
