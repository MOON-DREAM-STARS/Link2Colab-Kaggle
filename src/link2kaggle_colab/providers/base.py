"""Synchronous provider interface. Async job control comes in a later checkpoint."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from link2kaggle_colab.core.errors import NotSupportedError
from link2kaggle_colab.core.models import (
    Artifact,
    AuthStatus,
    Capability,
    JobHandle,
    JobResult,
    JobSpec,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)


class Provider(ABC):
    """Cloud or local executor. Methods stay synchronous for V0.1."""

    name: ProviderName
    is_mock: bool

    @abstractmethod
    def capabilities(self) -> Capability:
        """Return the single capability record for this provider."""

    @abstractmethod
    def auth_status(self) -> AuthStatus:
        """Return the current auth snapshot."""

    @abstractmethod
    def get_quota(self) -> QuotaInfo:
        """Return quota. Mock implementations must mark fixtures."""

    @abstractmethod
    def list_resources(self) -> list[ResourceSpec]:
        """Return accelerators this provider can offer."""

    @abstractmethod
    def submit(self, spec: JobSpec) -> JobHandle:
        """Start ``spec`` and return a handle."""

    @abstractmethod
    def status(self, job_id: str) -> JobResult:
        """Return the latest known state of ``job_id``."""

    @abstractmethod
    def logs(self, job_id: str) -> str:
        """Return the log text for ``job_id``."""

    @abstractmethod
    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        """Copy job outputs into ``dest_dir``."""

    def cancel(self, job_id: str) -> JobResult:
        """Cancel a job. The default is not supported (Kaggle CLI has no cancel)."""
        del job_id
        provider = self.name.value if isinstance(self.name, ProviderName) else str(self.name)
        raise NotSupportedError(
            f"{provider} does not support cancel",
            provider=provider,
        )
