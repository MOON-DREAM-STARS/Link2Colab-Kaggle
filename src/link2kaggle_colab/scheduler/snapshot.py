"""Provider facts the selector reads. The selector does not fetch them itself."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from link2kaggle_colab.core.models import AuthStatus, ProviderName, ResourceSpec
from link2kaggle_colab.quota.cache import CachedQuota


class ProviderSnapshot(BaseModel):
    """Enabled flag, auth, quota, and resources for one provider.

    ``error`` is ``"<code>: <message>"`` when auth, quota, or resources raised.
    ``keeps_runtime`` mirrors ``colab.keep_runtime`` on the Colab snapshot.
    When true, the runtime is not stopped after the job, so CCU spend is not
    bounded by ``timeout_seconds``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    provider: ProviderName
    enabled: bool
    auth: AuthStatus | None = None
    quota: CachedQuota | None = None
    resources: list[ResourceSpec] | None = None
    error: str | None = None
    keeps_runtime: bool = False


SnapshotSource = Callable[[ProviderName], ProviderSnapshot]
