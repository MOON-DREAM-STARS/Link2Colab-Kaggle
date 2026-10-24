"""Premium GPU gate shared by candidate and resource checks."""

from __future__ import annotations

from link2kaggle_colab.core.config import SchedulerSettings


def premium_gate(settings: SchedulerSettings, confirmed: bool) -> str | None:
    """Return why a premium GPU is blocked, or None when it is allowed.

    ``confirmed`` is the per-job flag. It never overrides
    ``scheduler.allow_premium_gpu``.
    """
    if not settings.allow_premium_gpu:
        return "premium GPU blocked: scheduler.allow_premium_gpu is false"
    if settings.require_premium_confirmation and not confirmed:
        return "premium GPU requires per-job confirmation (allow_premium)"
    return None
