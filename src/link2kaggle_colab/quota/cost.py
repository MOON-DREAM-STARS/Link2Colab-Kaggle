"""Bound a Colab job in compute units (CCU).

CCU (compute units) and GPU-hours are different units. This module never converts
one into the other. Kaggle quota is GPU-hours. Colab spend is CCU.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from link2kaggle_colab.core.config import CcuRate


class CostEstimate(BaseModel):
    """CCU spend for one candidate. ``unit`` is never ``gpu_hours``."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    unit: Literal["compute_units"] = "compute_units"
    ccu_per_hour: float | None
    rate_source: str | None
    rate_verified: bool
    estimated_hours: float | None
    estimated_ccu: float | None
    # Why ``estimated_ccu`` is None. Empty when the estimate is a number.
    reason: str = ""


def estimate_ccu(
    candidate_id: str,
    rates: Mapping[str, CcuRate],
    timeout_seconds: int | None,
    *,
    high_mem: bool = False,
) -> CostEstimate:
    """Estimate CCU for ``timeout_seconds`` at the configured hourly rate.

    ``estimated_hours`` is ``timeout_seconds / 3600`` when a timeout is set.
    ``estimated_ccu`` is that duration times ``ccu_per_hour``, rounded to 4
    decimal places. A missing timeout or a missing rate leaves ``estimated_ccu``
    as None and explains why in ``reason``. Hours are not a Kaggle GPU-hour
    balance and are not compared to one.

    When ``high_mem`` is true, prefer ``{candidate_id}+high_mem`` if configured,
    otherwise fall back to the base candidate rate.
    """
    rate_key = f"{candidate_id}+high_mem" if high_mem else candidate_id
    rate = rates.get(rate_key)
    if rate is None and high_mem:
        rate = rates.get(candidate_id)
        rate_key = candidate_id
    hours = None if timeout_seconds is None else timeout_seconds / 3600
    if rate is None:
        return CostEstimate(
            candidate_id=candidate_id,
            ccu_per_hour=None,
            rate_source=None,
            rate_verified=False,
            estimated_hours=hours,
            estimated_ccu=None,
            reason=f"no CCU rate configured for {rate_key}",
        )
    if timeout_seconds is None:
        return CostEstimate(
            candidate_id=candidate_id,
            ccu_per_hour=rate.ccu_per_hour,
            rate_source=rate.source,
            rate_verified=rate.verified,
            estimated_hours=None,
            estimated_ccu=None,
            reason="timeout_seconds is required to bound CCU spend",
        )
    return CostEstimate(
        candidate_id=candidate_id,
        ccu_per_hour=rate.ccu_per_hour,
        rate_source=rate.source,
        rate_verified=rate.verified,
        estimated_hours=hours,
        estimated_ccu=round(hours * rate.ccu_per_hour, 4),
        reason="",
    )


def kaggle_hours_needed(timeout_seconds: int | None) -> float | None:
    """GPU-hours a Kaggle quota would charge for this timeout.

    Returns None when ``timeout_seconds`` is omitted. The result is a Kaggle
    GPU-hour figure, not compute units.
    """
    if timeout_seconds is None:
        return None
    return timeout_seconds / 3600
