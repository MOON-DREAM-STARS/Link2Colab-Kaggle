"""Allowed job state transitions.

Same-state refresh (``X -> X``) is always allowed, including terminal states.
Every other pair must appear in :data:`TRANSITIONS`.

Transition table:

    PENDING    -> SUBMITTING, FAILED, CANCELLED
    SUBMITTING -> QUEUED, RUNNING, SUCCEEDED, FAILED, UNKNOWN
    QUEUED     -> RUNNING, SUCCEEDED, FAILED, CANCELLING, CANCELLED, UNKNOWN
    RUNNING    -> SUCCEEDED, FAILED, CANCELLING, CANCELLED, UNKNOWN
    CANCELLING -> CANCELLED, SUCCEEDED, FAILED, QUEUED, RUNNING, UNKNOWN
    UNKNOWN    -> QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLING, CANCELLED
    SUCCEEDED, FAILED, CANCELLED -> (none)

``CANCELLING`` may return to ``QUEUED`` or ``RUNNING`` when the provider
refused or ignored the cancel.

``UNKNOWN`` is not a place to guess. Crash recovery resolves it by polling
provider status, which is why an ``UNKNOWN`` job may move to any observed
non-pending state. :data:`ACTIVE_STATES` are the states that recovery must
reconcile by polling. It must not submit them again. ``PENDING`` is absent
from that set: nothing was submitted yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from link2kaggle_colab.core.errors import InvalidTransitionError
from link2kaggle_colab.core.models import JobState

TRANSITIONS: Mapping[JobState, frozenset[JobState]] = MappingProxyType(
    {
        JobState.PENDING: frozenset(
            {JobState.SUBMITTING, JobState.FAILED, JobState.CANCELLED}
        ),
        JobState.SUBMITTING: frozenset(
            {
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.UNKNOWN,
            }
        ),
        JobState.QUEUED: frozenset(
            {
                JobState.RUNNING,
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLING,
                JobState.CANCELLED,
                JobState.UNKNOWN,
            }
        ),
        JobState.RUNNING: frozenset(
            {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLING,
                JobState.CANCELLED,
                JobState.UNKNOWN,
            }
        ),
        JobState.CANCELLING: frozenset(
            {
                JobState.CANCELLED,
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.UNKNOWN,
            }
        ),
        JobState.UNKNOWN: frozenset(
            {
                JobState.QUEUED,
                JobState.RUNNING,
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLING,
                JobState.CANCELLED,
            }
        ),
        JobState.SUCCEEDED: frozenset(),
        JobState.FAILED: frozenset(),
        JobState.CANCELLED: frozenset(),
    }
)

ACTIVE_STATES: frozenset[JobState] = frozenset(
    {
        JobState.SUBMITTING,
        JobState.QUEUED,
        JobState.RUNNING,
        JobState.CANCELLING,
        JobState.UNKNOWN,
    }
)


def can_transition(current: JobState, target: JobState) -> bool:
    """Return whether ``current`` may move to ``target``.

    A same-state pair is a no-op refresh and is always allowed.
    """
    source = JobState(current)
    destination = JobState(target)
    if source is destination:
        return True
    return destination in TRANSITIONS[source]


def validate_transition(current: JobState, target: JobState) -> None:
    """Raise :class:`InvalidTransitionError` when the pair is not allowed."""
    source = JobState(current)
    destination = JobState(target)
    if can_transition(source, destination):
        return
    raise InvalidTransitionError(
        f"cannot transition from {source.value} to {destination.value}",
        details={"from_state": source.value, "to_state": destination.value},
    )
