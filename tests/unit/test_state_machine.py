"""Job state machine pairs, including same-state refresh and terminals."""

import pytest

from link2kaggle_colab.core.errors import InvalidTransitionError
from link2kaggle_colab.core.models import JobState
from link2kaggle_colab.jobs.state_machine import (
    ACTIVE_STATES,
    TRANSITIONS,
    can_transition,
    validate_transition,
)

# Independent of TRANSITIONS so a wrong table fails this module.
_EXPECTED: dict[JobState, frozenset[JobState]] = {
    JobState.PENDING: frozenset({JobState.SUBMITTING, JobState.FAILED, JobState.CANCELLED}),
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

_STATES = list(JobState)


def _allowed(current: JobState, target: JobState) -> bool:
    return current is target or target in _EXPECTED[current]


def test_transition_table_matches_contract() -> None:
    assert dict(TRANSITIONS) == _EXPECTED
    assert set(TRANSITIONS) == set(JobState)
    assert all(isinstance(value, frozenset) for value in TRANSITIONS.values())


def test_transition_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        TRANSITIONS[JobState.PENDING] = frozenset()  # type: ignore[index]


@pytest.mark.parametrize("current", _STATES)
@pytest.mark.parametrize("target", _STATES)
def test_every_pair_matches_the_table(current: JobState, target: JobState) -> None:
    allowed = _allowed(current, target)
    assert can_transition(current, target) is allowed
    if allowed:
        assert validate_transition(current, target) is None
        return
    with pytest.raises(InvalidTransitionError) as caught:
        validate_transition(current, target)
    error = caught.value
    assert error.code == "invalid_transition"
    assert error.details["from_state"] == current.value
    assert error.details["to_state"] == target.value


@pytest.mark.parametrize("state", _STATES)
def test_same_state_refresh_is_allowed(state: JobState) -> None:
    assert can_transition(state, state) is True
    assert validate_transition(state, state) is None


def test_unknown_can_reconcile_to_succeeded() -> None:
    assert can_transition(JobState.UNKNOWN, JobState.SUCCEEDED) is True
    validate_transition(JobState.UNKNOWN, JobState.SUCCEEDED)


def test_submitting_cannot_cancel() -> None:
    assert can_transition(JobState.SUBMITTING, JobState.CANCELLING) is False
    with pytest.raises(InvalidTransitionError):
        validate_transition(JobState.SUBMITTING, JobState.CANCELLING)


@pytest.mark.parametrize("terminal", [JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED])
@pytest.mark.parametrize("target", _STATES)
def test_terminal_states_reject_other_states(terminal: JobState, target: JobState) -> None:
    if terminal is target:
        validate_transition(terminal, target)
        return
    with pytest.raises(InvalidTransitionError):
        validate_transition(terminal, target)


def test_active_states_are_polled_not_resubmitted() -> None:
    assert ACTIVE_STATES == frozenset(
        {
            JobState.SUBMITTING,
            JobState.QUEUED,
            JobState.RUNNING,
            JobState.CANCELLING,
            JobState.UNKNOWN,
        }
    )
    assert isinstance(ACTIVE_STATES, frozenset)
    assert JobState.PENDING not in ACTIVE_STATES
    for terminal in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
        assert terminal not in ACTIVE_STATES
