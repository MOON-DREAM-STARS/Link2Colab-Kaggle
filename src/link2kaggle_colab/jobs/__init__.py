"""Job submission and lifecycle.

``JobManager`` is :mod:`link2kaggle_colab.jobs.manager`. It is not imported
here. :mod:`link2kaggle_colab.storage.sqlite_store` imports the state machine
while this package initializes, and the manager imports the store. An eager
import would cycle.
"""

from link2kaggle_colab.jobs.state_machine import (
    ACTIVE_STATES,
    TRANSITIONS,
    can_transition,
    validate_transition,
)

__all__ = [
    "ACTIVE_STATES",
    "TRANSITIONS",
    "can_transition",
    "validate_transition",
]
