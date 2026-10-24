"""Per-account environment overlay.

Values in :attr:`AccountEnv.set` are credentials. ``repr`` and ``str`` list
names only. Do not log the mapping, and do not write it back to ``os.environ``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from link2kaggle_colab.core.errors import InvalidInputError


@dataclass(frozen=True, slots=True)
class AccountEnv:
    """Child-process environment changes for one account.

    ``set`` replaces those names. ``unset`` removes them. Applying the overlay
    builds a new mapping and leaves the input mapping unchanged.
    """

    set: dict[str, str]
    unset: frozenset[str]

    def __post_init__(self) -> None:
        set_items = dict(self.set)
        unset_items = frozenset(self.unset)
        for key, value in set_items.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise InvalidInputError(
                    "account environment names and values must be strings",
                    details={"field": "env"},
                )
        if not all(isinstance(name, str) for name in unset_items):
            raise InvalidInputError(
                "account environment names must be strings",
                details={"field": "env"},
            )
        object.__setattr__(self, "set", set_items)
        object.__setattr__(self, "unset", unset_items)

    def __repr__(self) -> str:
        names = ",".join(sorted(self.set))
        removed = ",".join(sorted(self.unset))
        return f"AccountEnv(set_keys=[{names}], unset=[{removed}])"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def empty(cls) -> AccountEnv:
        """No changes. The child inherits the process environment."""
        return cls(set={}, unset=frozenset())


def apply_account_env(spec: AccountEnv, base: Mapping[str, str]) -> dict[str, str]:
    """Return a new environment: ``base``, minus ``unset``, then plus ``set``.

    ``base`` is not mutated. ``os.environ`` is not read and not written.
    """
    if not isinstance(spec, AccountEnv):
        raise InvalidInputError("env must be an AccountEnv", details={"field": "env"})
    child = dict(base)
    for name in spec.unset:
        child.pop(name, None)
    child.update(spec.set)
    return child
