"""Scripted command runner for offline adapter tests.

Responses are keyed by an argv prefix. The first unused script with the longest
matching prefix wins. A permanent script is reused. Nothing here invokes a real
``kaggle`` or ``colab`` binary.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from link2kaggle_colab.core.envspec import AccountEnv
from link2kaggle_colab.core.errors import InvalidInputError
from link2kaggle_colab.core.process import CommandResult


class _Script:
    def __init__(
        self,
        *,
        prefix: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
        error: BaseException | None,
        on_call: Callable[[list[str]], None] | None,
        permanent: bool,
    ) -> None:
        self.prefix = prefix
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        self.on_call = on_call
        self.permanent = permanent
        self.used = False


class RecordingRunner:
    """Record argv lists and replay a scripted stdout, stderr, or exception."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.invocations: list[dict[str, object]] = []
        self._scripts: list[_Script] = []

    def script(
        self,
        prefix: list[str],
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        error: BaseException | None = None,
        on_call: Callable[[list[str]], None] | None = None,
        permanent: bool = False,
    ) -> None:
        """Queue a response for commands that start with ``prefix``."""
        if not prefix or not all(isinstance(part, str) for part in prefix):
            raise ValueError("script prefix must be a non-empty list of strings")
        self._scripts.append(
            _Script(
                prefix=tuple(prefix),
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
                error=error,
                on_call=on_call,
                permanent=permanent,
            )
        )

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        input: str | None = None,
        env: AccountEnv | None = None,
    ) -> CommandResult:
        parts_ok = isinstance(args, list) and args and all(isinstance(part, str) for part in args)
        if not parts_ok:
            raise InvalidInputError("command args must be a non-empty list of strings")
        recorded = list(args)
        self.calls.append(recorded)
        self.invocations.append(
            {
                "args": recorded,
                "cwd": cwd,
                "timeout": timeout,
                "input": input,
                "env": env,
            }
        )
        chosen = self._match(recorded)
        if chosen is None:
            raise AssertionError(f"no scripted response for {recorded!r}")
        if not chosen.permanent:
            chosen.used = True
        if chosen.on_call is not None:
            chosen.on_call(recorded)
        if chosen.error is not None:
            raise chosen.error
        return CommandResult(
            args=recorded,
            returncode=chosen.returncode,
            stdout=chosen.stdout,
            stderr=chosen.stderr,
            duration=0.0,
        )

    def _match(self, args: list[str]) -> _Script | None:
        best: _Script | None = None
        best_len = -1
        for item in self._scripts:
            if item.used and not item.permanent:
                continue
            prefix = item.prefix
            if tuple(args[: len(prefix)]) != prefix:
                continue
            if len(prefix) > best_len:
                best = item
                best_len = len(prefix)
        return best
