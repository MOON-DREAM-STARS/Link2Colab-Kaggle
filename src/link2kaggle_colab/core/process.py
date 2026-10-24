"""Subprocess adapter for official CLIs.

Commands are lists of strings. ``shell`` is never enabled. The child inherits
the process environment unless an :class:`AccountEnv` is passed. That overlay
is a new dict: ``os.environ`` is never mutated, and the overlay is never copied
into results, errors, or logs. Stderr excerpts are redacted and truncated
before they leave this module.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from link2kaggle_colab.core.envspec import AccountEnv, apply_account_env
from link2kaggle_colab.core.errors import InvalidInputError, ProviderCLIError
from link2kaggle_colab.core.redact import redact

# Excerpts are diagnostic snippets, not log dumps.
STDERR_EXCERPT_LIMIT = 2000

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,49}$")
_KERNEL_REF = re.compile(r"^[A-Za-z0-9_-]+/[a-z0-9][a-z0-9-]*(/\d+)?$")
_SESSION_NAME = re.compile(r"^l2kc-[a-z0-9-]{1,40}$")


@dataclass(frozen=True)
class CommandResult:
    """One finished CLI invocation. Streams are already redacted by the runner."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float


class CommandRunner(Protocol):
    """Runs one argv list and returns the captured result."""

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        input: str | None = None,
        env: AccountEnv | None = None,
    ) -> CommandResult:
        """Run ``args`` without a shell.

        ``env`` None inherits the process environment. An :class:`AccountEnv`
        is applied to a new mapping for the child only.
        """


def _require_arg_list(args: Sequence[str]) -> list[str]:
    if not isinstance(args, list) or not args or not all(isinstance(item, str) for item in args):
        raise InvalidInputError("command args must be a non-empty list of strings")
    return list(args)


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def redact_excerpt(value: str | bytes | None, *, limit: int = STDERR_EXCERPT_LIMIT) -> str:
    """Redact secrets and cap the length of a CLI excerpt."""
    return redact(_as_text(value))[:limit]


def _reject_unsafe_text(value: str, kind: str) -> None:
    if not isinstance(value, str):
        raise InvalidInputError(f"{kind} must be a string", details={"kind": kind})
    if value.startswith("-"):
        raise InvalidInputError(
            f"{kind} must not start with '-'",
            details={"kind": kind, "reason": "option injection"},
        )
    if "\x00" in value or "\n" in value or "\r" in value:
        raise InvalidInputError(
            f"{kind} contains a forbidden control character",
            details={"kind": kind},
        )


def validate_cli_arg(value: str, kind: str) -> str:
    """Reject option injection and control characters in one CLI argument."""
    _reject_unsafe_text(value, kind)
    if value == "":
        raise InvalidInputError(f"{kind} must not be empty", details={"kind": kind})
    return value


def validate_slug(value: str) -> str:
    """Require a lowercase slug of 3 to 50 characters."""
    _reject_unsafe_text(value, "slug")
    if _SLUG.fullmatch(value) is None:
        raise InvalidInputError(
            "value is not a valid slug",
            details={"kind": "slug"},
        )
    return value


def validate_kernel_ref(value: str) -> str:
    """Require ``owner/slug`` or ``owner/slug/version``."""
    _reject_unsafe_text(value, "kernel ref")
    if _KERNEL_REF.fullmatch(value) is None:
        raise InvalidInputError(
            "value is not a valid kernel ref",
            details={"kind": "kernel ref"},
        )
    return value


def validate_session_name(value: str) -> str:
    """Require a ``l2kc-`` session name."""
    _reject_unsafe_text(value, "session name")
    if _SESSION_NAME.fullmatch(value) is None:
        raise InvalidInputError(
            "value is not a valid session name",
            details={"kind": "session name"},
        )
    return value


def validate_remote_relpath(value: str) -> str:
    """Require a relative remote path that cannot escape or look like a flag.

    Leading ``/`` and ``-``, backslashes, ``..``, and empty or dot parts are
    rejected. A single slash between safe parts is allowed.
    """
    _reject_unsafe_text(value, "remote path")
    if "\\" in value or value.startswith(("/", "\\")):
        raise InvalidInputError(
            "remote path must be relative",
            details={"kind": "remote path"},
        )
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise InvalidInputError(
            "remote path must stay relative",
            details={"kind": "remote path"},
        )
    if not path.parts:
        raise InvalidInputError("remote path must not be empty", details={"kind": "remote path"})
    for part in path.parts:
        if part in {"", ".", ".."} or part.startswith("-"):
            raise InvalidInputError(
                "remote path contains an unsafe component",
                details={"kind": "remote path"},
            )
    return value


def absolute_cli_path(path: Path) -> str:
    """Return an absolute resolved path that is safe to pass as one CLI argument."""
    resolved = Path(path).expanduser().resolve()
    text = str(resolved)
    if not resolved.is_absolute() or text.startswith("-"):
        raise InvalidInputError("local path must be an absolute path")
    if "\x00" in text or "\n" in text or "\r" in text:
        raise InvalidInputError("local path contains a forbidden control character")
    return text


class SubprocessRunner:
    """``subprocess.run`` with ``shell=False`` and captured text output.

    A missing executable becomes ``cli_missing``. A timeout becomes
    ``cli_timeout``. Non-zero exits are returned, not raised (``check=False``).
    """

    def run(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        input: str | None = None,
        env: AccountEnv | None = None,
    ) -> CommandResult:
        argv = _require_arg_list(args)
        started = time.perf_counter()
        # A new dict only when an overlay is requested. Omitting ``env`` lets
        # the child inherit. Never assign to ``os.environ`` and never store it.
        run_kwargs: dict[str, object] = {
            "cwd": cwd,
            "timeout": timeout,
            "input": input,
            "shell": False,
            "capture_output": True,
            "text": True,
            "check": False,
        }
        if env is not None:
            run_kwargs["env"] = apply_account_env(env, os.environ)
        try:
            completed = subprocess.run(argv, **run_kwargs)  # noqa: S603
        except subprocess.TimeoutExpired as exc:
            excerpt = redact_excerpt(exc.stderr) or redact_excerpt(exc.stdout)
            raise ProviderCLIError(
                "command timed out",
                code="cli_timeout",
                stderr_excerpt=excerpt or None,
            ) from exc
        except FileNotFoundError as exc:
            raise ProviderCLIError(
                f"CLI executable not found: {argv[0]}",
                code="cli_missing",
            ) from exc
        duration = time.perf_counter() - started
        return CommandResult(
            args=list(argv),
            returncode=int(completed.returncode),
            stdout=redact(completed.stdout or ""),
            stderr=redact(completed.stderr or ""),
            duration=duration,
        )
