"""Run a coroutine from synchronous Client and MCP call sites."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run ``coro_factory()`` to completion and return its value.

    Uses :func:`asyncio.run` when this thread has no running loop. When a
    loop is already running (MCP stdio), the coroutine is created and run
    with :func:`asyncio.run` on a worker thread so it is not bound to the
    caller's loop. This call blocks until that result is ready.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    box: dict[str, T] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            box["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001
            error["exc"] = exc

    worker = threading.Thread(target=_worker, name="l2kc-run-sync")
    worker.start()
    worker.join()
    if "exc" in error:
        raise error["exc"]
    return box["value"]
