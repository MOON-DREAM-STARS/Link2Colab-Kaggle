"""Client wiring for the SQLite job store and JobManager. Offline only."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import (
    FakeProviderSettings,
    JobsSettings,
    KaggleProviderSettings,
    MockSettings,
    ProvidersSettings,
    Settings,
)
from link2kaggle_colab.core.errors import (
    ConfigError,
    InvalidInputError,
    L2KCError,
    SubmitOutcomeUnknownError,
)
from link2kaggle_colab.core.models import (
    GpuType,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
)
from link2kaggle_colab.providers.fake.provider import FakeProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.storage.sqlite_store import JobRecord, SqliteJobStore, default_db_path


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        mock=MockSettings(),
    )


def _script(directory: Path, body: str = "print('ok')\n") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "job.py"
    path.write_text(body, encoding="utf-8")
    return path


def _cpu(directory: Path, **updates: object) -> JobSpec:
    body = updates.pop("body", None)
    payload: dict[str, object] = {
        "script_path": _script(directory, body if isinstance(body, str) else "print('ok')\n"),
        "provider": ProviderName.fake,
        "gpu_type": GpuType.NONE,
        "gpu_count": 0,
    }
    payload.update(updates)
    return JobSpec.model_validate(payload)


class _CountingFake(FakeProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.submits = 0
        self._lock = threading.Lock()

    def submit(self, spec: JobSpec) -> JobHandle:
        with self._lock:
            self.submits += 1
        return super().submit(spec)


class _CountingKaggle(MockKaggleProvider):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.submits = 0

    def submit(self, spec: JobSpec) -> JobHandle:
        self.submits += 1
        return super().submit(spec)


def test_fresh_client_on_same_state_dir_recovers_the_job(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = Client(settings)
    handle = client.submit(_cpu(tmp_path))
    restored = Client(settings)
    status = restored.status(handle.job_id)
    assert status.job_id == handle.job_id
    assert status.provider is ProviderName.fake
    assert status.state is JobState.SUCCEEDED
    rows = restored.list_jobs()
    assert [row["job_id"] for row in rows] == [handle.job_id]
    assert rows[0]["provider"] == "fake"
    assert rows[0]["origin"] == "sqlite"
    assert rows[0]["provider_job_id"]
    assert rows[0]["provider_job_id"] != handle.job_id


def test_v02_json_index_migrates_and_routes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = MockKaggleProvider(settings)
    handle = provider.submit(
        JobSpec(
            script_path=_script(tmp_path),
            provider=ProviderName.kaggle,
            gpu_type=GpuType.T4,
            gpu_count=2,
        )
    )
    jobs = settings.state_dir / "jobs"
    jobs.mkdir()
    path = jobs / f"{handle.job_id}.json"
    path.write_text(
        json.dumps({"job_id": handle.job_id, "provider": "kaggle"}),
        encoding="utf-8",
    )
    raw = path.read_text(encoding="utf-8")
    client = Client(settings)
    result = client.status(handle.job_id)
    assert result.job_id == handle.job_id
    assert result.provider is ProviderName.kaggle
    rows = client.list_jobs()
    assert rows[0]["origin"] == "json_migration"
    assert rows[0]["provider"] == "kaggle"
    assert rows[0]["provider_job_id"] == handle.job_id
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == raw
    again = Client(settings).status(handle.job_id)
    assert again.provider is ProviderName.kaggle
    assert again.job_id == handle.job_id


def test_duplicate_idempotency_key_calls_provider_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = _CountingFake(settings)
    client = Client(settings, providers={ProviderName.fake: provider})
    spec = _cpu(tmp_path, idempotency_key="same-key")
    first = client.submit_detailed(spec)
    second = client.submit_detailed(spec)
    assert provider.submits == 1
    assert first.created is True
    assert second.created is False
    assert first.handle.job_id == second.handle.job_id
    assert second.handle.provider is ProviderName.fake


def test_reused_idempotency_key_with_a_different_spec_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = _CountingFake(settings)
    client = Client(settings, providers={ProviderName.fake: provider})
    spec = _cpu(tmp_path, idempotency_key="same-key", name="first-run")
    first = client.submit(spec)
    before = client.status(first.job_id)
    stored = client.list_jobs()
    other = spec.model_copy(update={"name": "second-run"})
    with pytest.raises(InvalidInputError) as caught:
        client.submit(other)
    assert caught.value.code == "invalid_input"
    assert caught.value.details["reason"] == "spec_mismatch"
    assert provider.submits == 1
    after = client.status(first.job_id)
    assert after.job_id == first.job_id
    assert after.state is before.state
    assert client.list_jobs() == stored
    again = client.submit(spec)
    assert again.job_id == first.job_id
    assert provider.submits == 1


def test_eight_threads_with_one_key_call_provider_once(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    provider = _CountingFake(settings)
    client = Client(settings, providers={ProviderName.fake: provider})
    spec = _cpu(tmp_path, idempotency_key="shared")
    barrier = threading.Barrier(8)
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _worker() -> None:
        try:
            barrier.wait(5)
            job_id = client.submit(spec).job_id
            with lock:
                results.append(job_id)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive()
    assert errors == []
    assert len(results) == 8
    assert len(set(results)) == 1
    assert provider.submits == 1


def test_scheduler_and_cpu_submits_persist_rows(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = Client(settings)
    gpu = client.submit(
        JobSpec(
            script_path=_script(tmp_path),
            gpu_type=GpuType.T4,
            gpu_count=2,
            timeout_seconds=3600,
        )
    )
    cpu = client.submit(_cpu(tmp_path / "cpu", body="print('cpu')\n"))
    rows = {row["job_id"]: row for row in client.list_jobs()}
    assert rows[gpu.job_id]["provider"] == "kaggle"
    assert rows[gpu.job_id]["provider_job_id"]
    assert rows[gpu.job_id]["provider_job_id"] != gpu.job_id
    assert rows[cpu.job_id]["provider"] == "fake"
    assert rows[cpu.job_id]["origin"] == "sqlite"


def test_reconcile_never_calls_submit(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    kaggle = _CountingKaggle(settings)
    client = Client(settings, providers={ProviderName.kaggle: kaggle})
    handle = client.submit(
        JobSpec(
            script_path=_script(tmp_path),
            provider=ProviderName.kaggle,
            gpu_type=GpuType.T4,
            gpu_count=2,
            timeout_seconds=3600,
        )
    )
    assert kaggle.submits == 1
    one = client.reconcile_jobs([handle.job_id])
    assert one[0]["ok"] is True
    assert one[0]["job_id"] == handle.job_id
    assert kaggle.submits == 1
    polled = client.reconcile_jobs(None)
    assert kaggle.submits == 1
    assert all(item["ok"] for item in polled)


def test_disabled_provider_routes_status_and_submit_creates_no_row(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = Client(settings)
    handle = client.submit(
        JobSpec(
            script_path=_script(tmp_path),
            provider=ProviderName.kaggle,
            gpu_type=GpuType.T4,
            gpu_count=2,
        )
    )
    disabled = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=settings.state_dir,
        mock=settings.mock,
        providers=ProvidersSettings(kaggle=KaggleProviderSettings(enabled=False)),
    )
    restored = Client(disabled)
    status = restored.status(handle.job_id)
    assert status.provider is ProviderName.kaggle
    assert status.job_id == handle.job_id
    before = [row["job_id"] for row in restored.list_jobs()]

    off = Client(
        Settings(
            mode="mock",
            allow_real_jobs=False,
            state_dir=tmp_path / "fake-off",
            providers=ProvidersSettings(fake=FakeProviderSettings(enabled=False)),
        )
    )
    with pytest.raises(ConfigError) as caught:
        off.submit(_cpu(tmp_path / "blocked"))
    assert caught.value.code == "config_error"
    assert off.list_jobs() == []
    assert [row["job_id"] for row in restored.list_jobs()] == before


def _join_named(name: str) -> None:
    current = threading.current_thread()
    deadline = time.monotonic() + 2
    for thread in list(threading.enumerate()):
        if thread is current or thread.name != name:
            continue
        remaining = deadline - time.monotonic()
        thread.join(timeout=remaining if remaining > 0 else 0)
        assert not thread.is_alive()


async def _noop_sleep(_delay: float) -> None:
    return None


class _GateFake(FakeProvider):
    """Fake provider. One operation waits on a gate; the others run normally."""

    def __init__(self, settings: Settings, gate: threading.Event, *, block: str) -> None:
        super().__init__(settings)
        self._gate = gate
        self._block = block
        self.submits = 0
        self.status_calls = 0
        self.cancel_calls = 0
        self._lock = threading.Lock()

    def _wait(self, label: str) -> None:
        if not self._gate.wait(2):
            raise TimeoutError(f"{label} gate was not released")

    def submit(self, spec: JobSpec) -> JobHandle:
        with self._lock:
            self.submits += 1
        if self._block == "submit":
            self._wait("submit")
        return super().submit(spec)

    def status(self, job_id: str) -> JobResult:
        with self._lock:
            self.status_calls += 1
            call = self.status_calls
        if self._block == "status" and call == 1:
            self._wait("status")
        return super().status(job_id)

    def cancel(self, job_id: str) -> JobResult:
        with self._lock:
            self.cancel_calls += 1
        if self._block == "cancel":
            self._wait("cancel")
        return super().cancel(job_id)


def _gated_client(
    tmp_path: Path,
    *,
    block: str,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    **job_settings: float | int,
) -> tuple[Client, Settings, _GateFake, threading.Event]:
    settings = _settings(tmp_path).model_copy(update={"jobs": JobsSettings(**job_settings)})
    gate = threading.Event()
    provider = _GateFake(settings, gate, block=block)
    client = Client(settings, providers={ProviderName.fake: provider}, sleep=sleep)
    return client, settings, provider, gate


def _row(settings: Settings, job_id: str) -> JobRecord:
    return SqliteJobStore(default_db_path(settings.state_dir)).get(job_id)


def test_submit_timeout_returns_before_the_provider_thread(tmp_path: Path) -> None:
    client, settings, provider, gate = _gated_client(
        tmp_path,
        block="submit",
        submit_timeout_seconds=0.2,
    )
    spec = _cpu(tmp_path, idempotency_key="blocked-submit")
    try:
        started = time.perf_counter()
        with pytest.raises(SubmitOutcomeUnknownError) as caught:
            client.submit(spec)
        assert time.perf_counter() - started < 1.0
        assert caught.value.code == "submit_outcome_unknown"
        job_id = str(caught.value.details["job_id"])
        early = _row(settings, job_id)
        assert early.state is JobState.UNKNOWN
        assert early.provider_job_id is None
    finally:
        gate.set()
        _join_named("l2kc-submit")
    stored = _row(settings, job_id)
    assert stored.state is JobState.UNKNOWN
    assert stored.provider_job_id
    assert stored.provider_job_id != job_id
    assert provider.submits == 1
    reports = client.reconcile_jobs([job_id])
    assert provider.submits == 1
    assert reports[0]["ok"] is True
    assert reports[0]["before"] == "UNKNOWN"
    assert reports[0]["after"] == "SUCCEEDED"
    assert _row(settings, job_id).state is JobState.SUCCEEDED


def test_status_timeout_is_transient_and_keeps_state(tmp_path: Path) -> None:
    client, settings, provider, gate = _gated_client(
        tmp_path,
        block="status",
        status_timeout_seconds=0.2,
        retry_max_attempts=1,
    )
    handle = client.submit(_cpu(tmp_path, idempotency_key="blocked-status"))
    before = _row(settings, handle.job_id).state
    try:
        started = time.perf_counter()
        with pytest.raises(L2KCError) as caught:
            client.status(handle.job_id)
        assert time.perf_counter() - started < 1.0
        assert caught.value.code == "transient_error"
        assert caught.value.details["timed_out"] is True
        assert caught.value.details["operation"] == "status"
        assert provider.status_calls == 1
        assert _row(settings, handle.job_id).state is before
    finally:
        gate.set()
        _join_named("l2kc-status")


def test_status_timeout_is_retried_then_succeeds(tmp_path: Path) -> None:
    client, settings, provider, gate = _gated_client(
        tmp_path,
        block="status",
        sleep=_noop_sleep,
        status_timeout_seconds=0.2,
        retry_max_attempts=2,
    )
    handle = client.submit(_cpu(tmp_path, idempotency_key="retry-status"))
    try:
        result = client.status(handle.job_id)
        assert result.state is JobState.SUCCEEDED
        assert result.job_id == handle.job_id
        assert provider.status_calls == 2
    finally:
        gate.set()
        _join_named("l2kc-status")
    assert _row(settings, handle.job_id).state is JobState.SUCCEEDED


def test_cancel_timeout_stays_cancelling_and_is_not_retried(tmp_path: Path) -> None:
    client, settings, provider, gate = _gated_client(
        tmp_path,
        block="cancel",
        cancel_timeout_seconds=0.2,
        retry_max_attempts=2,
    )
    handle = client.submit(_cpu(tmp_path, idempotency_key="blocked-cancel"))
    try:
        started = time.perf_counter()
        with pytest.raises(L2KCError) as caught:
            client.cancel(handle.job_id)
        assert time.perf_counter() - started < 1.0
        assert caught.value.code == "transient_error"
        assert caught.value.details["timed_out"] is True
        assert caught.value.details["operation"] == "cancel"
        assert caught.value.details["job_state"] == "CANCELLING"
        assert provider.cancel_calls == 1
        assert _row(settings, handle.job_id).state is JobState.CANCELLING
    finally:
        gate.set()
        _join_named("l2kc-cancel")
    assert provider.cancel_calls == 1
    assert _row(settings, handle.job_id).state is JobState.CANCELLING


def test_status_timeout_inside_a_running_loop_returns(tmp_path: Path) -> None:
    client, settings, provider, gate = _gated_client(
        tmp_path,
        block="status",
        status_timeout_seconds=0.2,
        retry_max_attempts=1,
    )
    handle = client.submit(_cpu(tmp_path, idempotency_key="loop-status"))
    before = _row(settings, handle.job_id).state

    async def _inside() -> L2KCError:
        started = time.perf_counter()
        with pytest.raises(L2KCError) as caught:
            client.status(handle.job_id)
        assert time.perf_counter() - started < 1.0
        return caught.value

    try:
        error = asyncio.run(_inside())
    finally:
        gate.set()
        _join_named("l2kc-status")
    assert error.code == "transient_error"
    assert provider.status_calls == 1
    assert _row(settings, handle.job_id).state is before
