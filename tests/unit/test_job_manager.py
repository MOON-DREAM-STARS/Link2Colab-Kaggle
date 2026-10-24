"""Job manager: one submit, retries, cancel, desync, and crash recovery."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from link2kaggle_colab.core.config import JobsSettings
from link2kaggle_colab.core.errors import (
    AuthMissingError,
    ConfigError,
    InvalidInputError,
    InvalidTransitionError,
    L2KCError,
    NotSupportedError,
    PathSecurityError,
    ProviderCLIError,
    SubmitOutcomeUnknownError,
    TransientProviderError,
)
from link2kaggle_colab.core.models import (
    Artifact,
    AuthMethod,
    AuthState,
    AuthStatus,
    Capability,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)
from link2kaggle_colab.jobs.idempotency import spec_hash
from link2kaggle_colab.jobs.manager import JobManager
from link2kaggle_colab.jobs.retry import RetryPolicy
from link2kaggle_colab.jobs.sync import run_sync
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.storage.sqlite_store import JobRecord, SqliteJobStore


def _rng(seed: int) -> random.Random:
    return random.Random(seed)  # noqa: S311


class _Sleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def _no_sleep(_delay: float) -> None:
    return None


def _script(tmp_path: Path, body: str = "print('ok')\n") -> Path:
    path = tmp_path / "train.py"
    path.write_text(body, encoding="utf-8")
    return path


def _settings(**overrides: object) -> JobsSettings:
    payload: dict[str, object] = {
        "submit_timeout_seconds": 5,
        "status_timeout_seconds": 5,
        "logs_timeout_seconds": 5,
        "download_timeout_seconds": 5,
        "cancel_timeout_seconds": 5,
        "retry_max_attempts": 4,
        "retry_base_delay_seconds": 0.5,
        "retry_max_delay_seconds": 8.0,
        "reconcile_concurrency": 4,
        "stale_submitting_seconds": 900,
    }
    payload.update(overrides)
    return JobsSettings.model_validate(payload)


def _result(
    provider_job_id: str,
    state: JobState,
    message: str = "ok",
    **extra: object,
) -> JobResult:
    body: dict[str, object] = {
        "job_id": provider_job_id,
        "provider": ProviderName.fake,
        "state": state,
        "message": message,
        "is_mock": True,
    }
    body.update(extra)
    return JobResult.model_validate(body)


class ScriptedProvider(Provider):
    """In-test provider. Call counts are locked. Handlers replace defaults."""

    name = ProviderName.fake
    is_mock = True

    def __init__(self) -> None:
        self.calls = {name: 0 for name in ("submit", "status", "logs", "download", "cancel")}
        self.status_ids: list[str] = []
        self._lock = threading.Lock()
        self.submit_handler: Callable[[JobSpec], JobHandle] | None = None
        self.status_handler: Callable[[str], JobResult] | None = None
        self.logs_handler: Callable[[str], str] | None = None
        self.download_handler: Callable[[str, Path], list[Artifact]] | None = None
        self.cancel_handler: Callable[[str], JobResult] | None = None

    def _bump(self, operation: str) -> None:
        with self._lock:
            self.calls[operation] += 1

    def capabilities(self) -> Capability:
        return Capability(
            provider=self.name,
            supports_submit=True,
            supports_status=True,
            supports_logs=True,
            supports_cancel=True,
            supports_download=True,
            supports_quota=False,
            supports_interactive_session=False,
            is_batch_only=True,
        )

    def auth_status(self) -> AuthStatus:
        return AuthStatus(
            provider=self.name,
            state=AuthState.authenticated,
            method=AuthMethod.mock,
            is_mock=True,
            message="TEST FIXTURE",
        )

    def get_quota(self) -> QuotaInfo:
        return QuotaInfo.unknown(self.name, "TEST FIXTURE")

    def list_resources(self) -> list[ResourceSpec]:
        return []

    def submit(self, spec: JobSpec) -> JobHandle:
        self._bump("submit")
        if self.submit_handler is not None:
            return self.submit_handler(spec)
        return JobHandle(
            job_id=uuid4().hex,
            provider=self.name,
            remote_id="remote-1",
            submitted_at=datetime.now(UTC),
        )

    def status(self, job_id: str) -> JobResult:
        self._bump("status")
        with self._lock:
            self.status_ids.append(job_id)
        if self.status_handler is not None:
            return self.status_handler(job_id)
        return _result(job_id, JobState.RUNNING, "running")

    def logs(self, job_id: str) -> str:
        self._bump("logs")
        if self.logs_handler is not None:
            return self.logs_handler(job_id)
        return "log line"

    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        self._bump("download")
        if self.download_handler is not None:
            return self.download_handler(job_id, dest_dir)
        path = dest_dir / "result.txt"
        path.write_text("ok", encoding="utf-8")
        return [Artifact(name="result.txt", local_path=path, size_bytes=path.stat().st_size)]

    def cancel(self, job_id: str) -> JobResult:
        self._bump("cancel")
        if self.cancel_handler is not None:
            return self.cancel_handler(job_id)
        return _result(job_id, JobState.CANCELLED, "cancelled")


def _open(
    tmp_path: Path,
    provider: ScriptedProvider,
    *,
    seed: int = 1234,
    **settings: object,
) -> tuple[JobManager, SqliteJobStore, _Sleep]:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    sleep = _Sleep()
    manager = JobManager(
        store,
        lambda _name, _account_id: provider,
        _settings(**settings),
        sleep=sleep,
        rng=_rng(seed),
    )
    return manager, store, sleep


def _wait_until(predicate: Callable[[], bool], timeout: float = 2) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
    raise AssertionError("condition was not met")


async def test_submit_happy_path_persists_our_id(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="link2kaggle_colab.jobs")
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="job-1")
    outcome = await manager.submit(ProviderName.fake, spec)
    assert outcome.created is True
    assert outcome.handle.job_id == outcome.record.job_id
    assert outcome.record.provider_job_id
    assert outcome.handle.job_id != outcome.record.provider_job_id
    assert outcome.record.state is JobState.QUEUED
    assert outcome.record.remote_id == "remote-1"
    assert outcome.handle.account_id == "fake-default"
    assert outcome.record.account_id == "fake-default"
    assert outcome.record.details["submit_pid"] == os.getpid()
    assert isinstance(outcome.record.details["submit_started_at"], str)
    assert provider.calls["submit"] == 1
    events = store.events(outcome.record.job_id)
    pairs = [(item["from_state"], item["to_state"]) for item in events]
    assert pairs == [
        (JobState.SUBMITTING.value, JobState.QUEUED.value),
    ]
    assert "submit_start" in caplog.text
    assert "submit_ok" in caplog.text
    assert "state_change" in caplog.text
    assert manager._submit_timeout(JobSpec(script_path=spec.script_path, timeout_seconds=15)) == 20


async def test_submit_provider_lookup_failure_creates_no_row(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")

    def _missing(_name: ProviderName, _account_id: str) -> Provider:
        raise ConfigError("provider fake is disabled", provider="fake")

    manager = JobManager(store, _missing, _settings())
    with pytest.raises(ConfigError) as caught:
        await manager.submit(
            ProviderName.fake,
            JobSpec(script_path=_script(tmp_path), idempotency_key="no-row"),
        )
    assert caught.value.code == "config_error"
    assert store.list() == []


async def test_submit_timeout_stays_unknown_until_reconcile_polls_status(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    started = threading.Event()
    release = threading.Event()
    provider_job = uuid4().hex

    def _submit(spec: JobSpec) -> JobHandle:
        del spec
        started.set()
        assert release.wait(5)
        return JobHandle(
            job_id=provider_job,
            provider=ProviderName.fake,
            remote_id="remote-late",
            submitted_at=datetime.now(UTC),
        )

    provider.submit_handler = _submit
    manager, store, _sleep = _open(tmp_path, provider, submit_timeout_seconds=0.05)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="timeout-1")
    task = asyncio.create_task(manager.submit(ProviderName.fake, spec))
    job_id = ""
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(SubmitOutcomeUnknownError) as caught:
            await task
        assert caught.value.code == "submit_outcome_unknown"
        assert caught.value.details["job_state"] == "UNKNOWN"
        job_id = str(caught.value.details["job_id"])
        record = store.get(job_id)
        assert record.state is JobState.UNKNOWN
        assert record.provider_job_id is None
        assert "timed out" in record.message
        assert provider.calls["submit"] == 1
    finally:
        release.set()
    assert job_id

    def _recorded() -> bool:
        return store.get(job_id).provider_job_id == provider_job

    _wait_until(_recorded)
    late = store.get(job_id)
    assert late.state is JobState.UNKNOWN
    assert late.remote_id == "remote-late"

    def _status(provider_id: str) -> JobResult:
        assert provider_id == provider_job
        return _result(provider_id, JobState.RUNNING, "running")

    provider.status_handler = _status
    reports = await manager.reconcile()
    assert provider.calls["submit"] == 1
    assert provider.calls["status"] == 1
    assert reports[0].ok is True
    assert reports[0].before is JobState.UNKNOWN
    assert reports[0].after is JobState.RUNNING
    assert store.get(job_id).state is JobState.RUNNING
    assert provider_job in provider.status_ids
    assert job_id not in provider.status_ids


async def test_ambiguous_and_definitive_submit_errors_are_not_resubmitted(tmp_path: Path) -> None:
    provider = ScriptedProvider()

    def _cli(spec: JobSpec) -> JobHandle:
        del spec
        raise ProviderCLIError("cli blew up", provider="fake")

    provider.submit_handler = _cli
    manager, store, _sleep = _open(tmp_path, provider)
    script = _script(tmp_path)
    ambiguous = JobSpec(script_path=script, idempotency_key="amb-1")
    with pytest.raises(ProviderCLIError) as cli_error:
        await manager.submit(ProviderName.fake, ambiguous)
    assert cli_error.value.details["job_state"] == "UNKNOWN"
    ambiguous_id = str(cli_error.value.details["job_id"])
    assert store.get(ambiguous_id).state is JobState.UNKNOWN
    again = await manager.submit(ProviderName.fake, ambiguous)
    assert again.created is False
    assert again.record.job_id == ambiguous_id
    assert provider.calls["submit"] == 1

    def _reset(spec: JobSpec) -> JobHandle:
        del spec
        raise ConnectionError("connection reset")

    provider.submit_handler = _reset
    with pytest.raises(SubmitOutcomeUnknownError) as wrapped:
        await manager.submit(
            ProviderName.fake,
            JobSpec(script_path=script, idempotency_key="amb-2"),
        )
    assert wrapped.value.details["exception_type"] == "ConnectionError"
    assert wrapped.value.details["job_state"] == "UNKNOWN"
    assert store.get(str(wrapped.value.details["job_id"])).state is JobState.UNKNOWN
    assert provider.calls["submit"] == 2

    def _auth(spec: JobSpec) -> JobHandle:
        del spec
        raise AuthMissingError("no credential", provider="fake")

    provider.submit_handler = _auth
    definitive = JobSpec(script_path=script, idempotency_key="auth-1")
    with pytest.raises(AuthMissingError) as auth_error:
        await manager.submit(ProviderName.fake, definitive)
    assert auth_error.value.details["job_state"] == "FAILED"
    failed_id = str(auth_error.value.details["job_id"])
    assert store.get(failed_id).state is JobState.FAILED
    assert store.get(failed_id).details["error_code"] == "auth_missing"
    repeated = await manager.submit(ProviderName.fake, definitive)
    assert repeated.created is False
    assert repeated.record.job_id == failed_id
    assert provider.calls["submit"] == 3


async def test_derived_hash_dedupes_until_the_job_is_terminal(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path))
    first = await manager.submit(ProviderName.fake, spec)
    second = await manager.submit(ProviderName.fake, spec)
    assert second.created is False
    assert second.record.job_id == first.record.job_id
    assert provider.calls["submit"] == 1
    current = store.get(first.record.job_id)
    store.transition(
        current.job_id,
        JobState.SUCCEEDED,
        expected_version=current.version,
    )
    third = await manager.submit(ProviderName.fake, spec)
    assert third.created is True
    assert third.record.job_id != first.record.job_id
    assert provider.calls["submit"] == 2


async def test_create_inserts_submitting_before_the_provider_returns(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    started = threading.Event()
    release = threading.Event()

    def _submit(spec: JobSpec) -> JobHandle:
        del spec
        started.set()
        assert release.wait(5)
        return JobHandle(
            job_id=uuid4().hex,
            provider=ProviderName.fake,
            remote_id="remote-held",
            submitted_at=datetime.now(UTC),
        )

    provider.submit_handler = _submit
    manager, store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="held-1")
    task = asyncio.create_task(manager.submit(ProviderName.fake, spec, "held-account"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        inflight = store.list()
        assert len(inflight) == 1
        assert inflight[0].state is JobState.SUBMITTING
        assert inflight[0].account_id == "held-account"
        assert inflight[0].details["submit_pid"] == os.getpid()
        assert isinstance(inflight[0].details["submit_started_at"], str)
        assert store.events(inflight[0].job_id) == []
    finally:
        release.set()
    outcome = await task
    assert outcome.created is True
    assert outcome.handle.account_id == "held-account"
    assert outcome.record.state is JobState.QUEUED
    pairs = [(item["from_state"], item["to_state"]) for item in store.events(outcome.record.job_id)]
    assert pairs == [(JobState.SUBMITTING.value, JobState.QUEUED.value)]


async def test_submit_and_status_carry_the_bound_account_id(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="acct-1")
    outcome = await manager.submit(ProviderName.fake, spec, "lab-account")
    assert outcome.created is True
    assert outcome.handle.account_id == "lab-account"
    assert outcome.record.account_id == "lab-account"
    assert store.get(outcome.record.job_id).state is JobState.QUEUED
    result = await manager.status(outcome.record.job_id)
    assert result.account_id == "lab-account"
    assert result.job_id == outcome.handle.job_id


async def test_dedupe_returns_a_legacy_pending_row_unchanged(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="legacy-pending")
    now = datetime.now(UTC)
    legacy = store.insert(
        JobRecord(
            job_id=uuid4().hex,
            provider=ProviderName.fake,
            account_id="fake-default",
            state=JobState.PENDING,
            idempotency_key="legacy-pending",
            spec_hash=spec_hash(spec),
            created_at=now,
            updated_at=now,
        )
    )
    outcome = await manager.submit(ProviderName.fake, spec)
    assert outcome.created is False
    assert outcome.handle.job_id == legacy.job_id
    assert outcome.handle.account_id == "fake-default"
    kept = store.get(legacy.job_id)
    assert kept.state is JobState.PENDING
    assert kept.version == legacy.version
    assert provider.calls["submit"] == 0
    assert store.events(legacy.job_id) == []


async def test_idempotency_key_with_a_different_spec_does_not_submit(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="same-key", name="first")
    first = await manager.submit(ProviderName.fake, spec)
    stored = store.get(first.record.job_id)
    other = spec.model_copy(update={"name": "second"})
    with pytest.raises(InvalidInputError) as caught:
        await manager.submit(ProviderName.fake, other)
    assert caught.value.details["reason"] == "spec_mismatch"
    assert provider.calls["submit"] == 1
    current = store.get(first.record.job_id)
    assert current.state is stored.state
    assert current.version == stored.version
    assert current.spec_hash == stored.spec_hash
    duplicate = await manager.submit(ProviderName.fake, spec)
    assert duplicate.created is False
    assert duplicate.record.job_id == first.record.job_id
    assert provider.calls["submit"] == 1


async def test_concurrent_asyncio_submits_share_one_provider_call(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, _store, _sleep = _open(tmp_path, provider)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="same-key")
    outcomes = await asyncio.gather(
        *[manager.submit(ProviderName.fake, spec) for _ in range(10)]
    )
    ids = {item.record.job_id for item in outcomes}
    assert ids == {outcomes[0].record.job_id}
    assert sum(1 for item in outcomes if item.created) == 1
    assert provider.calls["submit"] == 1


def test_threaded_managers_share_one_provider_call(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    db = tmp_path / "l2kc.db"
    SqliteJobStore(db)
    spec = JobSpec(script_path=_script(tmp_path), idempotency_key="thread-key")
    barrier = threading.Barrier(10)
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _worker() -> None:
        try:
            barrier.wait(5)
            manager = JobManager(
                SqliteJobStore(db),
                lambda _name, _account_id: provider,
                _settings(),
                sleep=_no_sleep,
                rng=_rng(1),
            )
            outcome = run_sync(lambda: manager.submit(ProviderName.fake, spec))
            with lock:
                results.append(outcome.record.job_id)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert errors == []
    assert len(results) == 10
    assert len(set(results)) == 1
    assert provider.calls["submit"] == 1


async def test_status_retries_transient_failures_with_jitter(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    remaining = {"count": 2}

    def _status(provider_id: str) -> JobResult:
        if remaining["count"]:
            remaining["count"] -= 1
            raise ConnectionError("dropped")
        return _result(provider_id, JobState.RUNNING, "up")

    provider.status_handler = _status
    manager, store, sleep = _open(
        tmp_path,
        provider,
        seed=99,
        retry_max_attempts=4,
        retry_base_delay_seconds=0.5,
        retry_max_delay_seconds=8.0,
    )
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="status-1"),
    )
    result = await manager.status(outcome.record.job_id)
    assert result.job_id == outcome.record.job_id
    assert result.job_id != outcome.record.provider_job_id
    assert result.state is JobState.RUNNING
    assert result.is_mock is True
    assert provider.calls["status"] == 3
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0)
    expected = _rng(99)
    assert sleep.delays == [policy.delay(1, expected), policy.delay(2, expected)]
    assert 0 <= sleep.delays[0] <= 0.5
    assert 0 <= sleep.delays[1] <= 1.0
    assert store.get(outcome.record.job_id).state is JobState.RUNNING


async def test_status_retries_transient_provider_error_and_timeout(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    pending: list[BaseException] = [
        TransientProviderError("blip"),
        TimeoutError("slow"),
    ]

    def _status(provider_id: str) -> JobResult:
        if pending:
            raise pending.pop(0)
        return _result(provider_id, JobState.RUNNING, "up")

    provider.status_handler = _status
    manager, store, sleep = _open(tmp_path, provider, seed=7, retry_max_attempts=4)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="status-transient"),
    )
    result = await manager.status(outcome.record.job_id)
    assert result.state is JobState.RUNNING
    assert result.job_id == outcome.record.job_id
    assert provider.calls["status"] == 3
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0)
    expected = _rng(7)
    assert sleep.delays == [policy.delay(1, expected), policy.delay(2, expected)]
    assert store.get(outcome.record.job_id).state is JobState.RUNNING


async def test_non_transient_status_is_not_retried_and_keeps_state(tmp_path: Path) -> None:
    provider = ScriptedProvider()

    def _status(provider_id: str) -> JobResult:
        del provider_id
        raise InvalidInputError("bad id")

    provider.status_handler = _status
    manager, store, sleep = _open(tmp_path, provider)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="status-2"),
    )
    with pytest.raises(InvalidInputError):
        await manager.status(outcome.record.job_id)
    assert provider.calls["status"] == 1
    assert sleep.delays == []
    assert store.get(outcome.record.job_id).state is JobState.QUEUED


async def test_exhausted_status_retries_leave_the_state_unchanged(tmp_path: Path) -> None:
    provider = ScriptedProvider()

    def _status(provider_id: str) -> JobResult:
        del provider_id
        raise ConnectionError("still down")

    provider.status_handler = _status
    manager, store, sleep = _open(tmp_path, provider, retry_max_attempts=3)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="status-3"),
    )
    with pytest.raises(ConnectionError):
        await manager.status(outcome.record.job_id)
    assert provider.calls["status"] == 3
    assert len(sleep.delays) == 2
    assert store.get(outcome.record.job_id).state is JobState.QUEUED


async def test_cancel_transitions_and_timeout(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    script = _script(tmp_path)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=script, idempotency_key="cancel-1"),
    )
    job_id = outcome.record.job_id
    provider_job = outcome.record.provider_job_id
    assert provider_job is not None

    def _running(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.RUNNING, "running")

    provider.status_handler = _running
    running = await manager.status(job_id)
    assert running.state is JobState.RUNNING

    def _cancelled(provider_id: str) -> JobResult:
        assert provider_id == provider_job
        return _result(provider_id, JobState.CANCELLED, "cancelled")

    provider.cancel_handler = _cancelled
    cancelled = await manager.cancel(job_id)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.job_id == job_id
    assert provider.calls["cancel"] == 1
    pairs = [(item["from_state"], item["to_state"]) for item in store.events(job_id)]
    assert (JobState.RUNNING.value, JobState.CANCELLING.value) in pairs
    assert (JobState.CANCELLING.value, JobState.CANCELLED.value) in pairs

    calls = provider.calls["cancel"]
    again = await manager.cancel(job_id)
    assert again.state is JobState.CANCELLED
    assert provider.calls["cancel"] == calls

    pending = store.insert(
        _record(job_id=uuid4().hex, state=JobState.PENDING, provider_job_id=None)
    )
    pending_result = await manager.cancel(pending.job_id)
    assert pending_result.state is JobState.CANCELLED
    assert provider.calls["cancel"] == calls

    ambiguous = store.insert(
        _record(job_id=uuid4().hex, state=JobState.UNKNOWN, provider_job_id=None)
    )
    with pytest.raises(InvalidInputError, match="ambiguous"):
        await manager.cancel(ambiguous.job_id)
    assert store.get(ambiguous.job_id).state is JobState.UNKNOWN


async def test_cancel_not_supported_reverts_and_timeout_stays_cancelling(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider, cancel_timeout_seconds=0.05)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="cancel-2"),
    )
    job_id = outcome.record.job_id
    provider_job = str(outcome.record.provider_job_id)

    def _running(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.RUNNING, "running")

    provider.status_handler = _running
    await manager.status(job_id)

    def _unsupported(provider_id: str) -> JobResult:
        del provider_id
        raise NotSupportedError("cancel is not available", provider="fake")

    provider.cancel_handler = _unsupported
    with pytest.raises(NotSupportedError) as unsupported:
        await manager.cancel(job_id)
    assert store.get(job_id).state is JobState.RUNNING
    assert unsupported.value.details["job_state"] == "RUNNING"
    assert provider.calls["cancel"] == 1

    started = threading.Event()
    release = threading.Event()

    def _block(provider_id: str) -> JobResult:
        del provider_id
        started.set()
        assert release.wait(5)
        return _result(provider_job, JobState.CANCELLED, "late")

    provider.cancel_handler = _block
    task = asyncio.create_task(manager.cancel(job_id))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(L2KCError) as timed_out:
            await task
        assert timed_out.value.details["job_state"] == "CANCELLING"
        assert store.get(job_id).state is JobState.CANCELLING
    finally:
        release.set()

    def _done(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.CANCELLED, "cancelled")

    provider.status_handler = _done
    reports = await manager.reconcile([job_id])
    assert reports[0].ok is True
    assert reports[0].after is JobState.CANCELLED
    assert store.get(job_id).state is JobState.CANCELLED
    assert provider.calls["submit"] == 1
    assert provider.calls["cancel"] == 2


async def test_runtime_release_failure_keeps_failed(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="release-1"),
    )
    job_id = outcome.record.job_id

    def _failed(provider_id: str) -> JobResult:
        return _result(
            provider_id,
            JobState.FAILED,
            "runtime reclaimed and disconnected",
            details={"runtime_stop_failed": True},
        )

    def _stop(provider_id: str) -> JobResult:
        del provider_id
        raise RuntimeError("stop failed")

    provider.status_handler = _failed
    provider.cancel_handler = _stop
    result = await manager.status(job_id)
    assert result.state is JobState.FAILED
    assert provider.calls["cancel"] == 1
    assert result.details["runtime_release_attempted"] is True
    assert result.details["runtime_release_outcome"] == "failed"
    assert "runtime reclaimed" in result.message
    await manager.status(job_id)
    assert provider.calls["cancel"] == 1
    assert store.get(job_id).state is JobState.FAILED


async def test_state_desync_is_recorded_and_invalid_transition_is_rejected(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="desync-1"),
    )
    job_id = outcome.record.job_id

    def _cancel_then_running(provider_id: str) -> JobResult:
        current = store.get(job_id)
        store.transition(
            job_id,
            JobState.CANCELLED,
            expected_version=current.version,
            message="cancelled locally",
        )
        return _result(provider_id, JobState.RUNNING, "still running")

    provider.status_handler = _cancel_then_running
    result = await manager.status(job_id)
    assert result.state is JobState.CANCELLED
    assert result.details["desync"]["reported"] == "RUNNING"
    assert "at" in result.details["desync"]

    second = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path, "print(2)\n"), idempotency_key="desync-2"),
    )
    second_id = second.record.job_id

    def _to_running(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.RUNNING, "running")

    provider.status_handler = _to_running
    await manager.status(second_id)

    def _succeed_then_running(provider_id: str) -> JobResult:
        current = store.get(second_id)
        store.transition(
            second_id,
            JobState.SUCCEEDED,
            expected_version=current.version,
            message="finished",
        )
        return _result(provider_id, JobState.RUNNING, "stale running")

    provider.status_handler = _succeed_then_running
    kept = await manager.status(second_id)
    assert kept.state is JobState.SUCCEEDED
    assert kept.details["desync"]["reported"] == "RUNNING"
    fresh = store.get(second_id)
    with pytest.raises(InvalidTransitionError):
        store.transition(
            second_id,
            JobState.RUNNING,
            expected_version=fresh.version,
        )
    assert store.get(second_id).state is JobState.SUCCEEDED

    queued = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path, "print(3)\n"), idempotency_key="desync-3"),
    )

    def _pending(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.PENDING, "went backwards")

    provider.status_handler = _pending
    stayed = await manager.refresh(queued.record.job_id)
    assert stayed.state is JobState.QUEUED
    assert stayed.details["desync"]["reported"] == "PENDING"


async def test_download_records_only_paths_inside_dest(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="download-1"),
    )
    job_id = outcome.record.job_id
    current = store.get(job_id)
    store.transition(job_id, JobState.SUCCEEDED, expected_version=current.version)
    dest = tmp_path / "out"

    def _queued(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.QUEUED, "queued")

    running_id = (
        await manager.submit(
            ProviderName.fake,
            JobSpec(script_path=_script(tmp_path, "print(4)\n"), idempotency_key="download-2"),
        )
    ).record.job_id
    provider.status_handler = _queued
    with pytest.raises(InvalidInputError, match="SUCCEEDED"):
        await manager.download(running_id, dest)
    assert provider.calls["download"] == 0
    assert store.get(running_id).state is JobState.QUEUED

    def _boom(provider_id: str, folder: Path) -> list[Artifact]:
        del provider_id, folder
        raise InvalidInputError("disk full")

    provider.download_handler = _boom
    with pytest.raises(InvalidInputError, match="disk full"):
        await manager.download(job_id, dest)
    assert store.get(job_id).state is JobState.SUCCEEDED
    assert store.get(job_id).artifacts == []

    def _escape(provider_id: str, folder: Path) -> list[Artifact]:
        del provider_id, folder
        outside = tmp_path / "escaped.txt"
        outside.write_text("nope", encoding="utf-8")
        return [Artifact(name="escaped.txt", local_path=outside, size_bytes=4)]

    provider.download_handler = _escape
    with pytest.raises(PathSecurityError):
        await manager.download(job_id, dest / "safe")
    assert store.get(job_id).state is JobState.SUCCEEDED
    assert store.get(job_id).artifacts == []

    def _ok(provider_id: str, folder: Path) -> list[Artifact]:
        del provider_id
        path = folder / "result.txt"
        path.write_text("hi", encoding="utf-8")
        return [Artifact(name="result.txt", local_path=path, size_bytes=path.stat().st_size)]

    provider.download_handler = _ok
    artifacts = await manager.download(job_id, dest)
    assert [item.name for item in artifacts] == ["result.txt"]
    assert store.get(job_id).artifacts == ["result.txt"]
    assert store.get(job_id).state is JobState.SUCCEEDED


async def test_reconcile_recovers_active_rows_without_submit(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    past = datetime.now(UTC) - timedelta(seconds=5000)
    store = SqliteJobStore(tmp_path / "l2kc.db")
    rows: list[JobRecord] = []
    moment = past
    for state, with_id in (
        (JobState.QUEUED, True),
        (JobState.RUNNING, True),
        (JobState.CANCELLING, True),
        (JobState.UNKNOWN, True),
        (JobState.SUBMITTING, False),
    ):
        moment = moment + timedelta(seconds=1)
        provider_job_id = uuid4().hex if with_id else None
        rows.append(
            store.insert(
                _record(
                    job_id=uuid4().hex,
                    state=state,
                    provider_job_id=provider_job_id,
                    created_at=moment,
                    updated_at=past,
                )
            )
        )
    manager = JobManager(
        store,
        lambda _name, _account_id: provider,
        _settings(stale_submitting_seconds=900),
        sleep=_no_sleep,
        rng=_rng(5),
    )

    def _status(provider_id: str) -> JobResult:
        return _result(provider_id, JobState.RUNNING, "polled")

    provider.status_handler = _status
    reports = await manager.reconcile()
    assert provider.calls["submit"] == 0
    assert provider.calls["status"] == 4
    by_id = {item.job_id: item for item in reports}
    for row in rows[:4]:
        assert by_id[row.job_id].after is JobState.RUNNING
        assert store.get(row.job_id).state is JobState.RUNNING
        assert row.provider_job_id in provider.status_ids
        assert row.job_id not in provider.status_ids
    stale = rows[4]
    assert store.get(stale.job_id).state is JobState.UNKNOWN
    assert store.get(stale.job_id).provider_job_id is None
    assert "ambiguous" in (by_id[stale.job_id].note or "")


async def test_reconcile_keeps_going_when_one_status_fails(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    manager, store, _sleep = _open(tmp_path, provider, retry_max_attempts=2)
    script = _script(tmp_path)
    bad = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=script, idempotency_key="bad"),
    )
    good = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path, "print(9)\n"), idempotency_key="good"),
    )
    bad_provider = str(bad.record.provider_job_id)

    def _status(provider_id: str) -> JobResult:
        if provider_id == bad_provider:
            raise ConnectionError("down")
        return _result(provider_id, JobState.RUNNING, "up")

    provider.status_handler = _status
    reports = await manager.reconcile([bad.record.job_id, good.record.job_id])
    assert [item.ok for item in reports] == [False, True]
    assert reports[0].error is not None
    assert store.get(bad.record.job_id).state is JobState.QUEUED
    assert store.get(good.record.job_id).state is JobState.RUNNING
    assert provider.calls["submit"] == 2


async def test_logs_require_a_provider_id(tmp_path: Path) -> None:
    provider = ScriptedProvider()
    token = "KGAT_" + ("q" * 20)
    manager, store, _sleep = _open(tmp_path, provider)
    outcome = await manager.submit(
        ProviderName.fake,
        JobSpec(script_path=_script(tmp_path), idempotency_key="logs-1"),
    )

    def _logs(provider_id: str) -> str:
        assert provider_id == outcome.record.provider_job_id
        return f"line {token}"

    provider.logs_handler = _logs
    text = await manager.logs(outcome.record.job_id)
    assert token not in text
    assert "[REDACTED]" in text
    missing = store.insert(
        _record(job_id=uuid4().hex, state=JobState.UNKNOWN, provider_job_id=None)
    )
    with pytest.raises(InvalidInputError, match="ambiguous"):
        await manager.logs(missing.job_id)


def test_run_sync_with_and_without_a_running_loop() -> None:
    async def _value() -> int:
        return 7

    assert run_sync(_value) == 7

    async def _inside() -> int:
        return run_sync(_value)

    assert asyncio.run(_inside()) == 7

    async def _boom() -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_sync(_boom)


def _record(
    *,
    job_id: str,
    state: JobState,
    provider_job_id: str | None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> JobRecord:
    now = created_at or datetime.now(UTC)
    return JobRecord(
        job_id=job_id,
        provider=ProviderName.fake,
        account_id="fake-default",
        provider_job_id=provider_job_id,
        state=state,
        created_at=now,
        updated_at=updated_at or now,
    )
