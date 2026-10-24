"""Offline restart: mock submit, a new client, crash recovery, and the CLI.

Uses the real Kaggle and Colab mocks. No network, GPU, or Colab CCU.
The submit-timeout case returns while the provider is still blocked, then
opens the gate so the worker can store the provider id.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from link2kaggle_colab.cli.main import app
from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import JobsSettings, Settings
from link2kaggle_colab.core.errors import NotSupportedError, SubmitOutcomeUnknownError
from link2kaggle_colab.core.models import (
    AuthState,
    GpuType,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
)
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.providers.mock_fixtures import MOCK_RESULT_TEXT
from link2kaggle_colab.providers.mock_runtime import JobStore
from link2kaggle_colab.scheduler.decision import ScheduleRequest
from link2kaggle_colab.storage.sqlite_store import JobRecord, SqliteJobStore, default_db_path

_IDEMPOTENCY_KEY = "restart-flow-1"


def _script(directory: Path, name: str = "train.py") -> Path:
    path = directory / name
    path.write_text("print('ok')\n", encoding="utf-8")
    return path


def _store(settings: Settings) -> SqliteJobStore:
    return SqliteJobStore(default_db_path(settings.state_dir))


def _pairs(store: SqliteJobStore, job_id: str) -> list[tuple[str, str]]:
    return [(item["from_state"], item["to_state"]) for item in store.events(job_id)]


def _count_submits(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count submit calls on the real mock classes. Status and cancel are untouched."""
    counter = {"n": 0}
    lock = threading.Lock()

    def _install(cls: type[MockKaggleProvider] | type[MockColabProvider]) -> None:
        original = cls.submit

        def _counting(
            self: MockKaggleProvider | MockColabProvider,
            spec: JobSpec,
        ) -> JobHandle:
            with lock:
                counter["n"] += 1
            return original(self, spec)

        monkeypatch.setattr(cls, "submit", _counting)

    _install(MockKaggleProvider)
    _install(MockColabProvider)
    return counter


def _arm_colab_running(settings: Settings, provider_job_id: str) -> None:
    """Keep one non-terminal RUNNING tick.

    The stock progression is QUEUED, RUNNING, SUCCEEDED. The status call that
    stores RUNNING also arms SUCCEEDED, and cancel keeps a terminal tick.
    An extra RUNNING tick lets a later cancel still see a live job.
    """
    files = JobStore(settings.state_dir / "mock" / "colab", nested=False)
    job = files.load(provider_job_id)
    files.save(
        job.model_copy(
            update={
                "state": JobState.RUNNING,
                "step_index": 1,
                "states": [
                    JobState.QUEUED,
                    JobState.RUNNING,
                    JobState.RUNNING,
                    JobState.SUCCEEDED,
                ],
            }
        )
    )


def _poll_until(
    client: Client,
    job_id: str,
    target: JobState,
    *,
    limit: int = 6,
) -> list[JobState]:
    seen: list[JobState] = []
    for _ in range(limit):
        seen.append(client.status(job_id).state)
        if seen[-1] is target:
            return seen
    return seen


def _poll_until_succeeded(client: Client, store: SqliteJobStore, job_id: str) -> list[JobState]:
    seen: list[JobState] = []
    for _ in range(6):
        current = client.status(job_id)
        seen.append(current.state)
        assert store.get(job_id).state is current.state
        if current.state is JobState.SUCCEEDED:
            return seen
    return seen


class _BlockingKaggle(MockKaggleProvider):
    """Real Kaggle mock whose submit waits until the test opens the gate."""

    def __init__(self, settings: Settings, gate: threading.Event) -> None:
        super().__init__(settings)
        self._gate = gate
        self.submits = 0
        self.status_calls = 0
        self.status_ids: list[str] = []
        self._lock = threading.Lock()

    def submit(self, spec: JobSpec) -> JobHandle:
        with self._lock:
            self.submits += 1
        if not self._gate.wait(2):
            raise TimeoutError("submit gate was not released")
        return super().submit(spec)

    def status(self, job_id: str) -> JobResult:
        with self._lock:
            self.status_calls += 1
            self.status_ids.append(job_id)
        return super().status(job_id)


def test_mock_flow_survives_a_new_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(state_dir=tmp_path, artifact_root=tmp_path)
    counter = _count_submits(monkeypatch)
    client_a = Client(settings)
    script = _script(tmp_path)

    auth = client_a.auth_status()
    assert {item.provider for item in auth} == {
        ProviderName.kaggle,
        ProviderName.colab,
        ProviderName.fake,
    }
    assert all(item.is_mock for item in auth)
    assert all(item.state is AuthState.authenticated for item in auth)
    assert all("TEST FIXTURE" in item.message for item in auth)

    quotas = client_a.get_quota()
    assert quotas
    assert all(item.is_fixture for item in quotas)
    assert all("TEST FIXTURE" in item.message for item in quotas)
    assert any(item.provider is ProviderName.kaggle for item in quotas)

    scheduled = client_a.schedule(ScheduleRequest(gpu_type=GpuType.T4, gpu_count=2))
    assert scheduled.provider is ProviderName.kaggle
    recommended = client_a.recommend_backend(gpu_type=GpuType.T4, gpu_count=2)
    assert recommended["provider"] == "kaggle"

    spec = JobSpec(
        script_path=script,
        gpu_type=GpuType.T4,
        gpu_count=2,
        idempotency_key=_IDEMPOTENCY_KEY,
    )
    handle, decision = client_a.submit_with_decision(spec)
    assert decision is not None
    assert decision.provider is ProviderName.kaggle
    assert handle.provider is ProviderName.kaggle
    assert counter["n"] == 1

    store = _store(settings)
    queued = store.get(handle.job_id)
    assert queued.state is JobState.QUEUED
    assert queued.provider is ProviderName.kaggle
    assert queued.provider_job_id
    assert queued.provider_job_id != handle.job_id
    assert queued.idempotency_key == _IDEMPOTENCY_KEY
    assert _pairs(store, handle.job_id) == [
        (JobState.SUBMITTING.value, JobState.QUEUED.value),
    ]

    seen = _poll_until_succeeded(client_a, store, handle.job_id)
    assert seen[0] is JobState.QUEUED
    assert JobState.RUNNING in seen
    assert seen[-1] is JobState.SUCCEEDED
    progressed = _pairs(store, handle.job_id)
    assert (JobState.QUEUED.value, JobState.RUNNING.value) in progressed
    assert (JobState.RUNNING.value, JobState.SUCCEEDED.value) in progressed

    dest = tmp_path / "artifacts"
    artifacts = client_a.download(handle.job_id, dest)
    assert len(artifacts) == 1
    assert artifacts[0].name == "result.txt"
    assert artifacts[0].local_path.resolve().is_relative_to(dest.resolve())
    assert (dest / "result.txt").read_text(encoding="utf-8") == MOCK_RESULT_TEXT
    saved = store.get(handle.job_id)
    assert saved.state is JobState.SUCCEEDED
    assert saved.artifacts == ["result.txt"]

    del client_a
    client_b = Client(settings)
    listed = client_b.list_jobs()
    match = next(row for row in listed if row["job_id"] == handle.job_id)
    assert match["state"] == "SUCCEEDED"
    assert match["provider"] == "kaggle"
    assert match["artifacts"] == ["result.txt"]

    submits_before_reuse = counter["n"]
    again = client_b.status(handle.job_id)
    assert again.job_id == handle.job_id
    assert again.state is JobState.SUCCEEDED
    assert again.provider is ProviderName.kaggle
    duplicate = client_b.submit_detailed(spec)
    assert duplicate.created is False
    assert duplicate.handle.job_id == handle.job_id
    assert counter["n"] == submits_before_reuse == 1


def test_crash_recovery_reconciles_without_resubmit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(state_dir=tmp_path, artifact_root=tmp_path)
    script = _script(tmp_path)
    client_a = Client(settings)
    handle = client_a.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.colab,
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=60,
            idempotency_key="crash-colab",
        )
    )
    store = _store(settings)
    inflight = store.get(handle.job_id)
    assert inflight.state is JobState.QUEUED
    assert inflight.provider_job_id
    # The first mock status tick still reports QUEUED. Consume it so reconcile
    # observes RUNNING, and do not poll this client through to SUCCEEDED.
    MockColabProvider(settings).status(inflight.provider_job_id)
    assert store.get(handle.job_id).state is JobState.QUEUED

    provider_job = MockColabProvider(settings).submit(
        JobSpec(
            script_path=_script(tmp_path, "unknown.py"),
            provider=ProviderName.colab,
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=60,
        )
    )
    now = datetime.now(UTC)
    old = now - timedelta(hours=2)
    unknown_id = uuid4().hex
    stale_id = uuid4().hex
    store.insert(
        JobRecord(
            job_id=unknown_id,
            provider=ProviderName.colab,
            account_id="colab-default",
            provider_job_id=provider_job.job_id,
            remote_id=provider_job.remote_id,
            state=JobState.UNKNOWN,
            created_at=now,
            updated_at=now,
            message="unknown with provider id",
        )
    )
    store.insert(
        JobRecord(
            job_id=stale_id,
            provider=ProviderName.colab,
            account_id="colab-default",
            state=JobState.SUBMITTING,
            created_at=old,
            updated_at=old,
            details={"submit_started_at": old.isoformat()},
            message="stale submit",
        )
    )

    del client_a
    counter = _count_submits(monkeypatch)
    client_b = Client(settings)
    reports = {item["job_id"]: item for item in client_b.reconcile_jobs()}
    assert counter["n"] == 0

    assert reports[handle.job_id]["ok"] is True
    assert reports[handle.job_id]["before"] == "QUEUED"
    assert reports[handle.job_id]["after"] == "RUNNING"
    assert store.get(handle.job_id).state is JobState.RUNNING

    assert reports[unknown_id]["ok"] is True
    assert reports[unknown_id]["before"] == "UNKNOWN"
    assert reports[unknown_id]["after"] == "QUEUED"
    resolved = store.get(unknown_id)
    assert resolved.state is JobState.QUEUED
    assert resolved.provider is ProviderName.colab
    assert resolved.provider_job_id == provider_job.job_id

    stale = store.get(stale_id)
    assert stale.state is JobState.UNKNOWN
    assert stale.provider_job_id is None
    assert "check the provider manually" in stale.message
    note = stale.details.get("note")
    assert isinstance(note, str)
    assert "check the provider manually" in note
    assert "will not be resubmitted" in note
    assert reports[stale_id]["ok"] is True
    assert reports[stale_id]["before"] == "SUBMITTING"
    assert reports[stale_id]["after"] == "UNKNOWN"
    assert "manually" in (reports[stale_id]["note"] or "")


def test_v02_json_migration_routes_to_the_mock_provider(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path, artifact_root=tmp_path)
    handle = MockKaggleProvider(settings).submit(
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
    rows = client.list_jobs()
    assert [row["job_id"] for row in rows] == [handle.job_id]
    assert rows[0]["state"] == "UNKNOWN"
    assert rows[0]["origin"] == "json_migration"
    assert rows[0]["provider"] == "kaggle"

    status = client.status(handle.job_id)
    assert status.job_id == handle.job_id
    assert status.provider is ProviderName.kaggle
    assert status.state is JobState.QUEUED
    assert status.is_mock is True
    stored = _store(settings).get(handle.job_id)
    assert stored.state is JobState.QUEUED
    assert stored.origin == "json_migration"
    # V0.2 indexed the provider id, so the imported ids are equal.
    assert stored.provider_job_id == handle.job_id
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == raw


def test_submit_timeout_is_unknown_then_reconcile_polls(tmp_path: Path) -> None:
    # Submit returns at the manager timeout. The gate opens afterwards so the
    # worker can still record the provider id. The wait cap is only a leak guard.
    gate = threading.Event()
    settings = Settings(
        state_dir=tmp_path,
        artifact_root=tmp_path,
        jobs=JobsSettings(submit_timeout_seconds=0.05),
    )
    provider = _BlockingKaggle(settings, gate)
    client = Client(settings, providers={ProviderName.kaggle: provider})
    spec = JobSpec(
        script_path=_script(tmp_path),
        gpu_type=GpuType.T4,
        gpu_count=2,
        idempotency_key="timeout-e2e",
    )
    try:
        with pytest.raises(SubmitOutcomeUnknownError) as caught:
            client.submit_with_decision(spec)
    finally:
        gate.set()

    assert caught.value.code == "submit_outcome_unknown"
    assert caught.value.details["job_state"] == "UNKNOWN"
    job_id = str(caught.value.details["job_id"])
    deadline = time.monotonic() + 2
    for thread in list(threading.enumerate()):
        if thread.name == "l2kc-submit":
            thread.join(timeout=2)
            assert not thread.is_alive()
    record = _store(settings).get(job_id)
    while record.provider_job_id is None and time.monotonic() < deadline:
        record = _store(settings).get(job_id)
    assert record.state is JobState.UNKNOWN
    assert record.provider_job_id
    assert record.provider_job_id != job_id
    assert provider.submits == 1

    reports = client.reconcile_jobs([job_id])
    assert provider.submits == 1
    assert provider.status_calls == 1
    assert reports[0]["ok"] is True
    assert reports[0]["before"] == "UNKNOWN"
    assert reports[0]["after"] == "QUEUED"
    assert _store(settings).get(job_id).state is JobState.QUEUED
    assert record.provider_job_id in provider.status_ids
    assert job_id not in provider.status_ids


def test_cli_submit_status_download_lists_succeeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("L2KC_CONFIG", raising=False)
    state = tmp_path / "state"
    monkeypatch.setenv("L2KC_STATE_DIR", str(state))
    config = tmp_path / "l2kc.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "mode": "mock",
                "allow_real_jobs": False,
                "state_dir": str(state),
                "artifact_root": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    script = _script(tmp_path)
    runner = CliRunner()

    def _invoke(*args: str) -> Result:
        return runner.invoke(app, ["--config", str(config), *args])

    submitted = _invoke("submit", str(script), "--gpu", "T4", "--gpu-count", "2")
    assert submitted.exit_code == 0, submitted.stderr
    assert "MOCK MODE" in submitted.stderr
    assert "provider=kaggle" in submitted.stderr
    job_id = submitted.stdout.splitlines()[0].strip()
    assert len(job_id) == 32

    states: list[str] = []
    for _ in range(6):
        current = _invoke("status", job_id)
        assert current.exit_code == 0, current.stderr
        state = next(
            line.split(": ", 1)[1].strip()
            for line in current.stdout.splitlines()
            if line.startswith("state: ")
        )
        states.append(state)
        if state == "SUCCEEDED":
            break
    assert states[0] == "QUEUED"
    assert "RUNNING" in states
    assert states[-1] == "SUCCEEDED"

    dest = tmp_path / "downloaded"
    fetched = _invoke("download", job_id, "--dest", str(dest))
    assert fetched.exit_code == 0, fetched.stderr
    assert (dest / "result.txt").is_file()
    assert (dest / "result.txt").read_text(encoding="utf-8") == MOCK_RESULT_TEXT

    listed = _invoke("jobs", "list", "--json")
    assert listed.exit_code == 0, listed.stderr
    rows = json.loads(listed.stdout)
    match = next(row for row in rows if row["job_id"] == job_id)
    assert match["state"] == "SUCCEEDED"
    assert match["provider"] == "kaggle"
    assert match["artifacts"] == ["result.txt"]


def test_cancel_mock_colab_persists_and_kaggle_reverts(tmp_path: Path) -> None:
    settings = Settings(state_dir=tmp_path, artifact_root=tmp_path)
    script = _script(tmp_path)
    client = Client(settings)
    store = _store(settings)
    queued = client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.colab,
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=60,
            idempotency_key="colab-queued",
        )
    )
    assert store.get(queued.job_id).state is JobState.QUEUED
    cancelled = client.cancel(queued.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert store.get(queued.job_id).state is JobState.CANCELLED
    queued_pairs = _pairs(store, queued.job_id)
    assert (JobState.QUEUED.value, JobState.CANCELLING.value) in queued_pairs
    assert (JobState.CANCELLING.value, JobState.CANCELLED.value) in queued_pairs

    running = client.submit(
        JobSpec(
            script_path=_script(tmp_path, "running.py"),
            provider=ProviderName.colab,
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=60,
            idempotency_key="colab-running",
        )
    )
    provider_job_id = store.get(running.job_id).provider_job_id
    assert provider_job_id
    _arm_colab_running(settings, provider_job_id)
    assert _poll_until(client, running.job_id, JobState.RUNNING)[-1] is JobState.RUNNING
    stopped = client.cancel(running.job_id)
    assert stopped.state is JobState.CANCELLED
    assert store.get(running.job_id).state is JobState.CANCELLED
    running_pairs = _pairs(store, running.job_id)
    assert (JobState.RUNNING.value, JobState.CANCELLING.value) in running_pairs
    assert (JobState.CANCELLING.value, JobState.CANCELLED.value) in running_pairs

    kaggle = client.submit(
        JobSpec(
            script_path=_script(tmp_path, "kaggle.py"),
            provider=ProviderName.kaggle,
            gpu_type=GpuType.T4,
            gpu_count=2,
            idempotency_key="kaggle-cancel",
        )
    )
    assert _poll_until(client, kaggle.job_id, JobState.RUNNING)[-1] is JobState.RUNNING
    with pytest.raises(NotSupportedError) as caught:
        client.cancel(kaggle.job_id)
    restored = store.get(kaggle.job_id)
    assert restored.state is JobState.RUNNING
    assert caught.value.code == "not_supported"
    assert caught.value.details["job_state"] == "RUNNING"
    reverted = _pairs(store, kaggle.job_id)
    assert (JobState.RUNNING.value, JobState.CANCELLING.value) in reverted
    assert (JobState.CANCELLING.value, JobState.RUNNING.value) in reverted


def test_eight_distinct_clients_share_one_idempotency_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(state_dir=tmp_path, artifact_root=tmp_path)
    SqliteJobStore(default_db_path(settings.state_dir))
    counter = _count_submits(monkeypatch)
    spec = JobSpec(
        script_path=_script(tmp_path),
        gpu_type=GpuType.T4,
        gpu_count=2,
        idempotency_key="shared-e2e",
    )
    barrier = threading.Barrier(8)
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _worker() -> None:
        try:
            barrier.wait(5)
            job_id = Client(settings).submit(spec).job_id
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
    assert counter["n"] == 1
    rows = _store(settings).list()
    assert len(rows) == 1
    assert rows[0].job_id == results[0]
    assert rows[0].idempotency_key == "shared-e2e"


def test_eight_distinct_clients_and_keys_insert_eight_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(state_dir=tmp_path, artifact_root=tmp_path)
    SqliteJobStore(default_db_path(settings.state_dir))
    counter = _count_submits(monkeypatch)
    script = _script(tmp_path)
    barrier = threading.Barrier(8)
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def _worker(index: int) -> None:
        try:
            barrier.wait(5)
            spec = JobSpec(
                script_path=script,
                gpu_type=GpuType.T4,
                gpu_count=2,
                idempotency_key=f"key-{index}",
            )
            job_id = Client(settings).submit(spec).job_id
            with lock:
                results.append(job_id)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
        assert not thread.is_alive()
    assert errors == []
    assert counter["n"] == 8
    assert len(results) == 8
    assert len(set(results)) == 8
    rows = _store(settings).list()
    assert len(rows) == 8
    assert {row.job_id for row in rows} == set(results)
    assert {row.idempotency_key for row in rows} == {f"key-{index}" for index in range(8)}
    database = default_db_path(settings.state_dir)
    with sqlite3.connect(database) as connection:
        check = connection.execute("PRAGMA integrity_check").fetchone()
    assert check is not None
    assert check[0] == "ok"
