"""SQLite job store: schema, transitions, dedupe, redaction, and JSON import."""

from __future__ import annotations

import json
import sqlite3
import threading
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from link2kaggle_colab.core.errors import (
    ConcurrentUpdateError,
    ConfigError,
    InvalidInputError,
    InvalidTransitionError,
    JobNotFoundError,
)
from link2kaggle_colab.core.models import JobState, ProviderName
from link2kaggle_colab.storage.sqlite_store import (
    SCHEMA_VERSION,
    JobRecord,
    SqliteJobStore,
    default_db_path,
    migrate_json_index,
)

_IMPORT_MESSAGE = (
    "imported from V0.2 JSON index; state will be reconciled from the provider"
)


def _record(**overrides: object) -> JobRecord:
    now = datetime.now(UTC)
    payload: dict[str, object] = {
        "job_id": uuid4().hex,
        "provider": ProviderName.kaggle,
        "state": JobState.PENDING,
        "message": "",
        "created_at": now,
        "updated_at": now,
    }
    payload.update(overrides)
    if "account_id" not in payload:
        provider = payload["provider"]
        name = provider.value if isinstance(provider, ProviderName) else str(provider)
        payload["account_id"] = f"{name}-default"
    return JobRecord.model_validate(payload)


def _database_bytes(path: Path) -> bytes:
    blob = bytearray(path.read_bytes())
    for suffix in ("-wal", "-shm"):
        extra = Path(str(path) + suffix)
        if extra.is_file():
            blob.extend(extra.read_bytes())
    return bytes(blob)


def _user_version(path: Path) -> int:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


def _journal_mode(path: Path) -> str:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    finally:
        conn.close()
    assert row is not None
    return str(row[0])


def test_default_db_path_stays_inside_state_dir(tmp_path: Path) -> None:
    assert default_db_path(tmp_path) == (tmp_path / "l2kc.db").resolve()


def test_schema_is_created_at_current_version(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "l2kc.db"
    store = SqliteJobStore(db)
    assert db.is_file()
    assert _user_version(db) == SCHEMA_VERSION == 2
    assert _journal_mode(db) == "wal"
    with store._connect() as conn:
        assert conn.in_transaction is False
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()
        busy = conn.execute("PRAGMA busy_timeout").fetchone()
        assert foreign_keys is not None and int(foreign_keys[0]) == 1
        assert busy is not None and int(busy[0]) == 30000
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        assert "spec" not in columns
        assert "env" not in columns
        assert "state" in columns
        assert "idempotency_key" in columns
        assert "spec_hash" in columns
        assert "artifacts_json" in columns
        assert "account_id" in columns
        index_names = {row[1] for row in conn.execute("PRAGMA index_list(jobs)")}
        assert "idx_jobs_state" in index_names
        assert "idx_jobs_spec_hash" in index_names
        assert "idx_jobs_idempotency_key" in index_names
        assert "idx_jobs_account_id" in index_names
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO job_events (job_id, from_state, to_state, at, message) "
                "VALUES (?, ?, ?, ?, ?)",
                ("missing", "PENDING", "FAILED", "", ""),
            )
        conn.execute("ROLLBACK")


def test_reopen_does_not_run_migration_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "l2kc.db"
    calls = {"n": 0}
    import link2kaggle_colab.storage.sqlite_store as store_module

    originals = store_module._MIGRATIONS

    def _wrap(migrate):  # type: ignore[no-untyped-def]
        def wrapped(conn: sqlite3.Connection) -> None:
            calls["n"] += 1
            migrate(conn)

        return wrapped

    monkeypatch.setattr(
        store_module,
        "_MIGRATIONS",
        tuple(_wrap(migrate) for migrate in originals),
    )
    store = SqliteJobStore(db)
    record = store.insert(_record(message="kept"))
    assert calls["n"] == len(originals)

    again = SqliteJobStore(db)
    assert calls["n"] == len(originals)
    assert again.get(record.job_id).message == "kept"
    assert _user_version(db) == SCHEMA_VERSION

    raw = sqlite3.connect(db, timeout=30, isolation_level=None)
    raw.execute("PRAGMA user_version = 0")
    raw.close()
    SqliteJobStore(db)
    assert calls["n"] == len(originals) * 2
    assert _user_version(db) == SCHEMA_VERSION
    assert SqliteJobStore(db).get(record.job_id).message == "kept"


def test_newer_user_version_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "l2kc.db"
    store = SqliteJobStore(db)
    record = store.insert(_record(state=JobState.RUNNING, message="stay"))
    raw = sqlite3.connect(db, timeout=30, isolation_level=None)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(ConfigError) as caught:
        SqliteJobStore(db)
    assert caught.value.code == "config_error"
    assert _user_version(db) == 99
    raw = sqlite3.connect(db, timeout=30, isolation_level=None)
    try:
        row = raw.execute(
            "SELECT state, message FROM jobs WHERE job_id = ?",
            (record.job_id,),
        ).fetchone()
    finally:
        raw.close()
    assert row == ("RUNNING", "stay")


def test_insert_get_list_and_filters(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    early = datetime(2026, 10, 24, 1, 0, tzinfo=UTC)
    late = early + timedelta(minutes=5)
    first = store.insert(
        _record(
            provider=ProviderName.kaggle,
            state=JobState.PENDING,
            created_at=late,
            updated_at=late,
            idempotency_key="first",
        )
    )
    second = store.insert(
        _record(
            provider=ProviderName.colab,
            state=JobState.RUNNING,
            created_at=early,
            updated_at=early,
            spec_hash="abc",
        )
    )
    assert store.get(first.job_id).job_id == first.job_id
    assert store.get(first.job_id).created_at == late
    ordered = store.list()
    assert [item.job_id for item in ordered] == [second.job_id, first.job_id]
    assert [item.job_id for item in store.list(states=[JobState.RUNNING])] == [second.job_id]
    assert [item.job_id for item in store.list(provider="colab")] == [second.job_id]
    assert store.list(states=[JobState.FAILED]) == []
    assert len(store.list(limit=1)) == 1
    assert store.list(limit=1)[0].job_id == second.job_id
    assert store.find_by_idempotency_key("first").job_id == first.job_id  # type: ignore[union-attr]
    assert store.find_by_idempotency_key("missing") is None
    with pytest.raises(JobNotFoundError):
        store.get("not-a-job-id")
    with pytest.raises(JobNotFoundError):
        store.get("A" * 32)
    with pytest.raises(JobNotFoundError):
        store.get(uuid4().hex)


def test_transition_writes_event_and_invalid_is_unchanged(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    record = store.insert(_record(message="start", version=0))
    moved = store.transition(
        record.job_id,
        JobState.SUBMITTING,
        message="handed off",
        provider_job_id="prov-1",
        submitted_at=datetime(2026, 10, 24, 2, 0, tzinfo=UTC),
    )
    assert moved.state is JobState.SUBMITTING
    assert moved.version == 1
    assert moved.provider_job_id == "prov-1"
    assert moved.message == "handed off"
    events = store.events(record.job_id)
    assert len(events) == 1
    assert events[0]["from_state"] == JobState.PENDING.value
    assert events[0]["to_state"] == JobState.SUBMITTING.value
    assert events[0]["message"] == "handed off"

    refreshed = store.transition(
        record.job_id,
        JobState.SUBMITTING,
        message="still submitting",
        remote_id="remote-9",
    )
    assert refreshed.state is JobState.SUBMITTING
    assert refreshed.version == 2
    assert refreshed.remote_id == "remote-9"
    assert len(store.events(record.job_id)) == 1

    before = store.get(record.job_id)
    with pytest.raises(InvalidTransitionError):
        store.transition(record.job_id, JobState.CANCELLED, message="nope")
    after = store.get(record.job_id)
    assert after.state is before.state
    assert after.version == before.version
    assert after.message == before.message
    assert len(store.events(record.job_id)) == 1


@pytest.mark.parametrize(
    ("state", "target"),
    [
        (JobState.SUCCEEDED, JobState.RUNNING),
        (JobState.PENDING, JobState.QUEUED),
    ],
)
def test_store_rejects_illegal_transition_without_an_event(
    tmp_path: Path,
    state: JobState,
    target: JobState,
) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    record = store.insert(_record(state=state, message="kept"))
    with pytest.raises(InvalidTransitionError):
        store.transition(record.job_id, target, message="nope")
    current = store.get(record.job_id)
    assert current.state is state
    assert current.version == record.version
    assert current.message == "kept"
    assert store.events(record.job_id) == []


def test_expected_version_mismatch_is_concurrent_update(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    record = store.insert(_record())
    store.transition(record.job_id, JobState.SUBMITTING)
    with pytest.raises(ConcurrentUpdateError) as caught:
        store.transition(record.job_id, JobState.QUEUED, expected_version=0)
    assert caught.value.code == "concurrent_update"
    current = store.get(record.job_id)
    assert current.state is JobState.SUBMITTING
    assert current.version == 1
    queued = store.transition(record.job_id, JobState.QUEUED, expected_version=1)
    assert queued.state is JobState.QUEUED
    assert queued.version == 2


def test_unknown_fields_do_not_change_the_row(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    record = store.insert(_record(message="plain"))
    with pytest.raises(InvalidInputError):
        store.transition(record.job_id, JobState.SUBMITTING, spec="full script")
    with pytest.raises(InvalidInputError):
        store.update_fields(record.job_id, env="SECRET=1")
    current = store.get(record.job_id)
    assert current.state is JobState.PENDING
    assert current.version == 0
    assert current.message == "plain"
    updated = store.update_fields(record.job_id, artifacts=["out/model.bin"], exit_code=None)
    assert updated.artifacts == ["out/model.bin"]
    assert updated.state is JobState.PENDING
    assert updated.version == 1
    assert store.events(record.job_id) == []


def test_secrets_are_redacted_in_the_database_file(tmp_path: Path) -> None:
    github = "ghp_" + ("A" * 36)
    kaggle = "KGAT_" + ("b" * 20)
    bearer = "c" * 24
    db = tmp_path / "l2kc.db"
    store = SqliteJobStore(db)
    record = store.insert(
        _record(
            message="token=" + github,
            details={
                "note": kaggle,
                "header": "Authorization: Bearer " + bearer,
                "label": "visible",
            },
        )
    )
    store.transition(record.job_id, JobState.FAILED, message="still token=" + github)
    blob = _database_bytes(db)
    assert github.encode() not in blob
    assert kaggle.encode() not in blob
    assert bearer.encode() not in blob
    assert b"[REDACTED]" in blob
    assert b"visible" in blob
    stored = store.get(record.job_id)
    assert github not in stored.message
    assert kaggle not in json.dumps(stored.details)
    assert bearer not in json.dumps(stored.details)
    assert stored.details["label"] == "visible"


def test_transition_stores_redacted_env_and_mapping_secrets(tmp_path: Path) -> None:
    kaggle = "0123456789abcdef0123456789abcdef"
    hf = "hf_abcdefghijklmnop"
    db_value = "hunter2hunter2"
    aws = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    api_value = "plainvalue123"
    nested_value = "abcdefgh"
    message = (
        f"KAGGLE_KEY={kaggle} HF_TOKEN={hf} "
        f'export DB_PASSWORD="{db_value}" AWS_SECRET_ACCESS_KEY={aws}'
    )
    details = {
        "api_key": api_value,
        "nested": {"client_secret": nested_value},
        "idempotency_key": "k-1",
        "gpu_type": "T4",
        "count": 3,
    }
    db = tmp_path / "l2kc.db"
    store = SqliteJobStore(db)
    record = store.insert(_record())
    store.transition(record.job_id, JobState.FAILED, message=message, details=details)
    conn = sqlite3.connect(db, timeout=30)
    try:
        job = conn.execute(
            "SELECT message, details_json FROM jobs WHERE job_id = ?",
            (record.job_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT message FROM job_events WHERE job_id = ?",
            (record.job_id,),
        ).fetchone()
    finally:
        conn.close()
    assert job is not None
    assert event is not None
    stored = f"{job[0]}\n{job[1]}\n{event[0]}"
    for secret in (kaggle, hf, db_value, aws, api_value, nested_value):
        assert secret not in stored
    assert "[REDACTED]" in job[0]
    assert "[REDACTED]" in job[1]
    assert "[REDACTED]" in event[0]
    assert "k-1" in job[1]
    assert "T4" in job[1]
    assert "idempotency_key" in job[1]


def test_idempotency_key_is_unique(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    first = store.insert(_record(idempotency_key="same-key"))
    with pytest.raises(InvalidInputError):
        store.insert(_record(idempotency_key="same-key"))
    assert store.get(first.job_id).idempotency_key == "same-key"
    assert len(store.list()) == 1


def _race_create(
    db: Path,
    records: list[JobRecord],
    *,
    dedupe_spec_hash: bool,
) -> list[tuple[str, bool]]:
    SqliteJobStore(db)
    barrier = threading.Barrier(len(records))
    results: list[tuple[str, bool]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(record: JobRecord) -> None:
        try:
            barrier.wait(timeout=30)
            store = SqliteJobStore(db)
            got, created = store.create_or_get(record, dedupe_spec_hash=dedupe_spec_hash)
            with lock:
                results.append((got.job_id, created))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(record,)) for record in records]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert [thread for thread in threads if thread.is_alive()] == []
    assert errors == []
    return results


def test_create_or_get_reuses_matching_idempotency_spec_hash(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    first, created = store.create_or_get(
        _record(
            idempotency_key="same-key",
            spec_hash="hash-a",
            state=JobState.RUNNING,
            version=2,
            message="original",
        ),
        dedupe_spec_hash=False,
    )
    assert created is True
    again, created = store.create_or_get(
        _record(
            idempotency_key="same-key",
            spec_hash="hash-a",
            state=JobState.PENDING,
            message="ignored",
        ),
        dedupe_spec_hash=False,
    )
    assert created is False
    assert again.job_id == first.job_id
    assert again.state is JobState.RUNNING
    assert again.version == 2
    assert again.spec_hash == "hash-a"
    assert again.message == "original"
    assert [row.job_id for row in store.list()] == [first.job_id]


def test_create_or_get_rejects_a_different_spec_hash(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    stored, created = store.create_or_get(
        _record(
            idempotency_key="same-key",
            spec_hash="hash-a",
            state=JobState.QUEUED,
            version=4,
            message="keep",
        ),
        dedupe_spec_hash=False,
    )
    assert created is True
    with pytest.raises(InvalidInputError) as caught:
        store.create_or_get(
            _record(idempotency_key="same-key", spec_hash="hash-b", state=JobState.PENDING),
            dedupe_spec_hash=False,
        )
    assert caught.value.code == "invalid_input"
    assert caught.value.message == (
        "idempotency key is already used by a job with a different spec"
    )
    assert caught.value.details == {
        "field": "idempotency_key",
        "reason": "spec_mismatch",
        "existing_job_id": stored.job_id,
        "existing_spec_hash": "hash-a",
        "requested_spec_hash": "hash-b",
    }
    current = store.get(stored.job_id)
    assert current.version == stored.version == 4
    assert current.state is JobState.QUEUED
    assert current.spec_hash == "hash-a"
    assert current.message == "keep"
    assert [row.job_id for row in store.list()] == [stored.job_id]


def test_create_or_get_rejects_null_spec_hash_mismatch(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    missing, created = store.create_or_get(
        _record(idempotency_key="null-key", spec_hash=None, state=JobState.PENDING, version=1),
        dedupe_spec_hash=False,
    )
    assert created is True
    with pytest.raises(InvalidInputError) as caught:
        store.create_or_get(
            _record(idempotency_key="null-key", spec_hash="hash-new"),
            dedupe_spec_hash=False,
        )
    assert caught.value.details == {
        "field": "idempotency_key",
        "reason": "spec_mismatch",
        "existing_job_id": missing.job_id,
        "existing_spec_hash": None,
        "requested_spec_hash": "hash-new",
    }
    kept = store.get(missing.job_id)
    assert kept.spec_hash is None
    assert kept.version == 1
    assert kept.state is JobState.PENDING

    present, created = store.create_or_get(
        _record(idempotency_key="present-key", spec_hash="hash-old", state=JobState.RUNNING),
        dedupe_spec_hash=False,
    )
    assert created is True
    with pytest.raises(InvalidInputError) as other:
        store.create_or_get(
            _record(idempotency_key="present-key", spec_hash=None),
            dedupe_spec_hash=False,
        )
    assert other.value.details["reason"] == "spec_mismatch"
    assert other.value.details["existing_job_id"] == present.job_id
    assert other.value.details["existing_spec_hash"] == "hash-old"
    assert other.value.details["requested_spec_hash"] is None
    assert store.get(present.job_id).state is JobState.RUNNING
    assert store.get(present.job_id).spec_hash == "hash-old"
    assert len(store.list()) == 2


def test_create_or_get_idempotency_key_is_race_free(tmp_path: Path) -> None:
    db = tmp_path / "l2kc.db"
    records = [
        _record(
            idempotency_key="shared-key",
            spec_hash="shared-spec",
            state=JobState.PENDING,
        )
        for _ in range(16)
    ]
    results = _race_create(db, records, dedupe_spec_hash=False)
    assert len(results) == 16
    assert len({job_id for job_id, _created in results}) == 1
    assert sum(1 for _job_id, created in results if created) == 1
    stored = SqliteJobStore(db).list()
    assert len(stored) == 1
    assert stored[0].job_id == results[0][0]
    assert stored[0].idempotency_key == "shared-key"
    assert stored[0].spec_hash == "shared-spec"


def test_create_or_get_spec_hash_dedupe_is_race_free(tmp_path: Path) -> None:
    db = tmp_path / "l2kc.db"
    records = [
        _record(spec_hash="same-spec", state=JobState.QUEUED, idempotency_key=None)
        for _ in range(16)
    ]
    results = _race_create(db, records, dedupe_spec_hash=True)
    assert len(results) == 16
    assert len({job_id for job_id, _created in results}) == 1
    assert sum(1 for _job_id, created in results if created) == 1
    stored = SqliteJobStore(db).list()
    assert len(stored) == 1
    assert stored[0].spec_hash == "same-spec"
    assert stored[0].state is JobState.QUEUED


def test_terminal_spec_hash_does_not_block_a_new_job(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    done = store.insert(_record(state=JobState.SUCCEEDED, spec_hash="spec-1"))
    fresh = _record(state=JobState.PENDING, spec_hash="spec-1")
    got, created = store.create_or_get(fresh, dedupe_spec_hash=True)
    assert created is True
    assert got.job_id == fresh.job_id
    assert got.job_id != done.job_id
    running = store.insert(_record(state=JobState.RUNNING, spec_hash="spec-2"))
    duplicate = _record(state=JobState.PENDING, spec_hash="spec-2")
    reused, created = store.create_or_get(duplicate, dedupe_spec_hash=True)
    assert created is False
    assert reused.job_id == running.job_id
    key_owner = store.insert(
        _record(idempotency_key="key-1", spec_hash="spec-3", state=JobState.RUNNING)
    )
    other = store.insert(_record(spec_hash="spec-3", state=JobState.QUEUED))
    challenger = _record(
        idempotency_key="key-1",
        spec_hash="spec-3",
        state=JobState.PENDING,
    )
    found, created = store.create_or_get(challenger, dedupe_spec_hash=True)
    assert created is False
    assert found.job_id == key_owner.job_id
    assert found.job_id != other.job_id


def test_uncommitted_write_is_invisible_after_close(tmp_path: Path) -> None:
    db = tmp_path / "l2kc.db"
    store = SqliteJobStore(db)
    record = store.insert(_record(state=JobState.PENDING, message="consistent"))
    raw = sqlite3.connect(db, timeout=30, isolation_level=None)
    raw.execute("BEGIN IMMEDIATE")
    raw.execute(
        "UPDATE jobs SET state = ?, version = ?, message = ? WHERE job_id = ?",
        ("FAILED", 99, "partial", record.job_id),
    )
    raw.close()
    reopened = SqliteJobStore(db)
    current = reopened.get(record.job_id)
    assert current.state is JobState.PENDING
    assert current.version == record.version
    assert current.message == "consistent"


def _write_index(directory: Path, job_id: str, provider: str, **extra: object) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"job_id": job_id, "provider": provider}
    payload.update(extra)
    path = directory / f"{job_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_json_migration_imports_v02_records(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    kaggle_id = uuid4().hex
    colab_id = uuid4().hex
    fake_id = uuid4().hex
    paths = [
        _write_index(jobs, kaggle_id, "kaggle"),
        _write_index(jobs, colab_id, "colab"),
        _write_index(jobs, fake_id, "fake"),
    ]
    before = {path: path.read_bytes() for path in paths}
    imported = migrate_json_index(tmp_path)
    assert imported == 3
    store = SqliteJobStore(default_db_path(tmp_path))
    for job_id, provider in (
        (kaggle_id, ProviderName.kaggle),
        (colab_id, ProviderName.colab),
        (fake_id, ProviderName.fake),
    ):
        record = store.get(job_id)
        assert record.state is JobState.UNKNOWN
        assert record.provider is provider
        assert record.provider_job_id == job_id
        assert record.origin == "json_migration"
        assert record.message == _IMPORT_MESSAGE
        assert record.version == 0
    assert migrate_json_index(tmp_path) == 0
    assert len(store.list()) == 3
    for path, content in before.items():
        assert path.read_bytes() == content


def test_json_migration_skips_bad_files_and_keeps_existing_rows(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    kept_id = uuid4().hex
    store = SqliteJobStore(default_db_path(tmp_path))
    store.insert(
        _record(
            job_id=kept_id,
            state=JobState.RUNNING,
            message="already stored",
            origin="sqlite",
            provider=ProviderName.kaggle,
        )
    )
    _write_index(jobs, kept_id, "colab")
    valid_id = uuid4().hex
    valid_path = _write_index(jobs, valid_id, "kaggle")
    original = valid_path.read_bytes()
    (jobs / "notes.json").write_text("{}", encoding="utf-8")
    (jobs / f"{'ab' * 16}.json.tmp").write_text("{}", encoding="utf-8")
    (jobs / f"{'cd' * 16}.JSON").write_text("{}", encoding="utf-8")
    (jobs / f"{'ef' * 16}.json").write_text("{", encoding="utf-8")
    mismatch_id = "11" * 16
    (jobs / f"{mismatch_id}.json").write_text(
        json.dumps({"job_id": "22" * 16, "provider": "kaggle"}),
        encoding="utf-8",
    )
    (jobs / f"{'33' * 16}.json").write_text(
        json.dumps({"job_id": "33" * 16, "provider": "openai"}),
        encoding="utf-8",
    )
    (jobs / f"{'44' * 16}.json").write_text(
        json.dumps({"job_id": "44" * 16, "provider": "kaggle", "token": "nope"}),
        encoding="utf-8",
    )
    (jobs / "README.txt").write_text("ignore", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = outside / "secret.json"
    escaped.write_text(
        json.dumps({"job_id": "55" * 16, "provider": "kaggle"}),
        encoding="utf-8",
    )
    (jobs / f"{'55' * 16}.json").symlink_to(escaped)
    (jobs / f"{'66' * 16}.json").symlink_to(tmp_path / "missing-target.json")
    (jobs / f"{'77' * 16}.json").mkdir()

    imported = store.migrate_json_index(tmp_path)
    assert imported == 1
    assert store.migrate_json_index(tmp_path) == 0
    kept = store.get(kept_id)
    assert kept.state is JobState.RUNNING
    assert kept.origin == "sqlite"
    assert kept.message == "already stored"
    assert kept.provider is ProviderName.kaggle
    brought = store.get(valid_id)
    assert brought.state is JobState.UNKNOWN
    assert brought.origin == "json_migration"
    assert valid_path.read_bytes() == original
    assert escaped.read_text(encoding="utf-8").startswith("{")
    assert len(store.list()) == 2
    with pytest.raises(JobNotFoundError):
        store.get("55" * 16)


def test_operations_do_not_leak_connections(tmp_path: Path) -> None:
    db = tmp_path / "l2kc.db"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        store = SqliteJobStore(db)
        record = store.insert(_record())
        store.transition(record.job_id, JobState.SUBMITTING, message="ok")
        store.get(record.job_id)
        store.list(states=[JobState.SUBMITTING], provider="kaggle", limit=10)
        store.update_fields(record.job_id, artifacts=["a.txt"])
        store.events(record.job_id)
        store.find_by_idempotency_key("none")
        store.create_or_get(_record(idempotency_key="once"), dedupe_spec_hash=False)
        store.create_or_get(_record(idempotency_key="once"), dedupe_spec_hash=True)


# Historical v1 schema, copied from ``_migrate_to_1``. Kept here so a later
# edit to that function cannot pretend this database already had ``account_id``.
_V1_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_job_id TEXT,
    remote_id TEXT,
    state TEXT NOT NULL CHECK (
        state IN (
            'PENDING', 'SUBMITTING', 'QUEUED', 'RUNNING', 'SUCCEEDED',
            'FAILED', 'CANCELLING', 'CANCELLED', 'UNKNOWN'
        )
    ),
    idempotency_key TEXT,
    spec_hash TEXT,
    message TEXT NOT NULL DEFAULT '',
    exit_code INTEGER,
    artifacts_json TEXT NOT NULL DEFAULT '[]',
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT,
    updated_at TEXT,
    submitted_at TEXT,
    finished_at TEXT,
    version INTEGER NOT NULL DEFAULT 0,
    origin TEXT NOT NULL DEFAULT 'sqlite'
)
"""


def _create_v1_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(_V1_JOBS_SQL)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency_key
            ON jobs(idempotency_key)
            WHERE idempotency_key IS NOT NULL
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_spec_hash ON jobs(spec_hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                from_state TEXT,
                to_state TEXT,
                at TEXT,
                message TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id)"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


def test_v1_database_backfills_provider_default_account(tmp_path: Path) -> None:
    db = tmp_path / "l2kc.db"
    _create_v1_database(db)
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(db)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        assert "account_id" not in columns
        conn.executemany(
            "INSERT INTO jobs (job_id, provider, state, message, created_at, updated_at) "
            "VALUES (?, ?, 'QUEUED', 'legacy', ?, ?)",
            [("a" * 32, "kaggle", now, now), ("b" * 32, "colab", now, now)],
        )
        conn.commit()
    finally:
        conn.close()

    store = SqliteJobStore(db)
    assert _user_version(db) == 2
    rows = {record.provider.value: record for record in store.list()}
    assert rows["kaggle"].account_id == "kaggle-default"
    assert rows["colab"].account_id == "colab-default"
    assert store.list(account_id="kaggle-default")[0].job_id == "a" * 32
    assert store.list(account_id="colab-default")[0].job_id == "b" * 32
    assert store.list(account_id="kaggle-b") == []


def test_list_jobs_can_filter_by_account(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    kaggle = store.insert(_record(account_id="kaggle-a", message="a"))
    store.insert(_record(account_id="kaggle-b", provider=ProviderName.kaggle, message="b"))
    listed = store.list(account_id="kaggle-a")
    assert [row.job_id for row in listed] == [kaggle.job_id]
    assert listed[0].account_id == "kaggle-a"


def test_idempotency_key_reused_by_another_account_conflicts(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    first, created = store.create_or_get(
        _record(idempotency_key="shared-key", account_id="kaggle-a"),
        dedupe_spec_hash=False,
    )
    assert created is True
    with pytest.raises(ConfigError) as caught:
        store.create_or_get(
            _record(idempotency_key="shared-key", account_id="kaggle-b"),
            dedupe_spec_hash=False,
        )
    assert caught.value.code == "config_error"
    assert caught.value.details["reason"] == "account_conflict"
    assert caught.value.details["field"] == "idempotency_key"
    assert "shared-key" not in str(caught.value)
    assert "shared-key" not in json.dumps(caught.value.to_dict())
    rows = store.list()
    assert [row.job_id for row in rows] == [first.job_id]
    assert rows[0].account_id == "kaggle-a"


def test_spec_hash_dedupe_does_not_cross_accounts(tmp_path: Path) -> None:
    store = SqliteJobStore(tmp_path / "l2kc.db")
    first, created = store.create_or_get(
        _record(spec_hash="same-spec", state=JobState.QUEUED, account_id="kaggle-a"),
        dedupe_spec_hash=True,
    )
    assert created is True
    second, created_other = store.create_or_get(
        _record(spec_hash="same-spec", state=JobState.PENDING, account_id="kaggle-b"),
        dedupe_spec_hash=True,
    )
    assert created_other is True
    assert second.job_id != first.job_id
    assert {row.account_id for row in store.list()} == {"kaggle-a", "kaggle-b"}
