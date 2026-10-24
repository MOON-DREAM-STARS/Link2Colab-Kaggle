"""SQLite job store.

The database holds job rows and state-change events only. It does not store
script contents, environment variables, or credential material. ``message`` and
every string inside ``details`` pass through :func:`redact` / :func:`redact_value`
before they are written.

Writers use ``BEGIN IMMEDIATE`` so idempotency and spec-hash dedupe stay
race-free across threads and processes. A connection is opened for one
operation and closed before the method returns.

``user_version`` is the schema version. Migrations are ordered, idempotent, and
applied inside one transaction. A database newer than :data:`SCHEMA_VERSION`
is refused.

V0.2 ``state_dir/jobs/<32 hex>.json`` files can be imported with
:meth:`SqliteJobStore.migrate_json_index`. Imported rows start at ``UNKNOWN``
and are reconciled later by polling the provider. Import uses ``INSERT OR
IGNORE``, leaves the JSON files alone, and skips links that escape the jobs
directory.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from link2kaggle_colab.core.errors import (
    ConcurrentUpdateError,
    ConfigError,
    InvalidInputError,
    JobNotFoundError,
    PathSecurityError,
)
from link2kaggle_colab.core.models import JobState, ProviderName
from link2kaggle_colab.core.paths import ensure_within, safe_join
from link2kaggle_colab.core.redact import redact, redact_value
from link2kaggle_colab.jobs.state_machine import validate_transition

_LOG = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_CONNECT_TIMEOUT_SECONDS = 30.0

_JOB_ID = re.compile(r"^[a-f0-9]{32}$")
_INDEX_NAME = re.compile(r"^[a-f0-9]{32}\.json$")

_IMPORT_MESSAGE = (
    "imported from V0.2 JSON index; state will be reconciled from the provider"
)

_TERMINAL_STATES = (
    JobState.SUCCEEDED.value,
    JobState.FAILED.value,
    JobState.CANCELLED.value,
)

_MUTABLE_FIELDS = (
    "provider_job_id",
    "remote_id",
    "exit_code",
    "artifacts",
    "details",
    "finished_at",
    "submitted_at",
    "message",
)

_INSERT_SQL = (
    "INSERT INTO jobs ("
    "job_id, provider, account_id, provider_job_id, remote_id, state, idempotency_key, "
    "spec_hash, message, exit_code, artifacts_json, details_json, created_at, "
    "updated_at, submitted_at, finished_at, version, origin"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_INSERT_IGNORE_SQL = (
    "INSERT OR IGNORE INTO jobs ("
    "job_id, provider, account_id, provider_job_id, remote_id, state, idempotency_key, "
    "spec_hash, message, exit_code, artifacts_json, details_json, created_at, "
    "updated_at, submitted_at, finished_at, version, origin"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_SqlValue = str | int | None


class JobRecord(BaseModel):
    """One ``jobs`` row.

    Timestamps are timezone-aware UTC datetimes. SQLite stores them as
    ISO-8601 strings with microsecond precision. ``artifacts`` and ``details``
    mirror ``artifacts_json`` and ``details_json``.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    provider: ProviderName
    account_id: str
    provider_job_id: str | None = None
    remote_id: str | None = None
    state: JobState
    idempotency_key: str | None = None
    spec_hash: str | None = None
    message: str = ""
    exit_code: int | None = None
    artifacts: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime | None = None
    finished_at: datetime | None = None
    version: int = Field(default=0, ge=0)
    origin: str = "sqlite"

    @field_validator("created_at", "updated_at", "submitted_at", "finished_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("artifacts")
    @classmethod
    def _artifact_names(cls, value: list[str]) -> list[str]:
        if any(not isinstance(item, str) for item in value):
            raise ValueError("artifacts must be a list of strings")
        return value


class _V02Index(BaseModel):
    """V0.2 ``JobIndexRecord`` shape: job id plus provider, nothing else."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    provider: ProviderName


def default_db_path(state_dir: Path) -> Path:
    """Return ``state_dir / 'l2kc.db'``, refusing a path that escapes ``state_dir``."""
    return safe_join(state_dir, "l2kc.db")


def migrate_json_index(state_dir: Path) -> int:
    """Import a V0.2 JSON index into :func:`default_db_path`.

    This is the same operation as :meth:`SqliteJobStore.migrate_json_index`.
    """
    store = SqliteJobStore(default_db_path(state_dir))
    return store.migrate_json_index(state_dir)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _format_ts(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidInputError(
            "timestamp must be timezone-aware",
            details={"reason": "naive datetime"},
        )
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_ts(value: str | None, field: str) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(
            "stored timestamp is not ISO-8601",
            details={"field": field},
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigError(
            "stored timestamp is not timezone-aware",
            details={"field": field},
        )
    return parsed.astimezone(UTC)


def _require_job_id(job_id: object) -> str:
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise JobNotFoundError(
            "unknown job id",
            details={"reason": "job id is not a uuid hex"},
        )
    return job_id


def _read_user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    if row is None:
        return 0
    return int(row[0])


def _reject_unknown_fields(fields: dict[str, object]) -> None:
    unknown = sorted(set(fields) - set(_MUTABLE_FIELDS))
    if unknown:
        raise InvalidInputError(
            "unknown job field",
            details={"fields": unknown},
        )


def _clean_message(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidInputError("message must be a string", details={"field": "message"})
    return redact(value)


def _clean_optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidInputError(
            f"{field} must be a string",
            details={"field": field},
        )
    return value


def _clean_exit_code(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(
            "exit_code must be an integer",
            details={"field": "exit_code"},
        )
    return value


def _clean_timestamp(value: object, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed: datetime | str = datetime.fromisoformat(value)
        except ValueError as exc:
            raise InvalidInputError(
                "timestamp must be an ISO datetime",
                details={"field": field},
            ) from exc
        value = parsed
    if not isinstance(value, datetime):
        raise InvalidInputError(
            "timestamp must be a datetime",
            details={"field": field},
        )
    return _format_ts(value)


def _clean_artifacts(value: object) -> str:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InvalidInputError(
            "artifacts must be a list of strings",
            details={"field": "artifacts"},
        )
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _clean_details(value: object) -> str:
    if not isinstance(value, dict):
        raise InvalidInputError("details must be a mapping", details={"field": "details"})
    redacted = redact_value(value)
    if not isinstance(redacted, dict):
        raise InvalidInputError("details must be a mapping", details={"field": "details"})
    try:
        return json.dumps(redacted, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            "details must be JSON serializable",
            details={"field": "details"},
        ) from exc


def _sql_value(name: str, value: object) -> _SqlValue:
    if name in {"provider_job_id", "remote_id"}:
        return _clean_optional_str(value, name)
    if name == "exit_code":
        return _clean_exit_code(value)
    if name in {"finished_at", "submitted_at"}:
        return _clean_timestamp(value, name)
    if name == "artifacts":
        return _clean_artifacts(value)
    if name == "details":
        return _clean_details(value)
    if name == "message":
        return _clean_message(value)
    raise InvalidInputError("unknown job field", details={"fields": [name]})


def _column_for(name: str) -> str:
    if name == "artifacts":
        return "artifacts_json"
    if name == "details":
        return "details_json"
    if name in _MUTABLE_FIELDS:
        return name
    raise InvalidInputError("unknown job field", details={"fields": [name]})


def _field_assignments(fields: dict[str, object]) -> tuple[list[str], list[_SqlValue]]:
    clauses: list[str] = []
    params: list[_SqlValue] = []
    for name in _MUTABLE_FIELDS:
        if name not in fields:
            continue
        clauses.append(_column_for(name) + " = ?")
        params.append(_sql_value(name, fields[name]))
    return clauses, params


def _clean_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise InvalidInputError("limit must be an integer", details={"field": "limit"})
    if limit < 0:
        raise InvalidInputError(
            "limit must be non-negative",
            details={"field": "limit"},
        )
    return limit


def _clean_expected_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(
            "expected_version must be an integer",
            details={"field": "expected_version"},
        )
    if value < 0:
        raise InvalidInputError(
            "expected_version must be non-negative",
            details={"field": "expected_version"},
        )
    return value


def _coerce_state(value: object) -> JobState:
    if isinstance(value, JobState):
        return value
    try:
        return JobState(str(value))
    except ValueError as exc:
        raise InvalidInputError(
            "unknown job state",
            details={"state": str(value)},
        ) from exc


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    job_id = str(row["job_id"])
    try:
        artifacts = json.loads(row["artifacts_json"])
        details = json.loads(row["details_json"])
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "job row has invalid JSON",
            details={"job_id": job_id},
        ) from exc
    if not isinstance(artifacts, list) or not isinstance(details, dict):
        raise ConfigError("job row has invalid JSON", details={"job_id": job_id})
    created_at = _parse_ts(row["created_at"], "created_at")
    updated_at = _parse_ts(row["updated_at"], "updated_at")
    if created_at is None or updated_at is None:
        raise ConfigError(
            "job row is missing timestamps",
            details={"job_id": job_id},
        )
    return JobRecord.model_validate(
        {
            "job_id": job_id,
            "provider": row["provider"],
            "account_id": row["account_id"],
            "provider_job_id": row["provider_job_id"],
            "remote_id": row["remote_id"],
            "state": row["state"],
            "idempotency_key": row["idempotency_key"],
            "spec_hash": row["spec_hash"],
            "message": row["message"] if row["message"] is not None else "",
            "exit_code": row["exit_code"],
            "artifacts": artifacts,
            "details": details,
            "created_at": created_at,
            "updated_at": updated_at,
            "submitted_at": _parse_ts(row["submitted_at"], "submitted_at"),
            "finished_at": _parse_ts(row["finished_at"], "finished_at"),
            "version": row["version"],
            "origin": row["origin"],
        }
    )


def _record_values(record: JobRecord) -> tuple[object, ...]:
    return (
        record.job_id,
        record.provider.value,
        record.account_id,
        record.provider_job_id,
        record.remote_id,
        record.state.value,
        record.idempotency_key,
        record.spec_hash,
        redact(record.message),
        record.exit_code,
        _clean_artifacts(record.artifacts),
        _clean_details(record.details),
        _format_ts(record.created_at),
        _format_ts(record.updated_at),
        _format_ts(record.submitted_at) if record.submitted_at is not None else None,
        _format_ts(record.finished_at) if record.finished_at is not None else None,
        record.version,
        record.origin,
    )


def _fetch_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise JobNotFoundError("unknown job id", details={"job_id": job_id})
    return row


def _find_by_key(conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM jobs WHERE idempotency_key = ?",
        (key,),
    ).fetchone()


def _find_active_spec(
    conn: sqlite3.Connection,
    spec_hash: str,
    account_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM jobs WHERE spec_hash = ? AND account_id = ? "
        "AND state NOT IN (?, ?, ?) ORDER BY created_at, job_id LIMIT 1",
        (spec_hash, account_id, *_TERMINAL_STATES),
    ).fetchone()


def _insert_row(conn: sqlite3.Connection, record: JobRecord, *, ignore: bool = False) -> int:
    sql = _INSERT_IGNORE_SQL if ignore else _INSERT_SQL
    try:
        cursor = conn.execute(sql, _record_values(record))
    except sqlite3.IntegrityError as exc:
        raise InvalidInputError(
            "job row conflicts with an existing job",
            details={"job_id": record.job_id},
        ) from exc
    changed = conn.execute("SELECT changes()").fetchone()
    if changed is None:
        return int(cursor.rowcount if cursor.rowcount > 0 else 0)
    return int(changed[0])


def _apply_update(
    conn: sqlite3.Connection,
    job_id: str,
    clauses: list[str],
    params: list[object],
) -> None:
    # Column names are the fixed ``_MUTABLE_FIELDS`` whitelist. Values are bound.
    statement = "UPDATE jobs SET " + ", ".join(clauses) + " WHERE job_id = ?"  # noqa: S608
    conn.execute(statement, (*params, job_id))


def _migrate_to_1(conn: sqlite3.Connection) -> None:
    """Create the v1 tables. Safe to run again if ``user_version`` was not bumped."""
    conn.execute(
        """
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
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency_key
        ON jobs(idempotency_key)
        WHERE idempotency_key IS NOT NULL
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_spec_hash ON jobs(spec_hash)"
    )
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


def _migrate_to_2(conn: sqlite3.Connection) -> None:
    """Add ``account_id`` and backfill legacy rows as ``<provider>-default``.

    Safe to run again: the column is added only when it is missing, and the
    index uses ``IF NOT EXISTS``.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "account_id" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN account_id TEXT")
    conn.execute(
        "UPDATE jobs SET account_id = provider || '-default' WHERE account_id IS NULL"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_account_id ON jobs(account_id)")
    conn.execute("PRAGMA user_version = 2")


_MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (
    _migrate_to_1,
    _migrate_to_2,
)

if len(_MIGRATIONS) != SCHEMA_VERSION:
    raise RuntimeError("SCHEMA_VERSION must equal the number of migrations")


def _select_jobs(
    conn: sqlite3.Connection,
    *,
    states: list[JobState] | None,
    provider: str | None,
    account_id: str | None,
    limit: int | None,
) -> list[sqlite3.Row]:
    query = "SELECT * FROM jobs"
    params: list[object] = []
    filters: list[str] = []
    if states is not None:
        marks = ", ".join(["?"] * len(states))
        filters.append("state IN (" + marks + ")")
        params.extend(state.value for state in states)
    if provider is not None:
        filters.append("provider = ?")
        params.append(provider)
    if account_id is not None:
        filters.append("account_id = ?")
        params.append(account_id)
    if filters:
        query = query + " WHERE " + " AND ".join(filters)
    query = query + " ORDER BY created_at, job_id"
    if limit is not None:
        query = query + " LIMIT ?"
        params.append(limit)
    return list(conn.execute(query, params))


def _parse_states(states: Iterable[JobState] | None) -> list[JobState] | None:
    if states is None:
        return None
    parsed = [_coerce_state(item) for item in states]
    return parsed


def _provider_text(provider: str | None) -> str | None:
    if provider is None:
        return None
    if isinstance(provider, ProviderName):
        return provider.value
    if isinstance(provider, str):
        return provider
    raise InvalidInputError(
        "provider must be a string",
        details={"field": "provider"},
    )


class SqliteJobStore:
    """SQLite-backed job records and state-change events.

    Each operation opens its own connection (``timeout=30``, autocommit) and
    closes it before returning. Write methods take ``BEGIN IMMEDIATE`` and
    commit, or roll back when the body raises.
    """

    def __init__(self, db_path: Path) -> None:
        path = Path(db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(
            self.db_path,
            timeout=_CONNECT_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL").fetchone()
            conn.execute("PRAGMA foreign_keys=ON").fetchone()
            conn.execute("PRAGMA busy_timeout=30000").fetchone()
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                self._rollback(conn)
                raise
            else:
                conn.execute("COMMIT")

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error as exc:
            _LOG.warning("rollback failed: %s", type(exc).__name__)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            current = _read_user_version(conn)
        if current > SCHEMA_VERSION:
            raise ConfigError(
                f"database schema version {current} is newer than {SCHEMA_VERSION}",
                details={"user_version": current, "schema_version": SCHEMA_VERSION},
            )
        if current == SCHEMA_VERSION:
            return
        with self._transaction() as conn:
            current = _read_user_version(conn)
            if current > SCHEMA_VERSION:
                raise ConfigError(
                    f"database schema version {current} is newer than {SCHEMA_VERSION}",
                    details={"user_version": current, "schema_version": SCHEMA_VERSION},
                )
            if current == SCHEMA_VERSION:
                return
            if current < 0 or current > len(_MIGRATIONS):
                raise ConfigError(
                    "database schema version cannot be migrated",
                    details={"user_version": current, "schema_version": SCHEMA_VERSION},
                )
            for migrate in _MIGRATIONS[current:SCHEMA_VERSION]:
                migrate(conn)
            applied = _read_user_version(conn)
            if applied != SCHEMA_VERSION:
                raise ConfigError(
                    "schema migration did not reach the supported version",
                    details={"user_version": applied, "schema_version": SCHEMA_VERSION},
                )

    def insert(self, record: JobRecord) -> JobRecord:
        """Insert ``record`` and return the stored row."""
        _require_job_id(record.job_id)
        with self._transaction() as conn:
            _insert_row(conn, record)
            stored = _fetch_job(conn, record.job_id)
        return _row_to_record(stored)

    def create_or_get(
        self,
        record: JobRecord,
        *,
        dedupe_spec_hash: bool,
    ) -> tuple[JobRecord, bool]:
        """Return an existing row or insert ``record`` in one write transaction.

        An ``idempotency_key`` owned by another account raises
        :class:`ConfigError`. The same key and the same ``spec_hash``,
        including when both are NULL, returns that row. The same key with a
        different hash raises :class:`InvalidInputError` and does not insert.
        Otherwise, when ``dedupe_spec_hash`` is set, a non-terminal row with
        the same ``spec_hash`` is reused. Terminal rows do not block a new
        insert. The boolean is True when this call inserted the row.
        """
        _require_job_id(record.job_id)
        with self._transaction() as conn:
            if record.idempotency_key is not None:
                existing = _find_by_key(conn, record.idempotency_key)
                if existing is not None:
                    if str(existing["account_id"]) != record.account_id:
                        raise ConfigError(
                            "idempotency key is already used by another account",
                            details={
                                "field": "idempotency_key",
                                "reason": "account_conflict",
                                "account_id": record.account_id,
                            },
                        )
                    existing_hash = existing["spec_hash"]
                    if existing_hash != record.spec_hash:
                        raise InvalidInputError(
                            "idempotency key is already used by a job with a different spec",
                            details={
                                "field": "idempotency_key",
                                "reason": "spec_mismatch",
                                "existing_job_id": str(existing["job_id"]),
                                "existing_spec_hash": existing_hash,
                                "requested_spec_hash": record.spec_hash,
                            },
                        )
                    return _row_to_record(existing), False
            if dedupe_spec_hash and record.spec_hash is not None:
                existing = _find_active_spec(conn, record.spec_hash, record.account_id)
                if existing is not None:
                    return _row_to_record(existing), False
            _insert_row(conn, record)
            stored = _fetch_job(conn, record.job_id)
        return _row_to_record(stored), True

    def get(self, job_id: str) -> JobRecord:
        """Return the row for ``job_id``.

        Ids that are not 32 lowercase hex characters raise
        :class:`JobNotFoundError`, as does a missing row.
        """
        job_id = _require_job_id(job_id)
        with self._connect() as conn:
            row = _fetch_job(conn, job_id)
        return _row_to_record(row)

    def find_by_idempotency_key(self, key: str) -> JobRecord | None:
        """Return the row for ``key``, or None when no row uses it."""
        if not isinstance(key, str):
            raise InvalidInputError(
                "idempotency_key must be a string",
                details={"field": "idempotency_key"},
            )
        with self._connect() as conn:
            row = _find_by_key(conn, key)
        if row is None:
            return None
        return _row_to_record(row)

    def list(
        self,
        states: Iterable[JobState] | None = None,
        provider: str | None = None,
        limit: int | None = None,
        *,
        account_id: str | None = None,
    ) -> list[JobRecord]:
        """Return rows ordered by ``created_at`` then ``job_id``.

        ``account_id`` limits the result to that account. ``None`` returns
        every account.
        """
        parsed_states = _parse_states(states)
        if parsed_states is not None and not parsed_states:
            return []
        provider_text = _provider_text(provider)
        bounded = _clean_limit(limit)
        with self._connect() as conn:
            rows = _select_jobs(
                conn,
                states=parsed_states,
                provider=provider_text,
                account_id=account_id,
                limit=bounded,
            )
        return [_row_to_record(row) for row in rows]

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        message: str | None = None,
        expected_version: int | None = None,
        **fields: object,
    ) -> JobRecord:
        """Move ``job_id`` to ``target`` and update the allowed fields.

        The read, state check, version check, update, and optional event insert
        share one ``BEGIN IMMEDIATE`` transaction. An invalid transition or a
        version mismatch rolls back, so the row stays as it was. A same-state
        target updates fields and bumps ``version`` without writing an event.
        """
        job_id = _require_job_id(job_id)
        target_state = _coerce_state(target)
        updates = dict(fields)
        if message is not None:
            updates["message"] = message
        _reject_unknown_fields(updates)
        expected = (
            None if expected_version is None else _clean_expected_version(expected_version)
        )
        clauses, params = _field_assignments(updates)
        now = _format_ts(_utcnow())
        event_message = "" if message is None else _clean_message(message)
        with self._transaction() as conn:
            row = _fetch_job(conn, job_id)
            current = JobState(str(row["state"]))
            validate_transition(current, target_state)
            version = int(row["version"])
            if expected is not None and version != expected:
                raise ConcurrentUpdateError(
                    "job was updated concurrently",
                    details={
                        "job_id": job_id,
                        "expected_version": expected,
                        "version": version,
                    },
                )
            assignments = ["state = ?", "version = ?", "updated_at = ?", *clauses]
            values: list[object] = [target_state.value, version + 1, now, *params]
            _apply_update(conn, job_id, assignments, values)
            if current is not target_state:
                conn.execute(
                    "INSERT INTO job_events (job_id, from_state, to_state, at, message) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (job_id, current.value, target_state.value, now, event_message),
                )
            stored = _fetch_job(conn, job_id)
        return _row_to_record(stored)

    def update_fields(self, job_id: str, **fields: object) -> JobRecord:
        """Update allowed columns without changing ``state`` or writing an event."""
        job_id = _require_job_id(job_id)
        _reject_unknown_fields(fields)
        if not fields:
            raise InvalidInputError("no fields to update")
        clauses, params = _field_assignments(fields)
        now = _format_ts(_utcnow())
        with self._transaction() as conn:
            row = _fetch_job(conn, job_id)
            version = int(row["version"]) + 1
            assignments = ["version = ?", "updated_at = ?", *clauses]
            _apply_update(conn, job_id, assignments, [version, now, *params])
            stored = _fetch_job(conn, job_id)
        return _row_to_record(stored)

    def events(self, job_id: str) -> list[dict[str, Any]]:
        """Return state-change events for ``job_id``, oldest first."""
        job_id = _require_job_id(job_id)
        with self._connect() as conn:
            _fetch_job(conn, job_id)
            rows = conn.execute(
                "SELECT id, job_id, from_state, to_state, at, message "
                "FROM job_events WHERE job_id = ? ORDER BY id",
                (job_id,),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "job_id": row["job_id"],
                "from_state": row["from_state"],
                "to_state": row["to_state"],
                "at": row["at"],
                "message": row["message"],
            }
            for row in rows
        ]

    def migrate_json_index(self, state_dir: Path) -> int:
        """Insert missing V0.2 index files as ``UNKNOWN`` rows.

        Files whose names are not ``<32 lowercase hex>.json`` are ignored.
        Malformed JSON, a provider outside ``kaggle|colab|fake``, a job id that
        does not match the file name, and symlinks that leave ``state_dir/jobs``
        are logged and skipped. Existing SQLite rows are not overwritten. JSON
        files are not modified or deleted. Returns the number of rows inserted.
        """
        pending = _read_json_index(state_dir)
        if not pending:
            return 0
        imported = 0
        with self._transaction() as conn:
            for job_id, provider in pending:
                imported += _import_index_row(conn, job_id, provider)
        return imported


def _import_index_row(conn: sqlite3.Connection, job_id: str, provider: str) -> int:
    moment = _utcnow()
    record = JobRecord(
        job_id=job_id,
        provider=ProviderName(provider),
        account_id=f"{provider}-default",
        provider_job_id=job_id,
        state=JobState.UNKNOWN,
        message=_IMPORT_MESSAGE,
        created_at=moment,
        updated_at=moment,
        origin="json_migration",
    )
    try:
        conn.execute(_INSERT_IGNORE_SQL, _record_values(record))
    except sqlite3.IntegrityError:
        _LOG.warning("skipping job index %s: row conflict", job_id)
        return 0
    changed = conn.execute("SELECT changes()").fetchone()
    if changed is None:
        return 0
    return int(changed[0])


def _read_json_index(state_dir: Path) -> list[tuple[str, str]]:
    try:
        jobs_dir = ensure_within(state_dir, Path(state_dir) / "jobs")
    except PathSecurityError:
        _LOG.warning("skipping jobs directory that escapes the state directory")
        return []
    if not jobs_dir.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for entry in sorted(jobs_dir.iterdir(), key=lambda path: path.name):
        if _INDEX_NAME.fullmatch(entry.name) is None:
            continue
        try:
            resolved = ensure_within(jobs_dir, entry)
        except PathSecurityError:
            _LOG.warning("skipping %s: path escapes the jobs directory", entry.name)
            continue
        except OSError:
            _LOG.warning("skipping %s: unreadable path", entry.name)
            continue
        if not resolved.is_file():
            _LOG.warning("skipping %s: not a regular file", entry.name)
            continue
        parsed = _parse_index_file(resolved, entry.name)
        if parsed is not None:
            found.append(parsed)
    return found


def _parse_index_file(path: Path, name: str) -> tuple[str, str] | None:
    stem = name[: -len(".json")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = _V02Index.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        _LOG.warning("skipping invalid job index %s", name)
        return None
    if record.job_id != stem:
        _LOG.warning("skipping %s: job_id does not match the file name", name)
        return None
    return record.job_id, record.provider.value
