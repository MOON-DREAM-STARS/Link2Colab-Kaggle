"""Job rows stay bound to the account that submitted them. Offline only."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import (
    AccountDisabledError,
    AuthInvalidError,
    ConfigError,
    JobNotFoundError,
    NotSupportedError,
)
from link2kaggle_colab.core.models import GpuType, JobSpec, JobState, ProviderName
from link2kaggle_colab.providers.colab.mock import MockColabProvider
from link2kaggle_colab.providers.kaggle.mock import MockKaggleProvider
from link2kaggle_colab.storage.sqlite_store import SqliteJobStore, default_db_path

# Historical v1 schema, copied from ``_migrate_to_1``.
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


def _settings(
    state_dir: Path,
    *,
    kaggle_a_scenario: str = "quota_45h",
    kaggle_b_scenario: str = "quota_insufficient",
    artifact_root: Path | None = None,
) -> Settings:
    root = artifact_root if artifact_root is not None else state_dir.parent
    return Settings.model_validate(
        {
            "mode": "mock",
            "allow_real_jobs": False,
            "state_dir": state_dir,
            "artifact_root": str(root),
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-a-ref",
                    "mock_scenario": kaggle_a_scenario,
                    "default": True,
                },
                {
                    "id": "kaggle-b",
                    "provider": "kaggle",
                    "credential_ref": "kaggle-b-ref",
                    "mock_scenario": kaggle_b_scenario,
                },
                {
                    "id": "colab-a",
                    "provider": "colab",
                    "credential_ref": "colab-a-ref",
                    "mock_scenario": "ai_pro",
                    "default": True,
                },
                {
                    "id": "colab-b",
                    "provider": "colab",
                    "credential_ref": "colab-b-ref",
                    "mock_scenario": "ccu_insufficient",
                },
            ],
            "credentials": {
                "kaggle-a-ref": {"source": "env", "env": "KAGGLE_A_API_TOKEN"},
                "kaggle-b-ref": {"source": "env", "env": "KAGGLE_B_API_TOKEN"},
                "colab-a-ref": {"source": "env", "env": "COLAB_A_API_TOKEN"},
                "colab-b-ref": {"source": "env", "env": "COLAB_B_API_TOKEN"},
            },
        }
    )


def _script(directory: Path, name: str = "job.py") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f"print({name!r})\n", encoding="utf-8")
    return path


def _gpu(directory: Path, provider: ProviderName, *, name: str) -> JobSpec:
    count = 2 if provider is ProviderName.kaggle else 1
    return JobSpec(
        script_path=_script(directory, name),
        provider=provider,
        gpu_type=GpuType.T4,
        gpu_count=count,
        timeout_seconds=3600,
    )


def _instrument(monkeypatch: pytest.MonkeyPatch) -> dict[str | None, dict[str, int]]:
    counts: dict[str | None, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def wrap(cls: type, method: str) -> None:
        original = getattr(cls, method)

        def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            counts[self._account_id][method] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method, wrapped)

    for cls in (MockKaggleProvider, MockColabProvider):
        for method in ("submit", "status", "logs", "download", "cancel", "get_quota"):
            wrap(cls, method)
    return counts


def _count(counts: dict[str | None, dict[str, int]], account_id: str, method: str) -> int:
    return counts[account_id][method]


def test_jobs_stay_bound_to_the_submitting_account_across_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = _instrument(monkeypatch)
    # Exhausted GPU quota cannot pass the scheduler, so this submit uses a
    # schedulable scenario while the account id stays kaggle-b.
    settings = _settings(tmp_path / "state", kaggle_b_scenario="ok")
    client = Client(settings)
    kaggle_job = client.submit(
        _gpu(tmp_path / "kaggle-main", ProviderName.kaggle, name="main.py"),
        account="kaggle-b",
    )
    kaggle_cancel = client.submit(
        _gpu(tmp_path / "kaggle-cancel", ProviderName.kaggle, name="cancel.py"),
        account="kaggle-b",
    )
    colab_job = client.submit(
        _gpu(tmp_path / "colab-main", ProviderName.colab, name="main.py"),
        account="colab-a",
    )
    colab_cancel = client.submit(
        _gpu(tmp_path / "colab-cancel", ProviderName.colab, name="cancel.py"),
        account="colab-a",
    )
    assert kaggle_job.account_id == "kaggle-b"
    assert colab_job.account_id == "colab-a"
    rows = {row["job_id"]: row for row in client.list_jobs()}
    assert rows[kaggle_job.job_id]["account_id"] == "kaggle-b"
    assert rows[colab_job.job_id]["account_id"] == "colab-a"

    restarted = Client(settings)
    restarted.logs(kaggle_job.job_id)
    status = restarted.status(kaggle_job.job_id)
    assert status.account_id == "kaggle-b"
    while status.state is not JobState.SUCCEEDED:
        status = restarted.status(kaggle_job.job_id)
        assert status.account_id == "kaggle-b"
    restarted.download(kaggle_job.job_id, tmp_path / "kaggle-out")
    with pytest.raises(NotSupportedError):
        restarted.cancel(kaggle_cancel.job_id)

    restarted.logs(colab_job.job_id)
    status = restarted.status(colab_job.job_id)
    while status.state is not JobState.SUCCEEDED:
        status = restarted.status(colab_job.job_id)
    restarted.download(colab_job.job_id, tmp_path / "colab-out")
    cancelled = restarted.cancel(colab_cancel.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.account_id == "colab-a"

    assert _count(counts, "kaggle-b", "status") >= 1
    assert _count(counts, "kaggle-b", "logs") >= 1
    assert _count(counts, "kaggle-b", "download") >= 1
    assert _count(counts, "kaggle-b", "cancel") >= 1
    assert _count(counts, "colab-a", "status") >= 1
    assert _count(counts, "colab-a", "logs") >= 1
    assert _count(counts, "colab-a", "download") >= 1
    assert _count(counts, "colab-a", "cancel") >= 1
    for method in ("submit", "status", "logs", "download", "cancel", "get_quota"):
        assert _count(counts, "kaggle-a", method) == 0
        assert _count(counts, "colab-b", method) == 0


def test_disabled_account_is_not_submitted_or_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = _instrument(monkeypatch)
    settings = _settings(tmp_path / "state")
    client = Client(settings)
    handle = client.submit(
        _gpu(tmp_path / "keep", ProviderName.kaggle, name="keep.py"),
        account="kaggle-a",
    )
    stored = SqliteJobStore(default_db_path(settings.state_dir)).get(handle.job_id)
    version = stored.version
    state = stored.state

    disabled = client.disable_account("kaggle-a")
    assert disabled["enabled"] is False
    assert client.get_account("kaggle-a")["enabled"] is False

    with pytest.raises(AccountDisabledError):
        client.submit(
            _gpu(tmp_path / "blocked", ProviderName.kaggle, name="blocked.py"),
            account="kaggle-a",
        )
    assert _count(counts, "kaggle-a", "submit") == 1

    decision = client.recommend_backend(
        gpu_type=GpuType.T4,
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert decision["provider"] == "colab"
    assert any("kaggle-a" in reason for reason in decision["reasons"])
    assert _count(counts, "kaggle-b", "get_quota") == 0
    assert _count(counts, "kaggle-b", "submit") == 0

    reports = client.reconcile_jobs([handle.job_id])
    assert reports[0]["job_id"] == handle.job_id
    assert reports[0]["skipped"] is True
    assert reports[0]["reason"] == "account_disabled"
    fresh = SqliteJobStore(default_db_path(settings.state_dir)).get(handle.job_id)
    assert fresh.version == version
    assert fresh.state is state
    with pytest.raises(AccountDisabledError):
        client.status(handle.job_id)
    assert _count(counts, "kaggle-a", "status") == 0


def test_account_auth_error_does_not_try_another_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = _instrument(monkeypatch)
    settings = _settings(tmp_path / "state", kaggle_a_scenario="token_invalid")
    client = Client(settings)
    decision = client.recommend_backend(
        gpu_type=GpuType.T4,
        gpu_count=1,
        timeout_seconds=3600,
    )
    assert decision["provider"] == "colab"
    assert any("kaggle-a" in reason for reason in decision["reasons"])
    assert _count(counts, "kaggle-a", "get_quota") >= 1
    assert _count(counts, "kaggle-b", "get_quota") == 0
    assert _count(counts, "kaggle-b", "submit") == 0


def test_submit_error_does_not_create_a_row_for_another_account(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _instrument(monkeypatch)
    settings = _settings(tmp_path / "state")
    client = Client(settings)
    client.get_quota(account="kaggle-a")
    provider = client._built["kaggle-a"]

    def boom(_spec: JobSpec):  # type: ignore[no-untyped-def]
        raise AuthInvalidError("token rejected", provider="kaggle")

    provider.submit = boom  # type: ignore[method-assign]
    with pytest.raises(AuthInvalidError):
        client.submit(
            _gpu(tmp_path / "fail", ProviderName.kaggle, name="fail.py"),
            account="kaggle-a",
        )
    rows = client.list_jobs()
    assert rows
    assert {row["account_id"] for row in rows} == {"kaggle-a"}
    assert all(row["account_id"] != "kaggle-b" for row in rows)


def test_legacy_default_rows_follow_the_configured_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = _instrument(monkeypatch)
    settings = _settings(tmp_path / "state")
    db = default_db_path(settings.state_dir)
    db.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    kaggle_id = uuid4().hex
    colab_id = uuid4().hex
    conn = sqlite3.connect(db)
    try:
        conn.execute(_V1_JOBS_SQL)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency_key
            ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id)")
        conn.execute("PRAGMA user_version = 1")
        conn.executemany(
            "INSERT INTO jobs (job_id, provider, provider_job_id, state, message, "
            "created_at, updated_at, version) VALUES (?, ?, ?, 'QUEUED', 'legacy', ?, ?, 3)",
            [
                (kaggle_id, "kaggle", uuid4().hex, now, now),
                (colab_id, "colab", uuid4().hex, now, now),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    store = SqliteJobStore(db)
    assert store.get(kaggle_id).account_id == "kaggle-default"
    assert store.get(colab_id).account_id == "colab-default"
    client = Client(settings, job_store=store)
    with pytest.raises(JobNotFoundError):
        client.status(kaggle_id)
    with pytest.raises(JobNotFoundError):
        client.status(colab_id)
    assert _count(counts, "kaggle-a", "status") == 1
    assert _count(counts, "colab-a", "status") == 1
    assert _count(counts, "kaggle-b", "status") == 0
    assert _count(counts, "colab-b", "status") == 0


def test_idempotency_key_cannot_be_reused_across_accounts(tmp_path: Path) -> None:
    # The second account must clear the quota check before the key conflict.
    settings = _settings(tmp_path / "state", kaggle_b_scenario="ok")
    client = Client(settings)
    spec = _gpu(tmp_path / "once", ProviderName.kaggle, name="once.py")
    spec = spec.model_copy(update={"idempotency_key": "shared-key"})
    first = client.submit(spec, account="kaggle-a")
    other = spec.model_copy(update={"script_path": _script(tmp_path / "other", "other.py")})
    with pytest.raises(ConfigError) as caught:
        client.submit(other, account="kaggle-b")
    assert caught.value.details["reason"] == "account_conflict"
    rows = client.list_jobs()
    assert [row["job_id"] for row in rows] == [first.job_id]
    assert rows[0]["account_id"] == "kaggle-a"


def test_conflicting_account_and_provider_is_config_error(tmp_path: Path) -> None:
    client = Client(_settings(tmp_path / "state"))
    with pytest.raises(ConfigError) as caught:
        client.recommend_backend(
            gpu_type=GpuType.T4,
            gpu_count=1,
            timeout_seconds=3600,
            provider=ProviderName.colab,
            account="kaggle-a",
        )
    assert caught.value.code == "config_error"
    assert caught.value.details["account_id"] == "kaggle-a"
