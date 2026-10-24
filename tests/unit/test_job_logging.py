"""Job logs and stored messages do not keep credential material."""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from link2kaggle_colab.core.config import JobsSettings
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
from link2kaggle_colab.jobs.logging import RedactingFilter, get_logger, log_event
from link2kaggle_colab.jobs.manager import JobManager
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.storage.sqlite_store import SqliteJobStore

_TOKEN = "KGAT_" + ("n" * 24)


class _Provider(Provider):
    name = ProviderName.fake
    is_mock = True

    def __init__(self) -> None:
        self.status_message = "running"

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
        del spec
        return JobHandle(
            job_id="ab" * 16,
            provider=self.name,
            remote_id="remote-1",
            submitted_at=datetime.now(UTC),
        )

    def status(self, job_id: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            provider=self.name,
            state=JobState.RUNNING,
            message=self.status_message,
            is_mock=True,
            details={"context": self.status_message},
        )

    def logs(self, job_id: str) -> str:
        del job_id
        return "logs"

    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        del job_id, dest_dir
        return []


def _database_bytes(path: Path) -> bytes:
    blob = bytearray(path.read_bytes())
    for suffix in ("-wal", "-shm"):
        extra = Path(str(path) + suffix)
        if extra.is_file():
            blob.extend(extra.read_bytes())
    return bytes(blob)


def test_filter_is_attached_once_and_redacts_message_and_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="link2kaggle_colab.jobs")
    logger = get_logger()
    get_logger()
    attached = [item for item in logger.filters if isinstance(item, RedactingFilter)]
    assert len(attached) == 1
    log_event(logger, logging.INFO, "status", message=f"saw {_TOKEN}", job_id="abc")
    logger.info("provider said %s", _TOKEN)
    blob = caplog.text
    for record in caplog.records:
        blob += str(getattr(record, "l2kc", ""))
        blob += str(record.args)
    assert _TOKEN not in blob
    assert "[REDACTED]" in caplog.text


async def test_provider_message_is_redacted_in_logs_and_the_database(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="link2kaggle_colab.jobs")
    provider = _Provider()
    provider.status_message = f"runtime said {_TOKEN}"
    db = tmp_path / "l2kc.db"
    manager = JobManager(
        SqliteJobStore(db),
        lambda _name, _account_id: provider,
        JobsSettings(),
        sleep=_no_sleep,
        rng=random.Random(1),  # noqa: S311
    )
    script = tmp_path / "train.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    outcome = await manager.submit(ProviderName.fake, JobSpec(script_path=script))
    result = await manager.status(outcome.record.job_id)
    assert _TOKEN not in result.message
    assert "[REDACTED]" in result.message
    assert _TOKEN not in str(result.details)
    stored = manager.get(outcome.record.job_id)
    assert _TOKEN not in stored.message
    assert _TOKEN not in str(stored.details)
    blob = caplog.text + _database_bytes(db).decode("utf-8", errors="ignore")
    assert _TOKEN not in blob


async def _no_sleep(_delay: float) -> None:
    return None
