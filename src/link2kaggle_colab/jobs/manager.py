"""Async job manager over the synchronous provider interface.

Provider methods run on dedicated non-daemon threads, not the loop's default
executor. A timeout therefore returns while the provider call may still be
finishing. Submit and cancel are attempted once. Status, logs, and download
use :func:`retry_async`. Our ``job_id`` is a new uuid4 hex. The provider
handle id is stored as ``provider_job_id``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, field_validator

from link2kaggle_colab.core.config import JobsSettings
from link2kaggle_colab.core.errors import (
    AuthError,
    ConcurrentUpdateError,
    ConfigError,
    InvalidInputError,
    InvalidTransitionError,
    L2KCError,
    NotSupportedError,
    PathSecurityError,
    PremiumNotAllowed,
    QuotaExceededError,
    RealModeDisabledError,
    ResourceUnavailableError,
    SubmitOutcomeUnknownError,
    TransientProviderError,
)
from link2kaggle_colab.core.models import (
    Artifact,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
)
from link2kaggle_colab.core.paths import ensure_within
from link2kaggle_colab.core.redact import redact, redact_value
from link2kaggle_colab.jobs.idempotency import spec_hash
from link2kaggle_colab.jobs.logging import get_logger, log_event
from link2kaggle_colab.jobs.retry import RetryPolicy, retry_async
from link2kaggle_colab.jobs.state_machine import ACTIVE_STATES, can_transition
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.storage.sqlite_store import JobRecord, SqliteJobStore

_R = TypeVar("_R")

_TIMEOUT_MESSAGE = (
    "submit outcome is ambiguous (timed out); not resubmitted; run reconcile"
)
_NO_ID_MESSAGE = (
    "submit outcome is ambiguous and no provider id was recorded; "
    "check the provider manually; it will not be resubmitted"
)
_AMBIGUOUS_MESSAGE = "submit outcome is ambiguous; not resubmitted; run reconcile"

# Failures that mean the provider did not start the job.
_DEFINITIVE: tuple[type[BaseException], ...] = (
    AuthError,
    InvalidInputError,
    PremiumNotAllowed,
    ResourceUnavailableError,
    QuotaExceededError,
    RealModeDisabledError,
    NotSupportedError,
    PathSecurityError,
    ConfigError,
)


class SubmitOutcome(BaseModel):
    """Result of :meth:`JobManager.submit`. ``created`` is false on dedupe."""

    model_config = ConfigDict(extra="forbid")

    record: JobRecord
    handle: JobHandle
    created: bool


class ReconcileReport(BaseModel):
    """One job visited by :meth:`JobManager.reconcile`. ``error`` is redacted."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    provider: ProviderName | None = None
    before: JobState | None = None
    after: JobState | None = None
    ok: bool
    error: dict[str, Any] | None = None
    note: str | None = None
    skipped: bool = False
    reason: str | None = None

    @field_validator("error")
    @classmethod
    def _redact_error(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        redacted = redact_value(value)
        if not isinstance(redacted, dict):
            raise ValueError("error must be a mapping")
        return redacted

    @field_validator("note")
    @classmethod
    def _redact_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return redact(value)


@dataclass(frozen=True, slots=True)
class _SubmitCall:
    """What the submit worker observed. Exceptions stay inside the thread."""

    handle: JobHandle | None = None
    error: Exception | None = None


def _is_definitive(exc: BaseException) -> bool:
    return isinstance(exc, _DEFINITIVE)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, L2KCError):
        body: dict[str, Any] = exc.to_dict()
    else:
        body = {
            "code": "l2kc_error",
            "message": redact(str(exc)),
            "exception_type": type(exc).__name__,
        }
    redacted = redact_value(body)
    if isinstance(redacted, dict):
        return redacted
    return {"message": "[REDACTED]"}


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(exc).__name__


def _error_message(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    text = redact(str(exc))
    return text or type(exc).__name__


def _stamp(exc: L2KCError, job_id: str, state: JobState) -> None:
    exc.details["job_id"] = job_id
    exc.details["job_state"] = state.value


def _parse_started(details: dict[str, Any]) -> datetime | None:
    raw = details.get("started_at")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


class JobManager:
    """Own job ids, idempotency, timeouts, and crash recovery for one store."""

    def __init__(
        self,
        store: SqliteJobStore,
        provider_for: Callable[[ProviderName, str], Provider],
        settings: JobsSettings,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self._provider_for = provider_for
        self._settings = settings
        self._sleep = sleep
        # Jitter only. Not used for secrets or identifiers.
        self._rng = rng if rng is not None else random.Random()  # noqa: S311
        self._clock = clock
        self._logger = get_logger()
        self._policy = RetryPolicy(
            max_attempts=settings.retry_max_attempts,
            base_delay=settings.retry_base_delay_seconds,
            max_delay=settings.retry_max_delay_seconds,
        )

    async def submit(
        self,
        provider: ProviderName,
        spec: JobSpec,
        account_id: str | None = None,
    ) -> SubmitOutcome:
        """Insert one job and call ``provider.submit`` at most once.

        The provider adapter is resolved before a row is inserted. A lookup
        failure, such as a disabled provider, creates no row and cannot leave
        ``SUBMITTING``. A new row is inserted already in ``SUBMITTING``, with
        ``submit_started_at`` and ``submit_pid`` in ``details``, so a crash
        cannot leave a ``PENDING`` row outside reconcile. A client
        ``idempotency_key`` returns the existing row in any state when the
        spec hash matches, including a legacy ``PENDING`` row, and does not
        move it. A different hash raises ``InvalidInputError`` before the
        provider is called. Without a key, the spec hash dedupes only against
        non-terminal rows of the same account. A duplicate does not call the
        provider. ``account_id`` is stored on the row; later calls use that
        id. When omitted, the row is bound to ``<provider>-default``.
        """
        chosen = ProviderName(provider)
        bound = account_id if account_id else f"{chosen.value}-default"
        adapter = self._provider_for(chosen, bound)
        key = spec.idempotency_key
        digest = spec_hash(spec)
        now = self._now()
        proposal = JobRecord(
            job_id=uuid4().hex,
            provider=chosen,
            account_id=bound,
            state=JobState.SUBMITTING,
            idempotency_key=key,
            spec_hash=digest,
            details={
                "submit_started_at": now.isoformat(),
                "submit_pid": os.getpid(),
            },
            created_at=now,
            updated_at=now,
        )
        record, created = self._store.create_or_get(
            proposal,
            dedupe_spec_hash=key is None,
        )
        if not created:
            return self._outcome(record, created=False)

        log_event(
            self._logger,
            logging.INFO,
            "submit_start",
            job_id=record.job_id,
            provider=chosen.value,
            to_state=JobState.SUBMITTING.value,
        )
        try:
            call = await self._run_in_thread(
                self._call_provider_submit,
                adapter,
                spec,
                record.job_id,
                timeout=self._submit_timeout(spec),
                operation="submit",
            )
        except TimeoutError:
            return self._finish_submit_timeout(record.job_id, chosen)
        return self._finish_submit_call(record.job_id, chosen, call)

    async def refresh(self, job_id: str) -> JobRecord:
        """Poll provider status unless the row is terminal or has no provider id.

        Transport failures leave the stored state unchanged. An illegal
        reported state is stored as ``details["desync"]`` and is not applied.
        """
        record = self._store.get(job_id)
        return await self._refresh_record(record)

    async def status(self, job_id: str) -> JobResult:
        """Refresh ``job_id`` and return a result that uses our job id."""
        record = await self.refresh(job_id)
        return self._result_from(record)

    async def logs(self, job_id: str) -> str:
        """Return redacted logs. Requires a stored provider id."""
        record = self._store.get(job_id)
        provider_job_id = self._require_provider_id(record, action="fetch logs")
        adapter = self._provider_for(record.provider, record.account_id)

        async def _once() -> str:
            return await self._run_in_thread(
                adapter.logs,
                provider_job_id,
                timeout=self._settings.logs_timeout_seconds,
                operation="logs",
            )

        text = await retry_async(
            "logs",
            _once,
            policy=self._policy,
            sleep=self._sleep,
            rng=self._rng,
            logger=self._logger,
        )
        return redact(text)

    async def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        """Download artifacts for a ``SUCCEEDED`` job into ``dest_dir``.

        A failed download or a path outside ``dest_dir`` raises and does not
        change the job state or recorded artifact names.
        """
        record = await self.refresh(job_id)
        if record.state is not JobState.SUCCEEDED:
            raise InvalidInputError(
                "download requires a SUCCEEDED job",
                details={"job_id": record.job_id, "job_state": record.state.value},
            )
        provider_job_id = self._require_provider_id(record, action="download")
        dest = Path(dest_dir).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        adapter = self._provider_for(record.provider, record.account_id)

        async def _once() -> list[Artifact]:
            return await self._run_in_thread(
                adapter.download,
                provider_job_id,
                dest,
                timeout=self._settings.download_timeout_seconds,
                operation="download",
            )

        artifacts = await retry_async(
            "download",
            _once,
            policy=self._policy,
            sleep=self._sleep,
            rng=self._rng,
            logger=self._logger,
        )
        for artifact in artifacts:
            ensure_within(dest, artifact.local_path)
        self._store.update_fields(record.job_id, artifacts=[item.name for item in artifacts])
        return artifacts

    async def cancel(self, job_id: str) -> JobResult:
        """Cancel one job. The provider is called at most once and not retried.

        Terminal rows return the stored result. ``PENDING`` is cancelled
        locally. An ambiguous submit with no provider id cannot be cancelled.
        """
        record = self._store.get(job_id)
        if record.state.is_terminal():
            log_event(
                self._logger,
                logging.INFO,
                "cancel",
                job_id=record.job_id,
                provider=record.provider.value,
                from_state=record.state.value,
                to_state=record.state.value,
            )
            return self._result_from(record)
        if record.state is JobState.PENDING:
            updated = self._transition_record(
                record,
                JobState.CANCELLED,
                message="cancelled before the provider was called",
            )
            log_event(
                self._logger,
                logging.INFO,
                "cancel",
                job_id=updated.job_id,
                provider=updated.provider.value,
                from_state=JobState.PENDING.value,
                to_state=JobState.CANCELLED.value,
            )
            return self._result_from(updated)
        if record.provider_job_id is None:
            raise InvalidInputError(
                "cannot cancel an ambiguous submit; check the provider manually; "
                "it will not be resubmitted",
                details={"job_id": record.job_id, "job_state": record.state.value},
            )
        previous = record.state
        cancelling = self._transition_record(record, JobState.CANCELLING)
        log_event(
            self._logger,
            logging.INFO,
            "cancel",
            job_id=cancelling.job_id,
            provider=cancelling.provider.value,
            from_state=previous.value,
            to_state=JobState.CANCELLING.value,
        )
        adapter = self._provider_for(record.provider, record.account_id)
        provider_job_id = record.provider_job_id
        try:
            result = await self._run_in_thread(
                adapter.cancel,
                provider_job_id,
                timeout=self._settings.cancel_timeout_seconds,
                operation="cancel",
            )
        except NotSupportedError as exc:
            restored = self._revert_cancel(cancelling.job_id, previous)
            _stamp(exc, cancelling.job_id, restored.state)
            log_event(
                self._logger,
                logging.WARNING,
                "cancel",
                job_id=cancelling.job_id,
                provider=cancelling.provider.value,
                from_state=previous.value,
                to_state=restored.state.value,
                error_code=exc.code,
            )
            raise
        except Exception as exc:
            log_event(
                self._logger,
                logging.WARNING,
                "cancel",
                job_id=cancelling.job_id,
                provider=cancelling.provider.value,
                to_state=JobState.CANCELLING.value,
                error_code=_error_code(exc),
                message=_error_message(exc),
            )
            self._raise_cancel_error(exc, cancelling.job_id)
        observed = result.model_copy(update={"job_id": cancelling.job_id})
        applied = await self._apply_status(cancelling.job_id, observed, reapply=True)
        return self._result_from(applied)

    async def reconcile(self, job_ids: Iterable[str] | None = None) -> list[ReconcileReport]:
        """Poll active jobs. Never calls ``provider.submit``.

        One job's error is reported and does not stop the others.
        """
        if job_ids is None:
            chosen = [row.job_id for row in self._store.list(states=ACTIVE_STATES)]
        else:
            chosen = list(job_ids)
        log_event(self._logger, logging.INFO, "reconcile", count=len(chosen))
        semaphore = asyncio.Semaphore(self._settings.reconcile_concurrency)

        async def _one(job_id: str) -> ReconcileReport:
            async with semaphore:
                return await self._reconcile_one(job_id)

        reports = await asyncio.gather(*(_one(job_id) for job_id in chosen))
        return list(reports)

    def list_jobs(
        self,
        states: Iterable[JobState] | None = None,
        provider: str | None = None,
        limit: int | None = None,
        *,
        account_id: str | None = None,
    ) -> list[JobRecord]:
        """Return stored rows. This does not call a provider.

        ``account_id`` limits the result to that account when set.
        """
        return self._store.list(
            states=states,
            provider=provider,
            limit=limit,
            account_id=account_id,
        )

    def get(self, job_id: str) -> JobRecord:
        """Return one stored row."""
        return self._store.get(job_id)

    async def _run_in_thread(
        self,
        func: Callable[..., _R],
        *args: Any,
        timeout: float,
        operation: str,
    ) -> _R:
        """Run ``func(*args)`` on a dedicated thread named ``l2kc-<operation>``.

        The thread is non-daemon so process exit still waits for an in-flight
        submit to store the provider id and for the Colab adapter to stop its
        runtime. The result is delivered with ``call_soon_threadsafe``. This
        does not use the loop default executor, so :func:`asyncio.run` can
        return when ``timeout`` elapses. A provider exception is delivered as
        that exception. A wait that expires first becomes
        :class:`TransientProviderError` for every operation except submit,
        which still raises :class:`TimeoutError` so the caller can mark the
        job ``UNKNOWN`` without resubmitting.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[_R] = loop.create_future()

        def _worker() -> None:
            error: BaseException | None = None
            result: _R
            try:
                result = func(*args)
            except Exception as exc:
                error = exc

            def _deliver() -> None:
                # The loop may already be closed, or the wait may already have
                # finished. Drop the outcome. Never raise on this thread.
                if fut.done():
                    return
                try:
                    if error is not None:
                        fut.set_exception(error)
                        return
                    fut.set_result(result)
                except (asyncio.InvalidStateError, RuntimeError):
                    return

            try:
                loop.call_soon_threadsafe(_deliver)
            except RuntimeError:
                return

        thread = threading.Thread(
            target=_worker,
            name=f"l2kc-{operation}",
            daemon=False,
        )
        thread.start()
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout)
        except TimeoutError as exc:
            if fut.done() and not fut.cancelled():
                return fut.result()
            if operation == "submit":
                raise
            raise TransientProviderError(
                f"{operation} timed out after {timeout:g}s",
                details={
                    "timed_out": True,
                    "operation": operation,
                    "timeout_seconds": timeout,
                },
            ) from exc

    def _submit_timeout(self, spec: JobSpec) -> float:
        job_limit = float(spec.timeout_seconds or 0)
        return self._settings.submit_timeout_seconds + job_limit

    def _now(self) -> datetime:
        current = self._clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise InvalidInputError(
                "clock must return a timezone-aware datetime",
                details={"reason": "naive datetime"},
            )
        return current.astimezone(UTC)

    def _outcome(self, record: JobRecord, *, created: bool) -> SubmitOutcome:
        submitted = record.submitted_at or record.created_at
        handle = JobHandle(
            job_id=record.job_id,
            provider=record.provider,
            account_id=record.account_id,
            remote_id=record.remote_id,
            submitted_at=submitted,
        )
        return SubmitOutcome(record=record, handle=handle, created=created)

    def _result_from(self, record: JobRecord) -> JobResult:
        adapter = self._provider_for(record.provider, record.account_id)
        return JobResult(
            job_id=record.job_id,
            provider=record.provider,
            account_id=record.account_id,
            state=record.state,
            remote_id=record.remote_id,
            message=record.message,
            started_at=_parse_started(record.details),
            finished_at=record.finished_at,
            exit_code=record.exit_code,
            artifacts=list(record.artifacts),
            is_mock=bool(adapter.is_mock),
            details=dict(record.details),
        )

    def _transition_record(
        self,
        record: JobRecord,
        target: JobState,
        *,
        message: str | None = None,
        **fields: object,
    ) -> JobRecord:
        updated = self._store.transition(
            record.job_id,
            target,
            message=message,
            expected_version=record.version,
            **fields,
        )
        if record.state is not updated.state:
            log_event(
                self._logger,
                logging.INFO,
                "state_change",
                job_id=record.job_id,
                provider=record.provider.value,
                from_state=record.state.value,
                to_state=updated.state.value,
            )
        return updated

    def _call_provider_submit(
        self,
        provider: Provider,
        spec: JobSpec,
        job_id: str,
    ) -> _SubmitCall:
        try:
            handle = provider.submit(spec)
        except Exception as exc:
            return _SubmitCall(error=exc)
        try:
            self._persist_handle(job_id, handle)
        except Exception as exc:
            return _SubmitCall(handle=handle, error=exc)
        return _SubmitCall(handle=handle)

    def _persist_handle(self, job_id: str, handle: JobHandle) -> None:
        """Record a submit that returned. A late return after timeout stays UNKNOWN."""
        fields: dict[str, object] = {
            "provider_job_id": handle.job_id,
            "submitted_at": handle.submitted_at,
        }
        if handle.remote_id is not None:
            fields["remote_id"] = handle.remote_id
        for _ in range(2):
            record = self._store.get(job_id)
            if record.provider_job_id:
                return
            if record.state is JobState.SUBMITTING:
                try:
                    self._store.transition(
                        job_id,
                        JobState.QUEUED,
                        expected_version=record.version,
                        **fields,
                    )
                except ConcurrentUpdateError:
                    continue
                except InvalidTransitionError:
                    continue
                log_event(
                    self._logger,
                    logging.INFO,
                    "state_change",
                    job_id=job_id,
                    provider=record.provider.value,
                    from_state=JobState.SUBMITTING.value,
                    to_state=JobState.QUEUED.value,
                )
                return
            if record.state is JobState.UNKNOWN:
                self._store.update_fields(job_id, **fields)
                log_event(
                    self._logger,
                    logging.WARNING,
                    "submit_unknown",
                    job_id=job_id,
                    provider=record.provider.value,
                    to_state=JobState.UNKNOWN.value,
                    message="provider id recorded after the submit wait ended",
                )
                return
            return
        record = self._store.get(job_id)
        if record.state is JobState.UNKNOWN and not record.provider_job_id:
            self._store.update_fields(job_id, **fields)

    def _finish_submit_timeout(self, job_id: str, provider: ProviderName) -> SubmitOutcome:
        record = self._claim_unknown_submit(job_id)
        if record.state is JobState.QUEUED and record.provider_job_id:
            log_event(
                self._logger,
                logging.INFO,
                "submit_ok",
                job_id=job_id,
                provider=provider.value,
                to_state=record.state.value,
            )
            return self._outcome(record, created=True)
        log_event(
            self._logger,
            logging.WARNING,
            "submit_unknown",
            job_id=job_id,
            provider=provider.value,
            to_state=JobState.UNKNOWN.value,
            message=_TIMEOUT_MESSAGE,
        )
        raise SubmitOutcomeUnknownError(
            _TIMEOUT_MESSAGE,
            provider=provider.value,
            details={"job_id": job_id, "job_state": JobState.UNKNOWN.value},
        )

    def _claim_unknown_submit(self, job_id: str) -> JobRecord:
        for _ in range(2):
            record = self._store.get(job_id)
            if record.state is not JobState.SUBMITTING:
                return record
            details = dict(record.details)
            details["note"] = _TIMEOUT_MESSAGE
            try:
                updated = self._store.transition(
                    job_id,
                    JobState.UNKNOWN,
                    expected_version=record.version,
                    message=_TIMEOUT_MESSAGE,
                    details=details,
                )
            except ConcurrentUpdateError:
                continue
            except InvalidTransitionError:
                continue
            log_event(
                self._logger,
                logging.INFO,
                "state_change",
                job_id=job_id,
                provider=record.provider.value,
                from_state=JobState.SUBMITTING.value,
                to_state=JobState.UNKNOWN.value,
            )
            return updated
        return self._store.get(job_id)

    def _finish_submit_call(
        self,
        job_id: str,
        provider: ProviderName,
        call: _SubmitCall,
    ) -> SubmitOutcome:
        if call.handle is not None and call.error is not None:
            try:
                self._persist_handle(job_id, call.handle)
            except Exception as exc:
                call = _SubmitCall(handle=call.handle, error=exc)
        record = self._store.get(job_id)
        if record.state is JobState.QUEUED and record.provider_job_id and call.error is None:
            log_event(
                self._logger,
                logging.INFO,
                "submit_ok",
                job_id=job_id,
                provider=provider.value,
                to_state=JobState.QUEUED.value,
            )
            return self._outcome(record, created=True)
        if call.error is not None:
            if record.state is JobState.QUEUED and record.provider_job_id:
                log_event(
                    self._logger,
                    logging.INFO,
                    "submit_ok",
                    job_id=job_id,
                    provider=provider.value,
                    to_state=JobState.QUEUED.value,
                )
                return self._outcome(record, created=True)
            settled = self._settle_submit_error(job_id, provider, call.error)
            if settled is not call.error:
                raise settled from call.error
            raise settled
        if record.provider_job_id:
            event = "submit_ok" if record.state is JobState.QUEUED else "submit_unknown"
            level = logging.INFO if event == "submit_ok" else logging.WARNING
            log_event(
                self._logger,
                level,
                event,
                job_id=job_id,
                provider=provider.value,
                to_state=record.state.value,
            )
            return self._outcome(record, created=True)
        raise SubmitOutcomeUnknownError(
            _AMBIGUOUS_MESSAGE,
            provider=provider.value,
            details={"job_id": job_id, "job_state": record.state.value},
        )

    def _settle_submit_error(
        self,
        job_id: str,
        provider: ProviderName,
        exc: Exception,
    ) -> Exception:
        record = self._store.get(job_id)
        if record.state is not JobState.SUBMITTING:
            if isinstance(exc, L2KCError):
                _stamp(exc, job_id, record.state)
                return exc
            return self._unknown_wrapper(exc, job_id, provider, record.state)
        if _is_definitive(exc) and isinstance(exc, L2KCError):
            updated = self._mark_submit_failed(record, exc)
            _stamp(exc, job_id, updated.state)
            log_event(
                self._logger,
                logging.WARNING,
                "submit_failed" if updated.state is JobState.FAILED else "submit_unknown",
                job_id=job_id,
                provider=provider.value,
                to_state=updated.state.value,
                error_code=exc.code,
                message=exc.message,
            )
            return exc
        updated = self._mark_submit_unknown(record, exc)
        log_event(
            self._logger,
            logging.WARNING,
            "submit_unknown",
            job_id=job_id,
            provider=provider.value,
            to_state=updated.state.value,
            error_code=_error_code(exc),
            message=_error_message(exc),
        )
        if isinstance(exc, L2KCError):
            _stamp(exc, job_id, updated.state)
            return exc
        return self._unknown_wrapper(exc, job_id, provider, updated.state)

    def _mark_submit_failed(self, record: JobRecord, exc: L2KCError) -> JobRecord:
        details = dict(record.details)
        details["error_code"] = exc.code
        try:
            return self._transition_record(
                record,
                JobState.FAILED,
                message=exc.message,
                details=details,
            )
        except (ConcurrentUpdateError, InvalidTransitionError):
            return self._store.get(record.job_id)

    def _mark_submit_unknown(self, record: JobRecord, exc: Exception) -> JobRecord:
        details = dict(record.details)
        details["note"] = _AMBIGUOUS_MESSAGE
        if isinstance(exc, L2KCError):
            details["error_code"] = exc.code
            message = exc.message or _AMBIGUOUS_MESSAGE
        else:
            details["exception_type"] = type(exc).__name__
            message = _error_message(exc) or _AMBIGUOUS_MESSAGE
        try:
            return self._transition_record(
                record,
                JobState.UNKNOWN,
                message=message,
                details=details,
            )
        except (ConcurrentUpdateError, InvalidTransitionError):
            return self._store.get(record.job_id)

    def _unknown_wrapper(
        self,
        exc: Exception,
        job_id: str,
        provider: ProviderName,
        state: JobState,
    ) -> SubmitOutcomeUnknownError:
        return SubmitOutcomeUnknownError(
            _error_message(exc),
            provider=provider.value,
            details={
                "exception_type": type(exc).__name__,
                "job_id": job_id,
                "job_state": state.value,
            },
        )

    async def _refresh_record(self, record: JobRecord) -> JobRecord:
        if record.state.is_terminal():
            return record
        if record.provider_job_id is None and record.state in {
            JobState.UNKNOWN,
            JobState.SUBMITTING,
        }:
            if record.state is JobState.SUBMITTING and not self._is_stale_submitting(record):
                return record
            return self._mark_ambiguous(record)
        if record.provider_job_id is None:
            raise InvalidInputError(
                "job has no provider id, so status cannot be polled",
                details={"job_id": record.job_id, "job_state": record.state.value},
            )
        observed = await self._poll_status(record)
        return await self._apply_status(record.job_id, observed, reapply=True)

    def _is_stale_submitting(self, record: JobRecord) -> bool:
        moment = record.updated_at
        raw = record.details.get("submit_started_at")
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                parsed = None
            if parsed is not None and parsed.tzinfo is not None and parsed.utcoffset() is not None:
                moment = parsed.astimezone(UTC)
        return (self._now() - moment).total_seconds() > self._settings.stale_submitting_seconds

    def _mark_ambiguous(self, record: JobRecord) -> JobRecord:
        if (
            record.state is JobState.UNKNOWN
            and record.message == _NO_ID_MESSAGE
            and record.details.get("note") == _NO_ID_MESSAGE
        ):
            return record
        details = dict(record.details)
        details["note"] = _NO_ID_MESSAGE
        target = JobState.UNKNOWN
        try:
            updated = self._transition_record(
                record,
                target,
                message=_NO_ID_MESSAGE,
                details=details,
            )
        except (ConcurrentUpdateError, InvalidTransitionError):
            updated = self._store.get(record.job_id)
        log_event(
            self._logger,
            logging.WARNING,
            "submit_unknown",
            job_id=record.job_id,
            provider=record.provider.value,
            from_state=record.state.value,
            to_state=updated.state.value,
            message=_NO_ID_MESSAGE,
        )
        return updated

    async def _poll_status(self, record: JobRecord) -> JobResult:
        provider_job_id = record.provider_job_id
        if provider_job_id is None:
            raise InvalidInputError(
                "job has no provider id, so status cannot be polled",
                details={"job_id": record.job_id, "job_state": record.state.value},
            )
        adapter = self._provider_for(record.provider, record.account_id)

        async def _once() -> JobResult:
            return await self._run_in_thread(
                adapter.status,
                provider_job_id,
                timeout=self._settings.status_timeout_seconds,
                operation="status",
            )

        result = await retry_async(
            "status",
            _once,
            policy=self._policy,
            sleep=self._sleep,
            rng=self._rng,
            logger=self._logger,
        )
        return result.model_copy(update={"job_id": record.job_id})

    async def _apply_status(
        self,
        job_id: str,
        result: JobResult,
        *,
        reapply: bool,
    ) -> JobRecord:
        current = self._store.get(job_id)
        reported = result.state
        if not can_transition(current.state, reported):
            try:
                return self._write_desync(current, reported)
            except ConcurrentUpdateError:
                if not reapply:
                    raise
                return await self._apply_status(job_id, result, reapply=False)
        fields: dict[str, object] = {
            "exit_code": result.exit_code,
            "finished_at": result.finished_at,
            "artifacts": list(result.artifacts),
            "details": self._merged_details(current, result),
        }
        if result.remote_id is not None:
            fields["remote_id"] = result.remote_id
        try:
            updated = self._transition_record(
                current,
                reported,
                message=result.message,
                **fields,
            )
        except ConcurrentUpdateError:
            if not reapply:
                raise
            return await self._apply_status(job_id, result, reapply=False)
        except InvalidTransitionError:
            if not reapply:
                fresh = self._store.get(job_id)
                if not can_transition(fresh.state, reported):
                    return self._write_desync(fresh, reported)
                raise
            return await self._apply_status(job_id, result, reapply=False)
        return await self._maybe_release(updated, result)

    def _merged_details(self, current: JobRecord, result: JobResult) -> dict[str, Any]:
        merged = dict(current.details)
        merged.update(result.details)
        if result.started_at is not None and result.started_at.tzinfo is not None:
            merged["started_at"] = result.started_at.astimezone(UTC).isoformat()
        return merged

    def _write_desync(self, current: JobRecord, reported: JobState) -> JobRecord:
        details = dict(current.details)
        details["desync"] = {
            "reported": reported.value,
            "at": self._now().isoformat(),
        }
        log_event(
            self._logger,
            logging.WARNING,
            "state_desync",
            job_id=current.job_id,
            provider=current.provider.value,
            from_state=current.state.value,
            reported=reported.value,
        )
        return self._transition_record(current, current.state, details=details)

    async def _maybe_release(self, record: JobRecord, result: JobResult) -> JobRecord:
        if not record.state.is_terminal():
            return record
        if not result.details.get("runtime_stop_failed"):
            return record
        if record.details.get("runtime_release_attempted"):
            return record
        if not record.provider_job_id:
            return record
        details = dict(record.details)
        details["runtime_release_attempted"] = True
        details["runtime_release_outcome"] = "started"
        try:
            claimed = self._transition_record(record, record.state, details=details)
        except ConcurrentUpdateError:
            fresh = self._store.get(record.job_id)
            if fresh.details.get("runtime_release_attempted") or not fresh.state.is_terminal():
                return fresh
            return await self._maybe_release(fresh, result)
        outcome = "released"
        error_code: str | None = None
        provider_job_id = claimed.provider_job_id
        if provider_job_id is None:
            return claimed
        try:
            await self._run_in_thread(
                self._provider_for(claimed.provider, claimed.account_id).cancel,
                provider_job_id,
                timeout=self._settings.cancel_timeout_seconds,
                operation="cancel",
            )
        except Exception as exc:
            outcome = "failed"
            error_code = _error_code(exc)
            log_event(
                self._logger,
                logging.WARNING,
                "runtime_release",
                job_id=claimed.job_id,
                provider=claimed.provider.value,
                to_state=claimed.state.value,
                ok=False,
                error_code=error_code,
                message=_error_message(exc),
            )
        else:
            log_event(
                self._logger,
                logging.INFO,
                "runtime_release",
                job_id=claimed.job_id,
                provider=claimed.provider.value,
                to_state=claimed.state.value,
                ok=True,
            )
        final = dict(claimed.details)
        final["runtime_release_attempted"] = True
        final["runtime_release_outcome"] = outcome
        if error_code is not None:
            final["runtime_release_error_code"] = error_code
        try:
            return self._transition_record(claimed, claimed.state, details=final)
        except ConcurrentUpdateError:
            fresh = self._store.get(claimed.job_id)
            if fresh.state is not claimed.state:
                return fresh
            return self._store.update_fields(fresh.job_id, details=final)

    def _require_provider_id(self, record: JobRecord, *, action: str) -> str:
        if record.provider_job_id:
            return record.provider_job_id
        raise InvalidInputError(
            f"submit outcome is ambiguous and no provider id was recorded; cannot {action}",
            details={"job_id": record.job_id, "job_state": record.state.value},
        )

    def _revert_cancel(self, job_id: str, previous: JobState) -> JobRecord:
        current = self._store.get(job_id)
        if current.state is previous:
            return current
        if current.state is not JobState.CANCELLING or not can_transition(current.state, previous):
            return current
        try:
            return self._transition_record(current, previous)
        except (ConcurrentUpdateError, InvalidTransitionError):
            fresh = self._store.get(job_id)
            if fresh.state is JobState.CANCELLING and can_transition(fresh.state, previous):
                try:
                    return self._transition_record(fresh, previous)
                except (ConcurrentUpdateError, InvalidTransitionError):
                    return self._store.get(job_id)
            return fresh

    def _raise_cancel_error(self, exc: Exception, job_id: str) -> None:
        if isinstance(exc, L2KCError):
            _stamp(exc, job_id, JobState.CANCELLING)
            raise exc
        raise L2KCError(
            _error_message(exc),
            details={
                "job_id": job_id,
                "job_state": JobState.CANCELLING.value,
                "exception_type": type(exc).__name__,
            },
        ) from exc

    async def _reconcile_one(self, job_id: str) -> ReconcileReport:
        try:
            before = self._store.get(job_id)
        except L2KCError as exc:
            log_event(
                self._logger,
                logging.WARNING,
                "reconcile",
                job_id=job_id,
                ok=False,
                error_code=exc.code,
            )
            return ReconcileReport(job_id=job_id, ok=False, error=exc.to_dict())
        try:
            after = await self.refresh(job_id)
        except Exception as exc:
            current = self._store.get(job_id)
            payload = _error_payload(exc)
            log_event(
                self._logger,
                logging.WARNING,
                "reconcile",
                job_id=job_id,
                provider=before.provider.value,
                from_state=before.state.value,
                to_state=current.state.value,
                ok=False,
                error_code=str(payload.get("code", _error_code(exc))),
            )
            return ReconcileReport(
                job_id=job_id,
                provider=before.provider,
                before=before.state,
                after=current.state,
                ok=False,
                error=payload,
                note=None,
            )
        log_event(
            self._logger,
            logging.INFO,
            "reconcile",
            job_id=job_id,
            provider=before.provider.value,
            from_state=before.state.value,
            to_state=after.state.value,
            ok=True,
        )
        return ReconcileReport(
            job_id=job_id,
            provider=before.provider,
            before=before.state,
            after=after.state,
            ok=True,
            note=_report_note(before, after),
        )


def _report_note(before: JobRecord, after: JobRecord) -> str | None:
    raw = after.details.get("note")
    if isinstance(raw, str) and raw:
        return raw
    desync = after.details.get("desync")
    if isinstance(desync, dict) and desync != before.details.get("desync"):
        reported = desync.get("reported")
        return f"state_desync:{reported}"
    return None
