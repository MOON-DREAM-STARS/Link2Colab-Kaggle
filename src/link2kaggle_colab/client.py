"""SDK client. CLI and MCP are thin wrappers; job decisions live here."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from link2kaggle_colab.accounts.credentials import resolve_credential
from link2kaggle_colab.accounts.models import Account
from link2kaggle_colab.accounts.registry import AccountRegistry
from link2kaggle_colab.accounts.selection import select_account
from link2kaggle_colab.auth.real import ColabCredentialDetector, KaggleCredentialDetector
from link2kaggle_colab.core.config import Settings, load_config
from link2kaggle_colab.core.envspec import apply_account_env
from link2kaggle_colab.core.errors import (
    AccountDisabledError,
    AccountNotFoundError,
    ConfigError,
    InvalidInputError,
    L2KCError,
    PremiumNotAllowed,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import (
    Artifact,
    AuthStatus,
    GpuType,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
    ResourceSpec,
)
from link2kaggle_colab.core.paths import resolve_artifact_dest
from link2kaggle_colab.core.process import CommandRunner
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.jobs.manager import JobManager
from link2kaggle_colab.jobs.state_machine import ACTIVE_STATES
from link2kaggle_colab.jobs.sync import run_sync
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.providers.registry import build_provider
from link2kaggle_colab.quota.cache import QuotaCache
from link2kaggle_colab.scheduler.decision import SchedulerDecision, ScheduleRequest
from link2kaggle_colab.scheduler.selector import select_backend
from link2kaggle_colab.scheduler.snapshot import ProviderSnapshot
from link2kaggle_colab.storage.sqlite_store import JobRecord, SqliteJobStore, default_db_path

_T = TypeVar("_T")

_PROVIDER_ORDER = (ProviderName.kaggle, ProviderName.colab, ProviderName.fake)
_FIXTURE_WARNING = (
    "TEST FIXTURE: mock quota, balance, and membership are not a live account"
)


class JobIndexRecord(BaseModel):
    """V0.2 index shape kept for importers. Routing uses SQLite, not this model."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    provider: ProviderName


class SubmitDetails(BaseModel):
    """One ``Client`` submit.

    ``created`` is false when an existing row was returned and the provider
    was not called. ``state`` is the stored state, not a fresh status poll.
    ``handle.job_id`` is our id. The provider id stays in the store.
    """

    model_config = ConfigDict(extra="forbid")

    handle: JobHandle
    decision: SchedulerDecision | None = None
    created: bool
    state: JobState


_JOB_ROW_FIELDS = (
    "job_id",
    "provider",
    "account_id",
    "state",
    "provider_job_id",
    "remote_id",
    "message",
    "created_at",
    "updated_at",
    "finished_at",
    "exit_code",
    "artifacts",
    "idempotency_key",
    "origin",
    "details",
)


def _wants_gpu(gpu_type: GpuType | None, gpu_count: int) -> bool:
    """True when the request asks for an accelerator.

    ``gpu_type is None`` with a positive count means any GPU. ``NONE`` or a
    count of zero means CPU, which the V0.1 rule sends to ``fake``.
    """
    if gpu_count <= 0 or gpu_type is GpuType.NONE:
        return False
    return True


def _coerce_provider(value: ProviderName | str) -> ProviderName:
    if isinstance(value, ProviderName):
        return value
    try:
        return ProviderName(str(value))
    except ValueError as exc:
        raise InvalidInputError(
            f"unknown provider {value}",
            details={"provider": str(value)},
        ) from exc


def _coerce_gpu(value: GpuType | str | None) -> GpuType | None:
    if value is None or isinstance(value, GpuType):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return GpuType(text)
    except ValueError as exc:
        raise InvalidInputError(
            f"unknown gpu type {value}",
            details={"gpu_type": str(value)},
        ) from exc


class Client:
    """Synchronous facade over the enabled providers.

    Providers are built on first use from the registry. The first job
    operation opens ``state_dir/l2kc.db``, imports any V0.2 JSON index once,
    and routes submit, status, logs, download, and cancel through
    :class:`JobManager`. Our ``job_id`` is not the provider's id.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        config_path: Path | str | None = None,
        providers: dict[ProviderName, Provider] | None = None,
        runner: CommandRunner | None = None,
        clock: Callable[[], float] | None = None,
        job_store: SqliteJobStore | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_config(config_path)
        self._runner = runner
        self._accounts = AccountRegistry.from_settings(self.settings)
        self._quota_cache = QuotaCache(
            self.settings.quota.cache_ttl_seconds,
            clock or time.monotonic,
        )
        self._injected: dict[ProviderName, Provider] = {}
        if providers:
            for key, provider in providers.items():
                self._injected[_coerce_provider(key)] = provider
        self._built: dict[str, Provider] = {}
        self._job_store = job_store
        self._sleep = sleep
        self._manager: JobManager | None = None
        self._manager_lock = threading.Lock()

    def list_accounts(self) -> list[dict[str, Any]]:
        """Return every account in config order, including implicit defaults."""
        return [account.model_dump(mode="json") for account in self._accounts.list()]

    def get_account(self, account_id: str) -> dict[str, Any]:
        """Return one account. Unknown ids raise ``AccountNotFoundError``."""
        return self._accounts.get(account_id).model_dump(mode="json")

    def enable_account(self, account_id: str) -> dict[str, Any]:
        """Persist an enable override and drop the cached provider for that account."""
        self._accounts.set_enabled(account_id, True)
        self._drop_account_cache(account_id)
        return self.get_account(account_id)

    def disable_account(self, account_id: str) -> dict[str, Any]:
        """Persist a disable override and drop the cached provider for that account."""
        self._accounts.set_enabled(account_id, False)
        self._drop_account_cache(account_id)
        return self.get_account(account_id)

    def list_providers(self) -> list[dict[str, Any]]:
        """Return name, enabled, is_mock, mode, and capabilities for every provider."""

        def _load() -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for name in _PROVIDER_ORDER:
                enabled = self._is_enabled(name)
                row: dict[str, Any] = {
                    "name": name.value,
                    "enabled": enabled,
                    "is_mock": name is ProviderName.fake or self.settings.mode == "mock",
                    "mode": self.settings.mode,
                    "capabilities": None,
                }
                if enabled or name in self._injected:
                    try:
                        provider = self._provider(name)
                    except AccountDisabledError:
                        provider = None
                    if provider is not None:
                        row["is_mock"] = bool(provider.is_mock)
                        row["capabilities"] = provider.capabilities().model_dump(mode="json")
                rows.append(row)
            return rows

        return self._call(_load)

    def doctor(self) -> dict[str, Any]:
        """Report mode, writability, interpreter, CLI presence, and auth.

        CLI detection is ``shutil.which`` only. This method never executes
        ``kaggle`` or ``colab``. Real-mode auth is credential presence for each
        provider's default account, using that account's environ and home, not
        a live ``quota`` or ``usage`` call. A disabled default is reported as
        ``account_disabled`` and is not replaced by another account.
        """

        def _load() -> dict[str, Any]:
            state_dir = self.settings.state_dir
            providers: list[dict[str, Any]] = []
            for name in _PROVIDER_ORDER:
                enabled = self._is_enabled(name)
                entry: dict[str, Any] = {
                    "name": name.value,
                    "enabled": enabled,
                    "cli": self._cli_presence(name),
                    "auth": None,
                }
                if enabled or name in self._injected:
                    entry["auth"] = self._auth_summary(name)
                providers.append(entry)
            notice = ""
            if self.settings.mode == "mock":
                notice = (
                    "TEST FIXTURE: mock mode does not query a live account. "
                    "CLI binaries were not executed."
                )
            return {
                "mode": self.settings.mode,
                "allow_real_jobs": self.settings.allow_real_jobs,
                "state_dir": str(state_dir),
                "state_dir_writable": _directory_writable(state_dir),
                "python_version": sys.version.split()[0],
                "providers": providers,
                "fixture_notice": notice,
            }

        return self._call(_load)

    def auth_status(
        self,
        provider: ProviderName | str | None = None,
        *,
        account: str | None = None,
    ) -> list[AuthStatus]:
        """Return auth snapshots. Mock snapshots are test fixtures.

        ``account`` reads that account only. Otherwise each enabled provider
        uses its default account. There is no rotation.
        """

        def _load() -> list[AuthStatus]:
            accounts = self._accounts_for_call(provider, account)
            return [self._auth_for(item) for item in accounts]

        return self._call(_load)

    def get_quota(
        self,
        provider: ProviderName | str | None = None,
        *,
        account: str | None = None,
        refresh: bool = False,
    ) -> list[QuotaInfo]:
        """Return quota snapshots. Mock numbers are test fixtures.

        Uses the same :class:`QuotaCache` as :meth:`schedule`, keyed by
        provider and account, so a CLI quota read and a placement decision
        agree until the TTL expires. Another account is a separate entry.
        When ``refresh`` is True, each account that would be queried is
        invalidated in the cache before the read.
        """

        def _load() -> list[QuotaInfo]:
            accounts = self._accounts_for_call(provider, account)
            if refresh:
                for item in accounts:
                    self._quota_cache.invalidate(item.provider, item.id)
            return [self._quota_for(item) for item in accounts]

        return self._call(_load)

    def refresh_quota(
        self,
        provider: ProviderName | str | None = None,
        *,
        account: str | None = None,
    ) -> list[QuotaInfo]:
        """Drop cached quota for the selected accounts and read again."""
        return self.get_quota(provider, account=account, refresh=True)

    def list_resources(
        self,
        provider: ProviderName | str | None = None,
        *,
        account: str | None = None,
    ) -> list[ResourceSpec]:
        """Return accelerator offerings. Memory is per GPU, never a sum."""

        def _load() -> list[ResourceSpec]:
            rows: list[ResourceSpec] = []
            for item in self._accounts_for_call(provider, account):
                rows.extend(self._provider_for_account(item).list_resources())
            return rows

        return self._call(_load)

    def submit(self, spec: JobSpec, *, account: str | None = None) -> JobHandle:
        """Submit ``spec`` and return only the handle.

        GPU placement goes through :meth:`submit_with_decision`. A pinned
        provider is not a way around the premium gate. ``job_id`` is ours.
        """
        return self.submit_detailed(spec, account=account).handle

    def submit_with_decision(
        self,
        spec: JobSpec,
        *,
        account: str | None = None,
    ) -> tuple[JobHandle, SchedulerDecision | None]:
        """Submit ``spec`` and return the handle plus the scheduler decision.

        See :meth:`submit_detailed`. The decision is ``None`` when the
        scheduler was not used.
        """
        details = self.submit_detailed(spec, account=account)
        return details.handle, details.decision

    def submit_detailed(self, spec: JobSpec, *, account: str | None = None) -> SubmitDetails:
        """Submit ``spec`` and return the handle, decision, and dedupe flag.

        ``fake``, and any CPU request, skip the scheduler. A CPU request pinned
        to Kaggle or Colab is sent there as-is. A Colab CPU job still needs
        ``timeout_seconds`` so runtime spend is bounded; that check runs before
        any provider call. Every other GPU request, including one that names
        ``kaggle`` or ``colab``, is scheduled first. The chosen provider is
        called exactly once. There is no second provider. An ambiguous submit
        raises ``SubmitOutcomeUnknownError`` and is not resubmitted.

        ``created`` is false when an idempotency key with the same spec hash,
        or an in-flight spec hash, matched a stored row. That path does not
        call the provider. The same key with a different spec raises
        ``InvalidInputError`` before the provider is called.
        """

        def _load() -> SubmitDetails:
            working = spec
            bound: str | None = None
            if account is not None:
                resolved = self._require_explicit_account(account, spec.provider)
                bound = resolved.id
                if working.provider is None:
                    working = working.model_copy(update={"provider": resolved.provider})
            if working.provider is ProviderName.fake or not _wants_gpu(
                working.gpu_type, working.gpu_count
            ):
                # CPU skips the scheduler. The pinned provider is the one that
                # will run the job; Colab still has to bound runtime spend.
                if working.provider is ProviderName.colab and working.timeout_seconds is None:
                    raise InvalidInputError(
                        "timeout_seconds is required for Colab jobs so runtime spend is bounded",
                        provider="colab",
                    )
                handle, created, state = self._submit_unscheduled(working, account_id=bound)
                return SubmitDetails(handle=handle, decision=None, created=created, state=state)
            decision, chosen_account = self._decide(
                ScheduleRequest(
                    gpu_type=working.gpu_type,
                    gpu_count=working.gpu_count,
                    min_gpu_memory_gb_per_gpu=working.min_gpu_memory_gb_per_gpu,
                    timeout_seconds=working.timeout_seconds,
                    premium_confirmed=working.allow_premium,
                    high_mem=working.high_mem,
                    provider=working.provider,
                ),
                account=bound,
            )
            if decision.provider is None:
                details = {"decision": decision.to_dict()}
                if _premium_rejection(decision):
                    raise PremiumNotAllowed(_premium_message(decision), details=details)
                raise ResourceUnavailableError(
                    "no backend satisfies the request",
                    details=details,
                )
            accepted = next(item for item in decision.evaluations if item.accepted)
            updates: dict[str, Any] = {
                "provider": decision.provider,
                "gpu_type": decision.gpu_type,
            }
            if accepted.premium:
                updates["allow_premium"] = True
            chosen = working.model_copy(update=updates)
            handle, created, state = self._submit_chosen(
                decision.provider,
                chosen,
                account_id=chosen_account or bound,
            )
            return SubmitDetails(
                handle=handle,
                decision=decision,
                created=created,
                state=state,
            )

        return self._call(_load, provider=spec.provider.value if spec.provider else None)

    def schedule(
        self,
        spec_or_request: ScheduleRequest,
        *,
        account: str | None = None,
    ) -> SchedulerDecision:
        """Choose a backend. This does not submit a job.

        The snapshot loader is lazy: a provider is read only after one of its
        candidates passes the static checks. Auth, quota, and resources that
        raise :class:`L2KCError` become ``snapshot.error`` and are redacted.
        Auto mode uses one account per provider. ``account`` pins that
        account's provider. A disabled or missing default makes the provider
        ineligible and does not try another account of the same provider.
        """

        def _load() -> SchedulerDecision:
            decision, _chosen = self._decide(spec_or_request, account=account)
            return decision

        return self._call(_load)

    def status(self, job_id: str) -> JobResult:
        """Return the latest state of our ``job_id``.

        A disabled provider still answers for a job it already owns. An
        unknown or malformed id raises ``JobNotFoundError``.
        """

        def _load() -> JobResult:
            self._require_job_account(job_id)
            return run_sync(lambda: self._jobs().status(job_id))

        return self._call(_load)

    def logs(self, job_id: str) -> str:
        """Return log text for ``job_id``."""

        def _load() -> str:
            self._require_job_account(job_id)
            return run_sync(lambda: self._jobs().logs(job_id))

        return self._call(_load)

    def download(self, job_id: str, dest_dir: Path | None = None) -> list[Artifact]:
        """Copy artifacts for ``job_id`` into ``dest_dir`` under the artifact root.

        ``dest_dir`` of ``None`` uses ``effective_artifact_root() / job_id``.
        Paths outside the artifact root raise ``PathSecurityError``.
        """

        def _load() -> list[Artifact]:
            self._require_job_account(job_id)
            target = resolve_artifact_dest(
                self.settings.effective_artifact_root(),
                dest_dir,
                job_id,
            )
            return run_sync(lambda: self._jobs().download(job_id, target))

        return self._call(_load)

    def cancel(self, job_id: str) -> JobResult:
        """Cancel ``job_id``. Providers without cancel raise ``NotSupportedError``."""

        def _load() -> JobResult:
            self._require_job_account(job_id)
            return run_sync(lambda: self._jobs().cancel(job_id))

        return self._call(_load)

    def list_jobs(
        self,
        states: list[JobState | str] | None = None,
        provider: ProviderName | str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return stored jobs. This does not call a provider or submit."""

        def _load() -> list[dict[str, Any]]:
            name = None if provider is None else _coerce_provider(provider).value
            records = self._jobs().list_jobs(states=states, provider=name, limit=limit)
            return [_job_row(record) for record in records]

        return self._call(_load)

    def reconcile_jobs(self, job_ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Poll active jobs, or the given ids. This never submits."""

        def _load() -> list[dict[str, Any]]:
            manager = self._jobs()
            if job_ids is None:
                active = list(ACTIVE_STATES)
                records: list[JobRecord | None] = list(manager.list_jobs(states=active))
            else:
                records = []
                for job_id in job_ids:
                    try:
                        records.append(manager.get(job_id))
                    except L2KCError:
                        # Missing and malformed ids stay a manager report.
                        records.append(None)
            skipped: dict[str, tuple[JobRecord, str]] = {}
            runnable: list[str] = []
            order: list[str] = []
            for index, record in enumerate(records):
                if record is None:
                    if job_ids is None:
                        continue
                    order.append(job_ids[index])
                    runnable.append(job_ids[index])
                    continue
                order.append(record.job_id)
                reason = self._reconcile_skip_reason(record)
                if reason is None:
                    runnable.append(record.job_id)
                else:
                    skipped[record.job_id] = (record, reason)
            reports = run_sync(lambda: manager.reconcile(runnable))
            by_id = {item.job_id: item.model_dump(mode="json") for item in reports}
            ordered: list[dict[str, Any]] = []
            for job_id in order:
                found = skipped.get(job_id)
                if found is None:
                    ordered.append(by_id[job_id])
                    continue
                record, reason = found
                ordered.append(
                    {
                        "job_id": record.job_id,
                        "provider": record.provider.value,
                        "before": record.state.value,
                        "after": record.state.value,
                        "ok": True,
                        "error": None,
                        "note": None,
                        "skipped": True,
                        "reason": reason,
                    }
                )
            return ordered

        return self._call(_load)

    def recommend_backend(
        self,
        gpu_type: GpuType | str | None = None,
        gpu_count: int = 1,
        min_gpu_memory_gb_per_gpu: float | None = None,
        allow_premium: bool = False,
        timeout_seconds: int | None = None,
        provider: ProviderName | str | None = None,
        *,
        high_mem: bool = False,
        account: str | None = None,
    ) -> dict[str, Any]:
        """Pick a backend without submitting.

        CPU requests, and ``provider="fake"``, use the offline fake provider.
        Every GPU request goes through :meth:`schedule`. ``allow_premium`` is
        the per-job confirmation and does not override
        ``scheduler.allow_premium_gpu``. ``provider`` ``auto`` means the
        scheduler may choose. Memory is never summed across GPUs.
        """

        def _load() -> dict[str, Any]:
            gpu = _coerce_gpu(gpu_type)
            if gpu_count < 0:
                raise InvalidInputError("gpu_count must be >= 0")
            if min_gpu_memory_gb_per_gpu is not None and min_gpu_memory_gb_per_gpu <= 0:
                raise InvalidInputError("min_gpu_memory_gb_per_gpu must be > 0")
            if timeout_seconds is not None and timeout_seconds <= 0:
                raise InvalidInputError("timeout_seconds must be > 0")
            pinned = _scheduler_provider(provider)
            pinned_account: Account | None = None
            if account is not None:
                pinned_account = self._require_explicit_account(account, pinned)
                pinned = pinned_account.provider
            cpu = pinned is ProviderName.fake or not _wants_gpu(gpu, gpu_count)
            if account is None and cpu:
                reasons: list[str] = []
                warnings: list[str] = []
                if self.settings.mode == "mock":
                    warnings.append(_FIXTURE_WARNING)
                return self._recommend_cpu(reasons, warnings)
            if pinned_account is not None and not _wants_gpu(gpu, gpu_count):
                if pinned is ProviderName.fake:
                    reasons = []
                    warnings = []
                    if self.settings.mode == "mock":
                        warnings.append(_FIXTURE_WARNING)
                    return self._recommend_cpu(reasons, warnings)
                resources = self._provider_for_account(pinned_account).list_resources()
                resource = next((item for item in resources if item.cpu_only), None)
                warnings = [_FIXTURE_WARNING] if self.settings.mode == "mock" else []
                return {
                    "provider": pinned.value if pinned is not None else None,
                    "resource": None if resource is None else resource.model_dump(mode="json"),
                    "reasons": [f"account {pinned_account.id} pinned"],
                    "warnings": warnings,
                }
            request = ScheduleRequest(
                gpu_type=gpu,
                gpu_count=gpu_count,
                min_gpu_memory_gb_per_gpu=min_gpu_memory_gb_per_gpu,
                timeout_seconds=timeout_seconds,
                premium_confirmed=allow_premium,
                high_mem=high_mem,
                provider=pinned,
            )
            return self.schedule(request, account=account).to_dict()

        return self._call(_load)

    def _decide(
        self,
        request: ScheduleRequest,
        *,
        account: str | None,
    ) -> tuple[SchedulerDecision, str | None]:
        """Run ``select_backend`` and return the chosen account id, if any.

        ``account`` pins ``request.provider`` to that account's provider.
        A conflicting provider is ``ConfigError``. Each provider snapshot uses
        ``select_account`` once. A raising account makes that provider
        ineligible and is not replaced by another account.
        """
        if account is not None:
            pinned = self._require_explicit_account(account, request.provider)
            if request.provider is None:
                request = request.model_copy(update={"provider": pinned.provider})
        selected: dict[ProviderName, str] = {}

        def snapshots(name: ProviderName) -> ProviderSnapshot:
            return self._snapshot_account(name, explicit=account, selected=selected)

        decision = select_backend(request, self.settings.scheduler, snapshots)
        if self.settings.mode == "mock" and _FIXTURE_WARNING not in decision.warnings:
            decision.warnings.append(_FIXTURE_WARNING)
        chosen = None if decision.provider is None else selected.get(decision.provider)
        return decision, chosen

    def _snapshot_account(
        self,
        name: ProviderName,
        *,
        explicit: str | None,
        selected: dict[ProviderName, str],
    ) -> ProviderSnapshot:
        """Auth, cached quota, and resources for one provider's selected account.

        A provider setting that is disabled, and was not injected, returns
        before any account lookup so the reason stays ``"<provider> is disabled"``.
        A disabled or missing account sets ``enabled`` false and an error that
        names the account. The other account of that provider is not built.
        """
        injected = name in self._injected
        enabled = self._is_enabled(name) or injected
        keeps_runtime = (
            name is ProviderName.colab and self.settings.providers.colab.keep_runtime
        )
        snapshot = ProviderSnapshot(
            provider=name,
            enabled=enabled,
            keeps_runtime=keeps_runtime,
        )
        if not self._is_enabled(name) and not injected:
            return snapshot
        try:
            account = select_account(self._accounts, name, explicit)
        except (AccountDisabledError, AccountNotFoundError) as exc:
            snapshot.enabled = False
            raw_id = exc.details.get("account_id")
            label = raw_id if isinstance(raw_id, str) and raw_id else name.value
            snapshot.error = redact(f"{exc.code}: account {label}")
            return snapshot
        selected[name] = account.id
        try:
            provider = self._provider_for_account(account)
            snapshot.auth = self._stamp_auth(provider.auth_status(), account.id)
            snapshot.quota = self._cached_quota(provider, account)
            snapshot.resources = provider.list_resources()
        except L2KCError as exc:
            snapshot.error = redact(f"{exc.code}: {exc.message} (account {account.id})")
        return snapshot

    def _submit_chosen(
        self,
        name: ProviderName,
        spec: JobSpec,
        *,
        account_id: str | None,
    ) -> tuple[JobHandle, bool, JobState]:
        """Call ``name`` once. Invalidate that account's quota cache either way.

        The enabled check runs before a row is inserted. A dedupe hit returns
        the stored handle and does not call the provider. An exception is
        re-raised after the cache drop. This method does not try another
        provider or another account.
        """
        return self._submit_with_manager(
            name,
            spec,
            invalidate_quota=True,
            account_id=account_id,
        )

    def _submit_unscheduled(
        self,
        spec: JobSpec,
        *,
        account_id: str | None = None,
    ) -> tuple[JobHandle, bool, JobState]:
        """Submit a CPU job, or anything pinned to ``fake``, without scheduling.

        An unspecified GPU on ``fake`` is refused. ``fake`` has no accelerator,
        which matches the old resolver that found no fitting resource.
        """
        chosen = spec
        if chosen.provider is None:
            picked = self._recommend_cpu([], [])
            provider_name = picked.get("provider")
            if not provider_name:
                raise ResourceUnavailableError(
                    "no provider can run this job",
                    details={"reasons": picked.get("reasons", [])},
                )
            chosen = chosen.model_copy(update={"provider": ProviderName(str(provider_name))})
        if (
            chosen.provider is ProviderName.fake
            and _wants_gpu(chosen.gpu_type, chosen.gpu_count)
            and chosen.gpu_type is None
        ):
            raise ResourceUnavailableError(
                "no GPU resource fits gpu_count, per-GPU memory, and allow_premium",
                provider=chosen.provider.value,
            )
        if chosen.provider is None:
            raise ResourceUnavailableError("no provider can run this job")
        return self._submit_with_manager(
            chosen.provider,
            chosen,
            invalidate_quota=False,
            account_id=account_id,
        )

    def _submit_with_manager(
        self,
        name: ProviderName,
        spec: JobSpec,
        *,
        invalidate_quota: bool,
        account_id: str | None,
    ) -> tuple[JobHandle, bool, JobState]:
        """Submit once. Disabled providers and accounts are refused first.

        A provider that is turned off in settings raises ``ConfigError`` when
        no account was named, before the default account is considered. An
        explicit disabled account raises ``AccountDisabledError`` before any
        provider method runs.
        """
        if account_id is None:
            if not self._is_enabled(name) and name not in self._injected:
                raise ConfigError(
                    f"provider {name.value} is disabled",
                    provider=name.value,
                )
            account = self._accounts.default_for(name)
            account_id = account.id
        else:
            account = self._accounts.get(account_id)
            if account.provider is not name:
                raise ConfigError(
                    "account provider does not match",
                    details={
                        "field": "account",
                        "account_id": account.id,
                        "provider": name.value,
                    },
                )
            if not account.enabled:
                raise AccountDisabledError(
                    "account is disabled",
                    account_id=account.id,
                    provider=name.value,
                )
        try:
            outcome = run_sync(lambda: self._jobs().submit(name, spec, account_id))
        except Exception:
            if invalidate_quota:
                self._quota_cache.invalidate(name, account_id)
            raise
        if invalidate_quota:
            self._quota_cache.invalidate(name, account_id)
        return outcome.handle, outcome.created, outcome.record.state

    def _jobs(self) -> JobManager:
        """Build the store and manager on first use. Later calls share them."""
        with self._manager_lock:
            if self._manager is not None:
                return self._manager
            store = self._job_store
            if store is None:
                store = SqliteJobStore(default_db_path(self.settings.state_dir))
                self._job_store = store
            store.migrate_json_index(self.settings.state_dir)
            provider_for = self._provider_for_manager
            if self._sleep is None:
                manager = JobManager(store, provider_for, self.settings.jobs)
            else:
                manager = JobManager(
                    store,
                    provider_for,
                    self.settings.jobs,
                    sleep=self._sleep,
                )
            self._manager = manager
            return manager

    def _provider_for_manager(self, name: ProviderName, account_id: str) -> Provider:
        """Provider for status, logs, download, cancel, and reconcile.

        A job whose implicit account is off only because the provider setting
        is off still resolves, so an existing row can be read. A configured
        or runtime-disabled account raises ``AccountDisabledError``.
        """
        account = self._accounts.resolve_for_record(account_id, name)
        if not account.enabled and not self._allow_provider_disabled_job(account):
            raise AccountDisabledError(
                "account is disabled",
                account_id=account.id,
                provider=name.value,
            )
        return self._provider_for_account(account)

    def _recommend_cpu(self, reasons: list[str], warnings: list[str]) -> dict[str, Any]:
        if not self._is_enabled(ProviderName.fake) and ProviderName.fake not in self._injected:
            reasons.append("fake provider is disabled and no GPU was requested")
            return {"provider": None, "resource": None, "reasons": reasons, "warnings": warnings}
        try:
            resources = self._provider(ProviderName.fake).list_resources()
        except L2KCError as exc:
            reasons.append(f"fake: {exc.code}: {exc.message}")
            return {"provider": None, "resource": None, "reasons": reasons, "warnings": warnings}
        resource = next((item for item in resources if item.cpu_only), None)
        reasons.append("no GPU requested; using the offline fake provider")
        return {
            "provider": ProviderName.fake.value,
            "resource": None if resource is None else resource.model_dump(mode="json"),
            "reasons": reasons,
            "warnings": warnings,
        }

    def _cli_presence(self, name: ProviderName) -> dict[str, Any]:
        """Report whether the official binary is on PATH. Never execute it."""
        if name is ProviderName.fake:
            return {
                "cli_path": None,
                "found": False,
                "resolved_path": None,
                "executed": False,
                "note": "fake has no external CLI",
            }
        cli_path = (
            self.settings.providers.kaggle.cli_path
            if name is ProviderName.kaggle
            else self.settings.providers.colab.cli_path
        )
        resolved = shutil.which(cli_path)
        return {
            "cli_path": cli_path,
            "found": resolved is not None,
            "resolved_path": resolved,
            "executed": False,
        }

    def _auth_summary(self, name: ProviderName) -> dict[str, Any]:
        try:
            status = self._auth_snapshot(name)
        except L2KCError as exc:
            return {"error": exc.to_dict()}
        except Exception as exc:
            wrapped = L2KCError(
                redact(str(exc).strip() or type(exc).__name__),
                code="internal_error",
                provider=name.value,
                details={"exception_type": type(exc).__name__},
            )
            return {"error": wrapped.to_dict()}
        return status.model_dump(mode="json")

    def _auth_snapshot(self, name: ProviderName) -> AuthStatus:
        # ``_provider`` raises ``AccountDisabledError`` for a disabled default
        # and does not substitute another account of the same provider.
        provider = self._provider(name)
        account = self._default_account(name)
        if provider.is_mock:
            return self._stamp_auth(provider.auth_status(), account.id)
        # Real adapters verify auth by invoking the official CLI. Doctor stops
        # at credential presence for this account's overlay, so it cannot reach
        # the network or the process HOME.
        if name is ProviderName.kaggle or name is ProviderName.colab:
            return self._stamp_auth(self._credential_presence(account), account.id)
        return self._stamp_auth(provider.auth_status(), account.id)

    def _default_account(self, name: ProviderName) -> Account:
        """Return the registry default, including a disabled row.

        Does not fall through to another account. ``_provider`` has already
        rejected a disabled default when no provider was injected.
        """
        found = next((item for item in self._accounts.list(name) if item.is_default), None)
        if found is None:
            raise AccountNotFoundError(
                "no account is configured for this provider",
                account_id=f"{name.value}-default",
                provider=name.value,
            )
        return found

    def _credential_presence(self, account: Account) -> AuthStatus:
        """Detect credential files for ``account`` without running a CLI.

        The overlay matches the real adapters: ``resolve_credential`` then
        ``apply_account_env``. File checks stay on ``Path.is_file``.
        """
        ref_name = account.credential_ref
        ref_config = self.settings.credentials.get(ref_name) if ref_name else None
        resolved = resolve_credential(ref_name, ref_config, account.provider, self.settings)
        environ = apply_account_env(resolved.env, os.environ)
        home = resolved.home
        if home is None and "HOME" in resolved.env.set:
            home = Path(resolved.env.set["HOME"])
        if account.provider is ProviderName.kaggle:
            return KaggleCredentialDetector(home=home, environ=environ).status()
        return ColabCredentialDetector(
            home=home,
            environ=environ,
            auth=self.settings.providers.colab.auth,
        ).status()

    def _selected(self, provider: ProviderName | str | None) -> list[ProviderName]:
        if provider is None:
            return [name for name in _PROVIDER_ORDER if self._is_enabled(name)]
        name = _coerce_provider(provider)
        if name not in self._injected and not self._is_enabled(name):
            raise ConfigError(
                f"provider {name.value} is disabled",
                provider=name.value,
            )
        return [name]

    def _is_enabled(self, name: ProviderName) -> bool:
        if name is ProviderName.kaggle:
            return self.settings.providers.kaggle.enabled
        if name is ProviderName.colab:
            return self.settings.providers.colab.enabled
        return self.settings.providers.fake.enabled

    def _provider(self, name: ProviderName, *, allow_disabled: bool = False) -> Provider:
        injected = self._injected.get(name)
        if injected is not None:
            return injected
        if not allow_disabled and not self._is_enabled(name):
            raise ConfigError(
                f"provider {name.value} is disabled",
                provider=name.value,
            )
        try:
            account = self._accounts.default_for(name)
        except AccountDisabledError:
            if not allow_disabled:
                raise
            account = next(item for item in self._accounts.list(name) if item.is_default)
        return self._provider_for_account(account)

    def _provider_for_account(self, account: Account) -> Provider:
        """Return the cached adapter for ``account``.

        An injected provider is the default account of that provider. Other
        accounts are built with ``build_provider`` and cached by account id.
        """
        if account.is_default:
            injected = self._injected.get(account.provider)
            if injected is not None:
                return injected
        cached = self._built.get(account.id)
        if cached is None:
            cached = build_provider(
                account.provider,
                self.settings,
                runner=self._runner,
                account=account,
            )
            self._built[account.id] = cached
        return cached

    def _accounts_for_call(
        self,
        provider: ProviderName | str | None,
        account: str | None,
    ) -> list[Account]:
        """Resolve the accounts a read method should call.

        An explicit account is checked before any provider call. Without one,
        each selected provider uses its default account.
        """
        if account is not None:
            expected = None if provider is None else _coerce_provider(provider)
            return [self._require_explicit_account(account, expected)]
        return [select_account(self._accounts, name, None) for name in self._selected(provider)]

    def _require_explicit_account(
        self,
        account_id: str,
        provider: ProviderName | None,
    ) -> Account:
        """Return an enabled account, or raise before any provider call.

        A provider that does not match ``provider`` is ``ConfigError``. An
        unknown id is ``AccountNotFoundError``. A disabled account is
        ``AccountDisabledError``.
        """
        account = self._accounts.get(account_id)
        if provider is not None and account.provider is not provider:
            raise ConfigError(
                "account provider does not match",
                details={
                    "field": "account",
                    "account_id": account.id,
                    "provider": provider.value,
                },
            )
        if not account.enabled:
            raise AccountDisabledError(
                "account is disabled",
                account_id=account.id,
                provider=account.provider.value,
            )
        return account

    def _require_job_account(self, job_id: str) -> JobRecord:
        """Load the row and refuse a disabled or unknown account.

        The store lookup does not call a provider. Malformed ids stay
        ``JobNotFoundError``.
        """
        record = self._jobs().get(job_id)
        reason = self._reconcile_skip_reason(record)
        if reason == "account_not_found":
            raise AccountNotFoundError(
                "account not found",
                account_id=record.account_id,
                provider=record.provider.value,
            )
        if reason == "account_disabled":
            raise AccountDisabledError(
                "account is disabled",
                account_id=record.account_id,
                provider=record.provider.value,
            )
        return record

    def _reconcile_skip_reason(self, record: JobRecord) -> str | None:
        """Why reconcile should leave ``record`` untouched, or None to poll it."""
        try:
            account = self._accounts.resolve_for_record(record.account_id, record.provider)
        except AccountNotFoundError:
            return "account_not_found"
        except AccountDisabledError:
            return "account_disabled"
        if not account.enabled and not self._allow_provider_disabled_job(account):
            return "account_disabled"
        return None

    def _allow_provider_disabled_job(self, account: Account) -> bool:
        """True when the account is off only because its provider setting is off.

        Runtime disables recorded in ``accounts_state.json`` are not included.
        Configured accounts are not included.
        """
        return (
            account.implicit
            and not self._is_enabled(account.provider)
            and not self._accounts.override_disabled(account.id)
        )

    def _drop_account_cache(self, account_id: str) -> None:
        self._built.pop(account_id, None)
        try:
            account = self._accounts.get(account_id)
        except AccountNotFoundError:
            return
        self._quota_cache.invalidate(account.provider, account_id)

    def _quota_for(self, account: Account) -> QuotaInfo:
        provider = self._provider_for_account(account)
        cached = self._cached_quota(provider, account)
        return cached.quota

    def _cached_quota(self, provider: Provider, account: Account) -> Any:
        def _load() -> QuotaInfo:
            quota = provider.get_quota()
            if quota.account_id is None:
                return quota.model_copy(update={"account_id": account.id})
            return quota

        return self._quota_cache.get(account.provider, _load, account.id)

    def _auth_for(self, account: Account) -> AuthStatus:
        provider = self._provider_for_account(account)
        return self._stamp_auth(provider.auth_status(), account.id)

    def _stamp_auth(self, status: AuthStatus, account_id: str) -> AuthStatus:
        if status.account_id is None:
            return status.model_copy(update={"account_id": account_id})
        return status

    def _call(self, func: Callable[[], _T], *, provider: str | None = None) -> _T:
        try:
            return func()
        except L2KCError:
            raise
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            raise L2KCError(
                redact(text),
                code="internal_error",
                provider=provider,
                details={"exception_type": type(exc).__name__},
            ) from exc


def _job_row(record: JobRecord) -> dict[str, Any]:
    """JSON-ready job row. Omits the spec hash and the optimistic version."""
    dumped = record.model_dump(mode="json")
    return {field: dumped[field] for field in _JOB_ROW_FIELDS}


def _directory_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".l2kc-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def _scheduler_provider(value: ProviderName | str | None) -> ProviderName | None:
    """Return the pinned provider. ``None`` and ``auto`` leave the choice open."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    return _coerce_provider(value)


# Shape mismatches are not a premium refusal. A T4 that misses quota, or a
# request for H100 with no H100 candidate, stays ResourceUnavailableError.
_INELIGIBLE_MARKERS = (
    "does not match requested",
    "GPU(s), requested",
    "is below requested",
    "variant not selectable",
)


def _ineligible(reasons: list[str]) -> bool:
    """True when the candidate cannot satisfy the requested GPU shape."""
    return any(marker in reason for reason in reasons for marker in _INELIGIBLE_MARKERS)


def _premium_rejection(decision: SchedulerDecision) -> bool:
    """True when every eligible candidate failed only on the premium gate.

    Candidates rejected for GPU type, count, per-GPU memory, or an unselectable
    variant are ignored. A quota, timeout, auth, or fallback failure on any
    remaining candidate keeps the error as resource unavailable.
    """
    if any(item.accepted for item in decision.evaluations):
        return False
    eligible = [item for item in decision.evaluations if not _ineligible(item.reasons)]
    if not eligible:
        return False
    if any(
        "premium GPU" not in reason
        for item in eligible
        for reason in item.reasons
    ):
        return False
    return any("premium GPU" in reason for item in eligible for reason in item.reasons)


def _premium_message(decision: SchedulerDecision) -> str:
    """First premium-gate reason, for the error the caller sees."""
    for item in decision.evaluations:
        for reason in item.reasons:
            if "premium GPU" in reason:
                return reason
    return "premium GPU is not allowed"
