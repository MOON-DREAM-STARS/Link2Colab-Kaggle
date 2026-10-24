"""Colab adapter that shells out to the official ``colab`` CLI.

Global options (``--auth``) are placed before the subcommand. V0.1 runs a
session with ``new``, ``upload``, ``exec``, ``ls``, ``download``, and always
``stop`` unless ``colab.keep_runtime`` is set. A failed stop is retried once.
``colab run`` is not used. When ``timeout_seconds`` is unset, ``exec`` uses
``colab.default_exec_timeout_seconds``.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from link2kaggle_colab.accounts.identity import (
    identity_path,
    infer_colab_membership_tier,
    utc_now_iso,
    write_identity,
)
from link2kaggle_colab.auth.real import ColabCredentialDetector
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.envspec import AccountEnv, apply_account_env
from link2kaggle_colab.core.errors import (
    ConfigError,
    InvalidInputError,
    JobNotFoundError,
    L2KCError,
    NotSupportedError,
    PremiumNotAllowed,
    ProviderCLIError,
    RealModeDisabledError,
    ResourceUnavailableError,
    RuntimeDisconnectedError,
)
from link2kaggle_colab.core.models import (
    TPU_TYPES,
    Artifact,
    AuthMethod,
    AuthState,
    AuthStatus,
    Capability,
    GpuType,
    JobHandle,
    JobResult,
    JobSpec,
    JobState,
    ProviderName,
    QuotaInfo,
    QuotaUnit,
    ResourceSpec,
)
from link2kaggle_colab.core.paths import ensure_within, safe_join
from link2kaggle_colab.core.process import (
    CommandResult,
    CommandRunner,
    SubprocessRunner,
    absolute_cli_path,
    redact_excerpt,
    validate_cli_arg,
    validate_remote_relpath,
    validate_session_name,
)
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.providers.colab.ccu_info import (
    ccu_info_snapshot,
    read_token_account,
    token_json_path,
)
from link2kaggle_colab.scheduler.policy import premium_gate

_JOB_ID = re.compile(r"^[a-f0-9]{32}$")
_STEP_TIMEOUT = 120.0
# One retry after the first failure. Total attempts stay at least two.
_STOP_ATTEMPTS = 2
_PREMIUM = frozenset({GpuType.A100, GpuType.H100, GpuType.G4, *TPU_TYPES})
_GPU_FLAG = {
    GpuType.T4: "T4",
    GpuType.L4: "L4",
    GpuType.G4: "G4",
    GpuType.A100: "A100",
    GpuType.H100: "H100",
}
_TPU_FLAG = {
    GpuType.TPU_V5E1: "v5e1",
    GpuType.TPU_V6E1: "v6e1",
}
# Colab honors --high-mem for these; L4 and TPU ignore / reject upstream.
_HIGH_MEM_OK = frozenset({GpuType.NONE, GpuType.T4, GpuType.A100, GpuType.H100, GpuType.G4})
_BALANCE = re.compile(
    r"Current balance:\s*([0-9]+(?:\.[0-9]+)?)\s*compute units",
    re.IGNORECASE,
)
_RATE = re.compile(
    r"Usage rate:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*hr",
    re.IGNORECASE,
)
_ELIGIBLE_GPUS = re.compile(
    r"Eligible GPUs?:\s*(.+)",
    re.IGNORECASE,
)
_ELIGIBILITY = "eligibility unknown until `colab new` succeeds"
# CLI/runtime drop text only. A bare "disconnect" in user output is not this.
_RUNTIME_DISCONNECT = re.compile(
    r"\b(?:the\s+)?runtime\s+disconnected\b",
    re.IGNORECASE,
)


class ColabJobRecord(BaseModel):
    """On-disk Colab job. Credential material does not belong here."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    session: str
    state: JobState
    submitted_at: datetime
    exit_code: int | None = None
    log_path: str = ""
    artifacts: list[str] = Field(default_factory=list)
    runtime_stopped: bool = False
    stop_returncode: int | None = None
    message: str = ""
    outputs_dir: str = ""
    finished_at: datetime | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _runtime_disconnected_output(stderr: str, combined: str) -> bool:
    """True for a Colab runtime-disconnect marker, not a bare "disconnect" substring."""
    return bool(_RUNTIME_DISCONNECT.search(stderr) or _RUNTIME_DISCONNECT.search(combined))


def _auth_problem(text: str) -> AuthState | None:
    lowered = text.lower()
    if any(token in lowered for token in ("invalid_grant", "reauthentication", "expired")):
        return AuthState.expired
    if "401" in lowered or "403" in lowered:
        return AuthState.invalid
    return None


def _parse_eligible_gpus(stdout: str) -> list[str] | None:
    """Parse ``Eligible GPUs: a, b`` when present. Otherwise leave unknown."""
    match = _ELIGIBLE_GPUS.search(stdout)
    if match is None:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    names = [part for part in parts if part]
    return names or None


def _parse_usage(stdout: str) -> QuotaInfo:
    balance = _BALANCE.search(stdout)
    if balance is None:
        raise ValueError("balance missing")
    remaining = float(balance.group(1))
    eligible = _parse_eligible_gpus(stdout)
    # Informational only — scheduler must not read membership.
    membership = infer_colab_membership_tier(remaining, eligible)
    details: dict[str, object] = {"membership": membership}
    rate = _RATE.search(stdout)
    if rate is not None:
        details["usage_rate_per_hour"] = float(rate.group(1))
    return QuotaInfo(
        provider=ProviderName.colab,
        unit=QuotaUnit.compute_units,
        total=None,
        used=None,
        remaining=remaining,
        is_fixture=False,
        source="colab usage",
        message=(
            f"membership={membership} (informational; free_pro ≠ paid; "
            "scheduler uses eligible_gpus, not membership)"
        ),
        details=details,
        eligible_gpus=eligible,
    )


def _quota_from_ccu_info(snapshot: dict[str, object]) -> QuotaInfo:
    """Build QuotaInfo from a parsed ccu-info snapshot."""
    balance = float(snapshot["balance"])  # type: ignore[arg-type]
    eligible = snapshot.get("eligible_gpus")
    if eligible is not None and not isinstance(eligible, list):
        eligible = None
    eligible_list = [str(item) for item in eligible] if isinstance(eligible, list) else None
    membership = infer_colab_membership_tier(balance, eligible_list)
    return QuotaInfo(
        provider=ProviderName.colab,
        unit=QuotaUnit.compute_units,
        total=None,
        used=None,
        remaining=balance,
        is_fixture=False,
        source="ccu-info",
        message=(
            f"membership={membership} (informational; free_pro ≠ paid; "
            "scheduler uses eligible_gpus, not membership)"
        ),
        details={"membership": membership},
        eligible_gpus=eligible_list,
    )


def _plain_filename(value: str) -> str | None:
    """Accept one remote path segment. Unsafe names are skipped, not executed."""
    if not value or "/" in value or "\\" in value:
        return None
    try:
        return validate_remote_relpath(value)
    except InvalidInputError:
        return None


def _to_result(record: ColabJobRecord) -> JobResult:
    return JobResult(
        job_id=record.job_id,
        provider=ProviderName.colab,
        state=record.state,
        remote_id=record.session,
        message=record.message,
        finished_at=record.finished_at,
        exit_code=record.exit_code,
        artifacts=list(record.artifacts),
        is_mock=False,
        details=dict(record.details),
    )


def _runtime_stop_failed_message(session: str) -> str:
    """Warning stored when ``colab stop`` cannot release the runtime."""
    return (
        "runtime_stop_failed: the runtime may still be consuming CCU. "
        f"Stop it manually with `colab stop {session}`."
    )


def _merge_message(message: str, warning: str) -> str:
    if not warning or warning in message:
        return message
    if not message:
        return warning
    return f"{message}\n{warning}"


def _drop_stop_warning(message: str, session: str) -> str:
    warning = _runtime_stop_failed_message(session)
    if message == warning:
        return ""
    suffix = f"\n{warning}"
    if message.endswith(suffix):
        return message[:-len(suffix)]
    return message


@dataclass(frozen=True)
class _StopOutcome:
    """One finished ``colab stop`` sequence, including its retries."""

    stopped: bool
    returncode: int | None = None
    stderr: str = ""
    error: BaseException | None = None


def _record_after_stop(
    record: ColabJobRecord,
    outcome: _StopOutcome,
    *,
    terminal: bool,
) -> ColabJobRecord:
    """Apply a stop attempt without changing a terminal execution state."""
    details = dict(record.details)
    if outcome.stopped:
        details.pop("runtime_stop_failed", None)
        if terminal:
            return record.model_copy(
                update={
                    "runtime_stopped": True,
                    "stop_returncode": outcome.returncode,
                    "details": details,
                    "message": _drop_stop_warning(record.message, record.session),
                }
            )
        return record.model_copy(
            update={
                "state": JobState.CANCELLED,
                "runtime_stopped": True,
                "stop_returncode": outcome.returncode,
                "details": details,
                "message": "cancelled via colab stop",
                "finished_at": datetime.now(UTC),
            }
        )
    details["runtime_stop_failed"] = True
    return record.model_copy(
        update={
            "runtime_stopped": False,
            "stop_returncode": outcome.returncode,
            "details": details,
            "message": _merge_message(
                record.message,
                _runtime_stop_failed_message(record.session),
            ),
        }
    )


class ColabCliProvider(Provider):
    """Official Colab CLI. Read-only calls need ``mode: real``.

    ``new``, ``upload``, and ``exec`` also need ``allow_real_jobs``. ``stop``
    only needs real mode because it releases a runtime.
    """

    name = ProviderName.colab
    is_mock = False

    def __init__(
        self,
        settings: Settings,
        *,
        runner: CommandRunner | None = None,
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        stop_backoff_seconds: float = 0.0,
        sleeper: Callable[[float], None] | None = None,
        account_env: AccountEnv | None = None,
        state_dir: Path | None = None,
        account_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._runner = runner if runner is not None else SubprocessRunner()
        self._stop_backoff_seconds = stop_backoff_seconds
        self._sleeper = time.sleep if sleeper is None else sleeper
        self._account_env = account_env
        self._state_dir = None if state_dir is None else Path(state_dir)
        self._account_id = account_id
        detector_environ = environ
        detector_home = home
        if account_env is not None:
            if detector_environ is None:
                detector_environ = apply_account_env(account_env, os.environ)
            if detector_home is None and "HOME" in account_env.set:
                detector_home = Path(account_env.set["HOME"])
        self._home = Path(detector_home) if detector_home is not None else Path.home()
        self._detector = ColabCredentialDetector(
            home=self._home,
            environ=detector_environ,
            auth=settings.providers.colab.auth,
        )

    def capabilities(self) -> Capability:
        return Capability(
            provider=self.name,
            supports_submit=True,
            supports_status=True,
            supports_logs=True,
            supports_cancel=True,
            supports_download=True,
            supports_quota=True,
            supports_interactive_session=True,
            is_batch_only=False,
            notes=[
                "list_sessions() returns raw lines from `colab sessions`. "
                "It is not part of the base provider interface.",
                "The CLI can hold an interactive session, but V0.1 does not expose one.",
                "Cancel calls `colab stop` and needs mode real. It does not need allow_real_jobs.",
                "Submit calls `colab stop` afterwards unless colab.keep_runtime is true.",
                "Membership is display/identity only (free_pro ≠ paid).",
                "Scheduler uses eligible_gpus, never membership_tier.",
                "A100 per-GPU memory is unknown because 40 GB vs 80 GB is not selectable.",
                "TPU uses `colab new --tpu {v5e1|v6e1}` (mutually exclusive with --gpu).",
                "high_mem maps to `colab new --high-mem` (CPU/T4/A100/H100/G4; not L4/TPU).",
                "Premium eligibility is unknown until `colab new` succeeds.",
                "colab exec does not forward script args in V0.1.",
                "colab run is not used. Submit is new, upload, exec, download, then stop.",
            ],
        )

    def detect_cli(self) -> dict[str, object]:
        """Return whether ``colab`` is on PATH and what ``version`` prints."""
        self._require_real_mode()
        cli = self._executable()
        found = shutil.which(cli)
        if found is None:
            return {"found": False, "path": None, "version": None}
        result = self._run([cli, "version"], timeout=_STEP_TIMEOUT)
        version = None
        if result.returncode == 0:
            version = next(
                (line.strip() for line in result.stdout.splitlines() if line.strip()),
                None,
            )
        return {"found": True, "path": found, "version": version}

    def auth_status(self) -> AuthStatus:
        """Verify the configured auth source with ``colab usage``. Never assume success."""
        self._require_real_mode()
        detected = self._detector.status()
        if detected.state is AuthState.unauthenticated or detected.method is AuthMethod.none:
            return self._with_account_status(detected)
        try:
            result = self._run(self._argv("usage"), timeout=_STEP_TIMEOUT)
        except ProviderCLIError as exc:
            return self._with_account_status(
                self._auth_from_cli(detected, AuthState.unknown, exc.message)
            )
        blob = f"{result.stdout}\n{result.stderr}"
        problem = _auth_problem(blob)
        if result.returncode == 0 and problem is None:
            return self._with_account_status(
                self._auth_from_cli(
                    detected,
                    AuthState.authenticated,
                    "verified by `colab usage` exit 0",
                )
            )
        if problem is None:
            problem = AuthState.unknown
        excerpt = redact_excerpt(result.stderr or result.stdout, limit=300)
        fallback = "colab usage rejected the credential"
        status = self._auth_from_cli(detected, problem, excerpt or fallback)
        return self._with_account_status(status)

    def get_quota(self) -> QuotaInfo:
        """Parse ``colab usage`` by default; optional experimental ``ccu-info``.

        Default path is official ``colab usage`` text (balance + optional Eligible
        GPUs line). Undocumented ``ccu-info`` is fetched only when
        ``providers.colab.experimental_ccu_info`` or env
        ``L2KC_COLAB_EXPERIMENTAL_CCU_INFO`` is enabled; on failure it degrades to
        ``colab usage``. Membership in ``details`` is informational only
        (free_pro ≠ paid Pro). ``eligible_gpus`` is never invented.
        """
        self._require_real_mode()
        quota: QuotaInfo | None = None
        if self._experimental_ccu_info_enabled():
            quota = self._quota_via_ccu_info()
        if quota is None:
            result = self._run(self._argv("usage"), timeout=_STEP_TIMEOUT)
            if result.returncode != 0:
                return self._with_account_quota(
                    QuotaInfo.unknown(
                        self.name,
                        "colab usage exited non-zero",
                    )
                )
            try:
                quota = _parse_usage(result.stdout)
            except ValueError:
                return self._with_account_quota(
                    QuotaInfo.unknown(
                        self.name,
                        "colab usage could not be parsed",
                    )
                )
        enriched = self._with_account_quota(quota)
        self._persist_identity(enriched)
        return enriched

    def list_sessions(self) -> list[str]:
        """Return raw non-empty lines from ``colab sessions``."""
        self._require_real_mode()
        result = self._run(self._argv("sessions"), timeout=_STEP_TIMEOUT)
        if result.returncode != 0:
            raise ProviderCLIError(
                "colab sessions failed",
                returncode=result.returncode,
                stderr_excerpt=result.stderr or result.stdout,
                provider=self.name.value,
            )
        return [redact(line.strip()) for line in result.stdout.splitlines() if line.strip()]

    def list_resources(self) -> list[ResourceSpec]:
        """Static accelerator list. Eligibility is only known after ``colab new``."""
        self._require_real_mode()
        return [
            ResourceSpec(
                gpu_type=GpuType.T4,
                gpu_count=1,
                gpu_memory_gb_per_gpu=16,
                provider_resource_id="T4",
                premium=False,
                supports_high_mem=True,
                label=f"T4 (16 GB nominal per GPU; --high-mem ok); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.L4,
                gpu_count=1,
                gpu_memory_gb_per_gpu=24,
                provider_resource_id="L4",
                premium=False,
                supports_high_mem=False,
                label=f"L4 (24 GB per GPU; high-mem N/A); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.A100,
                gpu_count=1,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="A100",
                premium=True,
                supports_high_mem=True,
                label=(
                    "A100 (per-GPU memory unknown: 40 GB or 80 GB, not selectable; "
                    f"--high-mem ok); {_ELIGIBILITY}"
                ),
            ),
            ResourceSpec(
                gpu_type=GpuType.H100,
                gpu_count=1,
                gpu_memory_gb_per_gpu=80,
                provider_resource_id="H100",
                premium=True,
                supports_high_mem=True,
                label=f"H100 (80 GB per GPU, premium; --high-mem ok); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.G4,
                gpu_count=1,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="G4",
                premium=True,
                supports_high_mem=True,
                label=f"G4 (per-GPU memory unknown, premium; --high-mem ok); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.TPU_V5E1,
                gpu_count=1,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="TPU_V5E1",
                premium=True,
                supports_high_mem=False,
                label=f"TPU v5e1 (colab new --tpu v5e1; high-mem N/A); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.TPU_V6E1,
                gpu_count=1,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="TPU_V6E1",
                premium=True,
                supports_high_mem=False,
                label=f"TPU v6e1 (colab new --tpu v6e1; high-mem N/A); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.NONE,
                gpu_count=0,
                provider_resource_id="cpu",
                premium=False,
                supports_high_mem=True,
                label=f"CPU (--high-mem ok); {_ELIGIBILITY}",
            ),
        ]

    def submit(self, spec: JobSpec) -> JobHandle:
        """Create a session, run the script, pull ``outputs/``, then stop the runtime.

        ``colab stop`` is tried twice. If it still fails, the job keeps its
        execution state and records ``runtime_stop_failed``.
        """
        self._settings.require_real_jobs_allowed(self.name.value)
        if spec.args:
            raise NotSupportedError(
                "colab exec does not forward args; V0.1",
                provider=self.name.value,
            )
        gpu_flag, tpu_flag, high_mem = self._accelerator_flags(spec)
        uploads = self._checked_inputs(spec)
        job_id = uuid4().hex
        session = validate_session_name(f"l2kc-{job_id[:12]}")
        script_arg = absolute_cli_path(spec.script_path)
        job_dir = safe_join(self._root(), job_id)
        outputs = safe_join(job_dir, "outputs")
        log_path = safe_join(job_dir, "log.txt")
        job_dir.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        submitted_at = datetime.now(UTC)
        state = JobState.FAILED
        exit_code: int | None = None
        message = ""
        artifacts: list[str] = []
        runtime_stopped = False
        stop_returncode: int | None = None
        stop_details: dict[str, Any] = {}
        pending: Exception | None = None
        try:
            self._create_session(session, gpu_flag, tpu_flag, high_mem)
            for local, remote in uploads:
                self._upload(session, local, remote)
            exec_result = self._run(
                self._argv("exec", "-s", session, "-f", script_arg),
                timeout=self._exec_timeout(spec),
            )
            self._write_log(log_path, exec_result.stdout, exec_result.stderr)
            exit_code = exec_result.returncode
            combined = f"{exec_result.stdout}\n{exec_result.stderr}"
            if exec_result.returncode == 0:
                state = JobState.SUCCEEDED
            elif _runtime_disconnected_output(exec_result.stderr, combined):
                state = JobState.FAILED
                disconnected = RuntimeDisconnectedError(
                    "colab runtime disconnected",
                    provider=self.name.value,
                )
                message = disconnected.message
            else:
                state = JobState.FAILED
                message = redact(combined)[:500]
            artifacts = self._pull_outputs(session, outputs)
        except Exception as exc:
            pending = exc
            state = JobState.FAILED
            exit_code = None
            message = exc.message if isinstance(exc, L2KCError) else redact(str(exc))[:500]
            if not log_path.is_file():
                excerpt = getattr(exc, "stderr_excerpt", None)
                text = excerpt if isinstance(excerpt, str) and excerpt else str(exc)
                self._write_log(log_path, "", text)
        finally:
            if not self._settings.providers.colab.keep_runtime:
                outcome = self._stop_with_retry(session)
                runtime_stopped = outcome.stopped
                stop_returncode = outcome.returncode
                if not outcome.stopped:
                    message = _merge_message(message, _runtime_stop_failed_message(session))
                    stop_details["runtime_stop_failed"] = True
            finished = datetime.now(UTC) if state.is_terminal() else None
            self._save(
                ColabJobRecord(
                    job_id=job_id,
                    session=session,
                    state=state,
                    submitted_at=submitted_at,
                    exit_code=exit_code,
                    log_path=str(log_path),
                    artifacts=artifacts,
                    runtime_stopped=runtime_stopped,
                    stop_returncode=stop_returncode,
                    message=message,
                    outputs_dir=str(outputs),
                    finished_at=finished,
                    details=stop_details,
                )
            )
        if pending is not None:
            raise pending
        return JobHandle(
            job_id=job_id,
            provider=self.name,
            remote_id=session,
            submitted_at=submitted_at,
        )

    def status(self, job_id: str) -> JobResult:
        """Return the stored state. V0.1 execution is synchronous."""
        self._require_real_mode()
        return _to_result(self._load(job_id))

    def logs(self, job_id: str) -> str:
        """Return the redacted exec log captured during submit."""
        self._require_real_mode()
        record = self._load(job_id)
        if not record.log_path:
            return ""
        path = ensure_within(self._root(), Path(record.log_path))
        if path.is_symlink() or not path.is_file():
            return ""
        return redact(path.read_text(encoding="utf-8"))

    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        """Copy outputs already retrieved during submit into ``dest_dir``."""
        self._require_real_mode()
        record = self._load(job_id)
        dest = Path(dest_dir).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        if not record.outputs_dir:
            return []
        source_root = ensure_within(self._root(), Path(record.outputs_dir))
        if not source_root.is_dir():
            return []
        artifacts: list[Artifact] = []
        for path in sorted(source_root.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = ensure_within(source_root, path)
            if resolved.is_symlink() or not resolved.is_file():
                continue
            relative = resolved.relative_to(source_root)
            if any(part in {"", ".", ".."} or part.startswith("-") for part in relative.parts):
                continue
            target = ensure_within(dest, dest.joinpath(*relative.parts))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, target)
            artifacts.append(
                Artifact(
                    name=relative.as_posix(),
                    local_path=target,
                    size_bytes=target.stat().st_size,
                )
            )
        return artifacts

    def cancel(self, job_id: str) -> JobResult:
        """Stop a live session.

        A terminal job keeps that state. When its record has
        ``runtime_stopped`` false and ``keep_runtime`` is false, ``colab stop``
        is retried. Success records ``runtime_stopped`` and clears
        ``runtime_stop_failed``.
        """
        self._require_real_mode()
        record = self._load(job_id)
        keep_runtime = self._settings.providers.colab.keep_runtime
        if record.state.is_terminal():
            if record.runtime_stopped or keep_runtime:
                return _to_result(record)
            return self._stop_and_record(record, terminal=True)
        if record.runtime_stopped:
            updated = record.model_copy(
                update={
                    "state": JobState.CANCELLED,
                    "finished_at": record.finished_at or datetime.now(UTC),
                    "message": record.message or "cancelled via colab stop",
                }
            )
            self._save(updated)
            return _to_result(updated)
        return self._stop_and_record(record, terminal=False)

    def _exec_timeout(self, spec: JobSpec) -> float:
        if spec.timeout_seconds is not None:
            return float(spec.timeout_seconds)
        return float(self._settings.providers.colab.default_exec_timeout_seconds)

    def _stop_with_retry(self, session: str) -> _StopOutcome:
        """Call ``colab stop`` at least twice unless an earlier attempt works."""
        last = _StopOutcome(stopped=False)
        for attempt in range(_STOP_ATTEMPTS):
            if attempt > 0 and self._stop_backoff_seconds > 0:
                self._sleeper(self._stop_backoff_seconds)
            last = self._one_stop(session)
            if last.stopped:
                return last
        return last

    def _one_stop(self, session: str) -> _StopOutcome:
        try:
            result = self._run(self._argv("stop", "-s", session), timeout=_STEP_TIMEOUT)
        except Exception as exc:
            return _StopOutcome(stopped=False, error=exc)
        if result.returncode == 0:
            return _StopOutcome(stopped=True, returncode=result.returncode)
        return _StopOutcome(
            stopped=False,
            returncode=result.returncode,
            stderr=result.stderr or result.stdout,
        )

    def _stop_and_record(self, record: ColabJobRecord, *, terminal: bool) -> JobResult:
        session = validate_session_name(record.session)
        outcome = self._stop_with_retry(session)
        updated = _record_after_stop(record, outcome, terminal=terminal)
        self._save(updated)
        if outcome.stopped or terminal:
            return _to_result(updated)
        if outcome.error is not None:
            raise outcome.error
        raise ProviderCLIError(
            "colab stop failed",
            returncode=outcome.returncode,
            stderr_excerpt=outcome.stderr or None,
            provider=self.name.value,
        )

    def _require_real_mode(self) -> None:
        if self._settings.mode != "real":
            raise RealModeDisabledError(
                "Real mode is disabled. Set mode to 'real' before calling the Colab CLI.",
                provider=self.name.value,
                details={
                    "mode": self._settings.mode,
                    "allow_real_jobs": self._settings.allow_real_jobs,
                },
            )

    def _executable(self) -> str:
        cli = self._settings.providers.colab.cli_path
        try:
            return validate_cli_arg(cli, "cli_path")
        except InvalidInputError as exc:
            raise ConfigError(
                "colab cli_path is not safe to execute",
                provider=self.name.value,
            ) from exc

    def _argv(self, *parts: str) -> list[str]:
        auth = validate_cli_arg(self._settings.providers.colab.auth, "auth")
        return [self._executable(), "--auth", auth, *parts]

    def _detector_home(self) -> Path:
        """Home directory used for Colab token lookup."""
        return self._home

    def _experimental_ccu_info_enabled(self) -> bool:
        """True when undocumented ccu-info is explicitly opted in."""
        if self._settings.providers.colab.experimental_ccu_info:
            return True
        env = os.environ.get("L2KC_COLAB_EXPERIMENTAL_CCU_INFO", "").strip().lower()
        return env in {"1", "true", "yes", "on"}

    def _quota_via_ccu_info(self) -> QuotaInfo | None:
        """Fetch experimental ccu-info when enabled and a token is present.

        Returns None on parse/HTTP/token failure so callers can use ``colab usage``.
        """
        try:
            snap = ccu_info_snapshot(self._detector_home())
        except (ValueError, OSError, TypeError, KeyError):
            return None
        return _quota_from_ccu_info(snap)

    def _persist_identity(self, quota: QuotaInfo) -> None:
        """Write non-secret identity.json for display/cache. Best-effort."""
        if not self._account_id:
            return
        email: str | None = None
        token_path = token_json_path(self._detector_home())
        if token_path.is_file():
            _bearer, email = read_token_account(token_path)
        membership = "unknown"
        if isinstance(quota.details.get("membership"), str):
            membership = str(quota.details["membership"])
        elif quota.remaining is not None:
            membership = infer_colab_membership_tier(quota.remaining, quota.eligible_gpus)
        payload = {
            "provider": "colab",
            "email": email,
            "membership_tier": membership,
            "eligible_gpus": quota.eligible_gpus,
            "ccu_remaining": quota.remaining,
            "refreshed_at": utc_now_iso(),
            "source": quota.source,
        }
        try:
            if self._state_dir is not None:
                # Non-implicit account state_dir is already accounts/<id>/.
                path = Path(self._state_dir) / "identity.json"
            else:
                path = identity_path(Path(self._settings.state_dir), self._account_id)
            write_identity(path, payload)
        except OSError:
            return

    def _with_account_status(self, status: AuthStatus) -> AuthStatus:
        if self._account_id and status.account_id is None:
            return status.model_copy(update={"account_id": self._account_id})
        return status

    def _with_account_quota(self, quota: QuotaInfo) -> QuotaInfo:
        if self._account_id and quota.account_id is None:
            return quota.model_copy(update={"account_id": self._account_id})
        return quota

    def _run(self, args: list[str], *, timeout: float | None) -> CommandResult:
        try:
            return self._runner.run(args, timeout=timeout, env=self._account_env)
        except ProviderCLIError as exc:
            if exc.provider == self.name.value:
                raise
            raise ProviderCLIError(
                exc.message,
                code=exc.code,
                returncode=exc.returncode,
                stderr_excerpt=exc.stderr_excerpt,
                provider=self.name.value,
            ) from exc

    def _auth_from_cli(self, detected: AuthStatus, state: AuthState, message: str) -> AuthStatus:
        return AuthStatus(
            provider=self.name,
            state=state,
            method=detected.method,
            account_label=None,
            is_mock=False,
            message=message,
            details=dict(detected.details),
        )

    def _accelerator_flags(self, spec: JobSpec) -> tuple[str | None, str | None, bool]:
        """Return ``(gpu_flag, tpu_flag, high_mem)`` for ``colab new``.

        ``--gpu`` and ``--tpu`` are mutually exclusive. ``high_mem`` is rejected
        for L4/TPU (JobSpec also validates). Premium gate runs before flags.
        """
        high_mem = bool(spec.high_mem)
        # Zero GPUs is CPU even when a concrete type was attached. This runs
        # before the premium gate so a bypassed JobSpec cannot start a paid GPU.
        if spec.gpu_count <= 0:
            gpu = GpuType.NONE if spec.gpu_type is None else spec.gpu_type
            if high_mem and gpu not in _HIGH_MEM_OK and gpu is not GpuType.NONE:
                raise NotSupportedError(
                    f"colab --high-mem is not supported for {gpu.value}",
                    provider=self.name.value,
                )
            return None, None, high_mem
        gpu = spec.gpu_type
        # None is CPU only when no GPU was requested. Direct adapter use
        # (bypassing Client) defaults a positive count to T4, matching the mocks.
        if gpu is None and spec.gpu_count > 0:
            gpu = GpuType.T4
        if gpu is None or gpu is GpuType.NONE:
            return None, None, high_mem
        if gpu in _PREMIUM:
            blocked = premium_gate(self._settings.scheduler, spec.allow_premium)
            if blocked is not None:
                raise PremiumNotAllowed(
                    blocked,
                    provider=self.name.value,
                    details={"gpu_type": gpu.value},
                )
        if high_mem and gpu not in _HIGH_MEM_OK:
            raise NotSupportedError(
                f"colab --high-mem is not supported for {gpu.value}",
                provider=self.name.value,
            )
        tpu = _TPU_FLAG.get(gpu)
        if tpu is not None:
            return None, validate_cli_arg(tpu, "tpu"), False
        flag = _GPU_FLAG.get(gpu)
        if flag is None:
            raise NotSupportedError(
                f"colab CLI has no verified accelerator flag for {gpu.value}",
                provider=self.name.value,
            )
        return validate_cli_arg(flag, "gpu"), None, high_mem

    def _checked_inputs(self, spec: JobSpec) -> list[tuple[str, str]]:
        prepared: list[tuple[str, str]] = []
        seen: set[str] = set()
        for path in spec.input_files:
            if not path.is_file():
                raise InvalidInputError(f"input file does not exist: {path.name}")
            remote = validate_remote_relpath(path.name)
            if "/" in remote or "\\" in remote:
                raise InvalidInputError("input upload name must be a single path segment")
            if remote in seen:
                raise InvalidInputError(f"duplicate input name: {remote}")
            seen.add(remote)
            prepared.append((absolute_cli_path(path), remote))
        return prepared

    def _create_session(
        self,
        session: str,
        gpu_flag: str | None,
        tpu_flag: str | None = None,
        high_mem: bool = False,
    ) -> None:
        args = self._argv("new", "-s", session)
        if tpu_flag is not None:
            args.extend(["--tpu", tpu_flag])
        elif gpu_flag is not None:
            args.extend(["--gpu", gpu_flag])
        if high_mem:
            args.append("--high-mem")
        result = self._run(args, timeout=_STEP_TIMEOUT)
        combined = f"{result.stdout}\n{result.stderr}"
        lowered = combined.lower()
        if result.returncode == 0 and "error" not in lowered:
            return
        excerpt = redact_excerpt(combined)
        if "unavailable" in lowered or "quota" in lowered or "eligib" in lowered:
            raise ResourceUnavailableError(
                excerpt or "colab runtime is unavailable",
                provider=self.name.value,
            )
        raise ProviderCLIError(
            "colab new failed",
            returncode=result.returncode,
            stderr_excerpt=excerpt,
            provider=self.name.value,
        )

    def _upload(self, session: str, local: str, remote: str) -> None:
        result = self._run(
            self._argv("upload", "-s", session, local, remote),
            timeout=_STEP_TIMEOUT,
        )
        if result.returncode != 0:
            raise ProviderCLIError(
                "colab upload failed",
                returncode=result.returncode,
                stderr_excerpt=result.stderr or result.stdout,
                provider=self.name.value,
            )

    def _pull_outputs(self, session: str, outputs: Path) -> list[str]:
        listed = self._run(self._argv("ls", "-s", session, "outputs"), timeout=_STEP_TIMEOUT)
        if listed.returncode != 0:
            return []
        saved: list[str] = []
        for raw in listed.stdout.splitlines():
            name = _plain_filename(raw.strip())
            if name is None:
                continue
            local_path = ensure_within(outputs, outputs / name)
            remote = validate_remote_relpath(f"outputs/{name}")
            fetched: CommandResult | None
            try:
                fetched = self._run(
                    self._argv("download", "-s", session, remote, absolute_cli_path(local_path)),
                    timeout=_STEP_TIMEOUT,
                )
            except ProviderCLIError:
                fetched = None
            if fetched is None or fetched.returncode != 0:
                continue
            if local_path.is_symlink() or not local_path.is_file():
                continue
            saved.append(name)
        return saved

    def _write_log(self, path: Path, stdout: str, stderr: str) -> None:
        text = stdout
        if stderr:
            if text and not text.endswith("\n"):
                text += "\n"
            text += stderr
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact(text), encoding="utf-8")

    def _root(self) -> Path:
        if self._state_dir is not None:
            base = self._state_dir.expanduser().resolve()
        else:
            base = Path(self._settings.state_dir).expanduser().resolve()
        return base / "colab"

    def _record_path(self, job_id: str) -> Path:
        if _JOB_ID.fullmatch(job_id) is None:
            raise JobNotFoundError("unknown job id", provider=self.name.value)
        return safe_join(self._root() / "jobs", f"{job_id}.json")

    def _save(self, record: ColabJobRecord) -> None:
        path = self._record_path(record.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def _load(self, job_id: str) -> ColabJobRecord:
        path = self._record_path(job_id)
        if not path.is_file():
            raise JobNotFoundError("unknown job id", provider=self.name.value)
        try:
            return ColabJobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise JobNotFoundError(
                "job record could not be read",
                provider=self.name.value,
            ) from exc
