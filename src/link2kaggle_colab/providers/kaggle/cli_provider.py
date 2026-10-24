"""Kaggle adapter that shells out to the official ``kaggle`` CLI.

Argv values come from docs/api-verification.md. This module does not call
``kernels delete`` and does not pretend the CLI can cancel a run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from link2kaggle_colab.accounts.identity import identity_path, utc_now_iso, write_identity
from link2kaggle_colab.auth.real import KaggleCredentialDetector
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.envspec import AccountEnv, apply_account_env
from link2kaggle_colab.core.errors import (
    ConfigError,
    InvalidInputError,
    JobNotFoundError,
    NotSupportedError,
    PremiumNotAllowed,
    ProviderCLIError,
    QuotaExceededError,
    RealModeDisabledError,
)
from link2kaggle_colab.core.models import (
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
    validate_kernel_ref,
    validate_slug,
)
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.scheduler.policy import premium_gate

_JOB_ID = re.compile(r"^[a-f0-9]{32}$")
# A100 and H100 match the premium set Colab and the scheduler gate. G4 is Colab-only.
_PREMIUM = frozenset({GpuType.A100, GpuType.H100})
_READ_TIMEOUT = 60.0
_STATUS_LINE = re.compile(r'has status "([^"]+)"', re.IGNORECASE)
_FAILURE_LINE = re.compile(r'Failure message:\s*"([^"]*)"', re.IGNORECASE)
_ELIGIBILITY = "eligibility needs real verification"

_STATUS_MAP = {
    "QUEUED": JobState.QUEUED,
    "NEW_SCRIPT": JobState.QUEUED,
    "RUNNING": JobState.RUNNING,
    "COMPLETE": JobState.SUCCEEDED,
    "ERROR": JobState.FAILED,
    "CANCEL_REQUESTED": JobState.CANCELLING,
    "CANCEL_ACKNOWLEDGED": JobState.CANCELLED,
}


class KaggleJobRecord(BaseModel):
    """On-disk Kaggle job. Credential material does not belong here."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    kernel_ref: str
    state: JobState
    submitted_at: datetime
    message: str = ""
    failure_message: str = ""
    finished_at: datetime | None = None


def _kaggle_machine_shape(gpu: GpuType, gpu_count: int) -> str:
    """Map a request onto a verified accelerator id.

    ``NvidiaTeslaT4`` is the T4 x2 shape from the CLI docs. ``JobSpec`` defaults
    ``gpu_count`` to 1; Kaggle still only offers that T4 id, so every T4 request
    uses it. ``NvidiaL4`` versus ``NvidiaL4X1`` follows the requested count.
    """
    if gpu is GpuType.T4:
        return "NvidiaTeslaT4"
    if gpu is GpuType.L4:
        if gpu_count <= 1:
            return "NvidiaL4X1"
        return "NvidiaL4"
    if gpu is GpuType.A100:
        return "NvidiaTeslaA100"
    if gpu is GpuType.H100:
        return "NvidiaH100"
    raise NotSupportedError(
        f"Kaggle machine_shape for {gpu.value} is not a verified accelerator id",
        provider=ProviderName.kaggle.value,
    )


def _parse_amount(value: object) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError("quota amount is missing")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        for suffix in ("hours", "hour", "h"):
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                break
        return float(text)
    raise ValueError("quota amount is not a number")


def _parse_refresh(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("refreshAt is not an ISO timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _quota_rows(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if any(key in payload for key in ("resource", "used", "remaining", "total")):
            return [payload]
        rows: list[dict[str, object]] = []
        for value in payload.values():
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict) and "resource" in value:
                rows.append(value)
        return rows
    raise ValueError("unsupported quota payload")


def _parse_kaggle_quota(stdout: str) -> QuotaInfo:
    payload = json.loads(stdout)
    rows = _quota_rows(payload)
    gpu = next(
        (row for row in rows if str(row.get("resource", "")).strip().upper() == "GPU"),
        None,
    )
    if gpu is None:
        raise ValueError("GPU row missing")
    refresh_raw = gpu.get("refreshAt", gpu.get("refresh_at"))
    return QuotaInfo(
        provider=ProviderName.kaggle,
        unit=QuotaUnit.gpu_hours,
        total=_parse_amount(gpu["total"]),
        used=_parse_amount(gpu["used"]),
        remaining=_parse_amount(gpu["remaining"]),
        refresh_at=_parse_refresh(refresh_raw),
        is_fixture=False,
        source="kaggle quota",
        message="",
    )


def _auth_problem(text: str) -> AuthState | None:
    lowered = text.lower()
    if "401" in lowered or "unauthenticated" in lowered:
        return AuthState.invalid
    if "403" in lowered or "permission" in lowered:
        return AuthState.error
    return None


def parse_kernel_status(text: str) -> tuple[JobState, str]:
    """Map ``kernels status`` text. Unknown or unparseable text is UNKNOWN."""
    failure = ""
    found = _FAILURE_LINE.search(text)
    if found is not None:
        failure = redact(found.group(1))
    match = _STATUS_LINE.search(text)
    if match is None:
        return JobState.UNKNOWN, "unrecognized kernel status"
    token = match.group(1).strip()
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    state = _STATUS_MAP.get(token.upper(), JobState.UNKNOWN)
    if state is JobState.FAILED:
        return state, failure or "kernel worker status ERROR"
    if state is JobState.UNKNOWN:
        return state, "unrecognized kernel status"
    return state, ""


_SLUG_PREFIX = "l2kc-"
_PUSH_OK = "successfully pushed"


def _kernel_slug(name: str | None, job_id: str) -> str:
    """Return a validated slug that starts with ``l2kc-``.

    A caller-supplied name is prefixed unless it already starts with ``l2kc-``,
    so a push cannot overwrite a kernel this tool did not create. The name and
    the final slug both go through :func:`validate_slug`.
    """
    if name is None:
        return validate_slug(f"{_SLUG_PREFIX}{job_id[:12]}")
    checked = validate_slug(name)
    if checked.startswith(_SLUG_PREFIX):
        slug = checked
    else:
        slug = f"{_SLUG_PREFIX}{checked}"
    return validate_slug(slug)


def _push_verdict(result: CommandResult) -> str:
    """Classify ``kernels push`` output as accepted, rejected, or ambiguous.

    The official success line contains ``successfully pushed`` and exits 0.
    A non-zero exit, or the CLI's exit-0 ``Kernel push error`` text, is a
    clear rejection when that success line is absent. A mix of the two, or
    text that confirms neither, is ambiguous: the kernel may already be running.
    """
    text = f"{result.stdout}\n{result.stderr}".lower()
    acknowledged = _PUSH_OK in text
    rejected = result.returncode != 0 or "error" in text
    if acknowledged and not rejected:
        return "accepted"
    if rejected and not acknowledged:
        return "rejected"
    return "ambiguous"


def _unknown_record(record: KaggleJobRecord, reason: str) -> KaggleJobRecord:
    """Mark a push whose outcome cannot be confirmed. The kernel may be running."""
    message = (
        f"{reason}; kernel {record.kernel_ref} may be running. "
        "Check status before submitting again."
    )
    return record.model_copy(
        update={
            "state": JobState.UNKNOWN,
            "message": message,
            "failure_message": "",
        }
    )


def _failed_record(record: KaggleJobRecord, message: str) -> KaggleJobRecord:
    return record.model_copy(
        update={
            "state": JobState.FAILED,
            "message": message,
            "failure_message": message,
            "finished_at": datetime.now(UTC),
        }
    )


def _to_result(record: KaggleJobRecord) -> JobResult:
    return JobResult(
        job_id=record.job_id,
        provider=ProviderName.kaggle,
        state=record.state,
        remote_id=record.kernel_ref,
        message=record.message,
        finished_at=record.finished_at,
        is_mock=False,
    )


def _artifacts_under(dest: Path) -> list[Artifact]:
    root = dest.resolve()
    found: list[Artifact] = []
    if not root.is_dir():
        return found
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        resolved = ensure_within(root, path)
        if resolved.is_symlink() or not resolved.is_file():
            continue
        relative = resolved.relative_to(root)
        found.append(
            Artifact(
                name=relative.as_posix(),
                local_path=resolved,
                size_bytes=resolved.stat().st_size,
            )
        )
    return found


class KaggleCliProvider(Provider):
    """Official Kaggle CLI. Read-only calls need ``mode: real``.

    ``kernels push`` also needs ``allow_real_jobs``. Cancel is not supported:
    the CLI has no cancel command, and ``kernels delete`` is not a cancel.
    """

    name = ProviderName.kaggle
    is_mock = False

    def __init__(
        self,
        settings: Settings,
        *,
        runner: CommandRunner | None = None,
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        account_env: AccountEnv | None = None,
        state_dir: Path | None = None,
        account_id: str | None = None,
    ) -> None:
        self._settings = settings
        self._runner = runner if runner is not None else SubprocessRunner()
        self._account_env = account_env
        self._state_dir = None if state_dir is None else Path(state_dir)
        self._account_id = account_id
        # An explicit home or environ wins. Otherwise the detector sees a copy
        # of the process environment with this account's overlay applied.
        detector_environ = environ
        detector_home = home
        if account_env is not None:
            if detector_environ is None:
                detector_environ = apply_account_env(account_env, os.environ)
            if detector_home is None and "HOME" in account_env.set:
                detector_home = Path(account_env.set["HOME"])
        self._detector = KaggleCredentialDetector(home=detector_home, environ=detector_environ)

    def capabilities(self) -> Capability:
        return Capability(
            provider=self.name,
            supports_submit=True,
            supports_status=True,
            supports_logs=True,
            supports_cancel=False,
            supports_download=True,
            supports_quota=True,
            supports_interactive_session=False,
            is_batch_only=True,
            notes=[
                "Kaggle is a batch runner. It is not an interactive shell.",
                "The official CLI has no kernel cancel. "
                "kagglesdk cancel_kernel_session is unverified and not wired.",
                "kernels delete removes the kernel and is not a cancel.",
                "kernels status accepts KernelWorkerStatus.* and bare tokens; "
                "anything else is UNKNOWN.",
                "Kernel slugs are prefixed with l2kc- unless the name already "
                "has that prefix.",
                "kernels push persists a SUBMITTING record before the CLI call. "
                "A timeout or ambiguous result is stored as UNKNOWN and still "
                "returns a handle.",
                "GPU quota JSON that cannot be parsed is UNKNOWN, not a guessed balance.",
                "NvidiaTeslaT4 is T4 x2 with 16 GB per GPU. "
                "Other accelerator eligibility needs real verification.",
                "Script arguments and local input files are not supported in V0.1. "
                "Use Kaggle datasets (dataset_sources).",
            ],
        )

    def detect_cli(self) -> dict[str, object]:
        """Return whether ``kaggle`` is on PATH and what ``--version`` prints."""
        self._require_real_mode()
        cli = self._executable()
        found = shutil.which(cli)
        if found is None:
            return {"found": False, "path": None, "version": None}
        result = self._run([cli, "--version"], timeout=_READ_TIMEOUT)
        version = None
        if result.returncode == 0:
            version = next(
                (line.strip() for line in result.stdout.splitlines() if line.strip()),
                None,
            )
        return {"found": True, "path": found, "version": version}

    def auth_status(self) -> AuthStatus:
        """Verify a detected credential with ``kaggle quota``. Never assume success."""
        self._require_real_mode()
        detected = self._detector.status()
        if detected.state is AuthState.unauthenticated or detected.method is AuthMethod.none:
            return self._with_account_status(detected)
        try:
            result = self._run(self._argv("quota", "--format", "json"), timeout=_READ_TIMEOUT)
        except ProviderCLIError as exc:
            return self._with_account_status(
                self._auth_from_cli(detected, AuthState.unknown, exc.message)
            )
        blob = f"{result.stdout}\n{result.stderr}"
        problem = _auth_problem(blob)
        if result.returncode == 0 and problem is None:
            username = self._settings.providers.kaggle.username
            return self._with_account_status(
                self._auth_from_cli(
                    detected,
                    AuthState.authenticated,
                    "verified by `kaggle quota` exit 0",
                    account_label=username or None,
                )
            )
        if problem is None:
            problem = AuthState.unknown
        excerpt = redact_excerpt(result.stderr or result.stdout, limit=300)
        fallback = "kaggle quota rejected the credential"
        status = self._auth_from_cli(detected, problem, excerpt or fallback)
        return self._with_account_status(status)

    def get_quota(self) -> QuotaInfo:
        """Parse ``kaggle quota --format json``. Failures are unknown, not fixtures."""
        self._require_real_mode()
        result = self._run(self._argv("quota", "--format", "json"), timeout=_READ_TIMEOUT)
        if result.returncode != 0:
            return self._with_account_quota(
                QuotaInfo.unknown(self.name, "kaggle quota exited non-zero")
            )
        try:
            quota = _parse_kaggle_quota(result.stdout)
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            return self._with_account_quota(
                QuotaInfo.unknown(self.name, "kaggle quota output could not be parsed")
            )
        enriched = self._with_account_quota(quota)
        self._persist_identity(enriched)
        return enriched

    def list_resources(self) -> list[ResourceSpec]:
        """Static accelerator list from the CLI docs. Eligibility is unverified."""
        self._require_real_mode()
        return [
            ResourceSpec(
                gpu_type=GpuType.T4,
                gpu_count=2,
                gpu_memory_gb_per_gpu=16,
                provider_resource_id="NvidiaTeslaT4",
                premium=False,
                label=f"NvidiaTeslaT4 (T4 x2, 16 GB per GPU); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.L4,
                gpu_count=1,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="NvidiaL4X1",
                premium=False,
                label=f"NvidiaL4X1 (L4 x1, unverified); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.L4,
                gpu_count=1,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="NvidiaL4",
                premium=False,
                # The CLI lists NvidiaL4 separately from NvidiaL4X1 and does not
                # say how many GPUs that id is. The label records that the count
                # is unverified; the schema still requires an integer.
                label=f"L4 (count unverified); {_ELIGIBILITY}",
            ),
            ResourceSpec(
                gpu_type=GpuType.NONE,
                gpu_count=0,
                provider_resource_id="cpu",
                premium=False,
                label=f"CPU; {_ELIGIBILITY}",
            ),
        ]

    def submit(self, spec: JobSpec) -> JobHandle:
        """Push a private kernel and return a handle. The kernel id is ``remote_id``."""
        self._settings.require_real_jobs_allowed(self.name.value)
        username = self._require_username()
        if spec.args:
            raise NotSupportedError(
                "Kaggle kernels take no CLI args",
                provider=self.name.value,
            )
        if spec.input_files:
            raise NotSupportedError(
                "use Kaggle datasets (dataset_sources) — not supported in V0.1",
                provider=self.name.value,
            )
        job_id = uuid4().hex
        slug = _kernel_slug(spec.name, job_id)
        kernel_ref = validate_kernel_ref(f"{username}/{slug}")
        gpu = spec.gpu_type
        # None is CPU only when no GPU was requested. Direct adapter use
        # (bypassing Client) defaults a positive count to T4, matching the mocks.
        if gpu is None and spec.gpu_count > 0:
            gpu = GpuType.T4
        if gpu is not None and gpu is not GpuType.NONE:
            # Before machine shape and kernels push. Client/scheduler already
            # gate these; a direct adapter call must not skip premium_gate.
            if gpu in _PREMIUM:
                blocked = premium_gate(self._settings.scheduler, spec.allow_premium)
                if blocked is not None:
                    raise PremiumNotAllowed(
                        blocked,
                        provider=self.name.value,
                        details={"gpu_type": gpu.value},
                    )
            machine_shape = validate_cli_arg(
                _kaggle_machine_shape(gpu, spec.gpu_count),
                "accelerator",
            )
            enable_gpu = True
        else:
            machine_shape = None
            enable_gpu = False
        kernel_type = "notebook" if spec.script_path.suffix.lower() == ".ipynb" else "script"
        staging = safe_join(self._root(), job_id)
        staging.mkdir(parents=True, exist_ok=True)
        script_name = spec.script_path.name
        script_target = safe_join(staging, script_name)
        shutil.copyfile(spec.script_path, script_target)
        metadata = {
            "id": kernel_ref,
            "title": slug,
            "code_file": script_name,
            "language": "python",
            "kernel_type": kernel_type,
            # is_private is always true and is not configurable.
            "is_private": True,
            "enable_gpu": enable_gpu,
            "enable_internet": False,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        if machine_shape is not None:
            metadata["machine_shape"] = machine_shape
        meta_path = safe_join(staging, "kernel-metadata.json")
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        staging_arg = absolute_cli_path(staging)
        argv = self._argv("kernels", "push", "-p", staging_arg)
        if machine_shape is not None:
            argv.extend(["--accelerator", machine_shape])
        if spec.timeout_seconds is not None:
            argv.extend(["-t", validate_cli_arg(str(spec.timeout_seconds), "timeout")])
        submitted_at = datetime.now(UTC)
        record = KaggleJobRecord(
            job_id=job_id,
            kernel_ref=kernel_ref,
            state=JobState.SUBMITTING,
            submitted_at=submitted_at,
        )
        # Persist before push so a timeout cannot leave a running kernel untracked.
        self._save(record)
        handle = JobHandle(
            job_id=job_id,
            provider=self.name,
            remote_id=kernel_ref,
            submitted_at=submitted_at,
        )
        try:
            result = self._run(argv, timeout=self._push_subprocess_timeout(spec))
        except ProviderCLIError as exc:
            if exc.code == "cli_timeout":
                self._save(_unknown_record(record, "kaggle kernels push timed out"))
                return handle
            self._save(_failed_record(record, exc.message))
            raise
        except TimeoutError:
            self._save(_unknown_record(record, "kaggle kernels push timed out"))
            return handle
        verdict = _push_verdict(result)
        if verdict == "accepted":
            self._save(record.model_copy(update={"state": JobState.QUEUED}))
            return handle
        if verdict == "ambiguous":
            self._save(_unknown_record(record, "kaggle kernels push output was ambiguous"))
            return handle
        self._reject_push(record, result)

    def status(self, job_id: str) -> JobResult:
        """Refresh state from ``kaggle kernels status``. Unparseable text is UNKNOWN."""
        self._require_real_mode()
        record = self._load(job_id)
        kernel_ref = validate_kernel_ref(record.kernel_ref)
        result = self._run(
            self._argv("kernels", "status", kernel_ref),
            timeout=_READ_TIMEOUT,
        )
        state, message = parse_kernel_status(f"{result.stdout}\n{result.stderr}")
        if record.state.is_terminal() and record.finished_at is not None:
            finished = record.finished_at
        elif state.is_terminal():
            finished = datetime.now(UTC)
        else:
            finished = None
        updated = record.model_copy(
            update={
                "state": state,
                "message": message,
                "failure_message": message if state is JobState.FAILED else "",
                "finished_at": finished,
            }
        )
        self._save(updated)
        return _to_result(updated)

    def logs(self, job_id: str) -> str:
        """Return redacted ``kaggle kernels logs`` stdout."""
        self._require_real_mode()
        record = self._load(job_id)
        kernel_ref = validate_kernel_ref(record.kernel_ref)
        result = self._run(self._argv("kernels", "logs", kernel_ref), timeout=_READ_TIMEOUT)
        if result.returncode != 0:
            raise ProviderCLIError(
                "kaggle kernels logs failed",
                returncode=result.returncode,
                stderr_excerpt=result.stderr or result.stdout,
                provider=self.name.value,
            )
        return redact(result.stdout)

    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        """Download kernel output into ``dest_dir`` and skip symlinks."""
        self._require_real_mode()
        record = self._load(job_id)
        kernel_ref = validate_kernel_ref(record.kernel_ref)
        dest = Path(dest_dir).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        dest_arg = absolute_cli_path(dest)
        result = self._run(
            self._argv("kernels", "output", kernel_ref, "-p", dest_arg),
            timeout=float(self._settings.providers.kaggle.download_timeout_seconds),
        )
        if result.returncode != 0:
            raise ProviderCLIError(
                "kaggle kernels output failed",
                returncode=result.returncode,
                stderr_excerpt=result.stderr or result.stdout,
                provider=self.name.value,
            )
        return _artifacts_under(dest)

    def cancel(self, job_id: str) -> JobResult:
        """Refuse cancel. ``kernels delete`` is never invoked."""
        self._require_real_mode()
        del job_id
        raise NotSupportedError(
            "Kaggle CLI has no cancel command. "
            "kagglesdk cancel_kernel_session exists but is unverified and not wired. "
            "kernels delete is not a cancel and is never called.",
            provider=self.name.value,
        )

    def _require_real_mode(self) -> None:
        if self._settings.mode != "real":
            raise RealModeDisabledError(
                "Real mode is disabled. Set mode to 'real' before calling the Kaggle CLI.",
                provider=self.name.value,
                details={
                    "mode": self._settings.mode,
                    "allow_real_jobs": self._settings.allow_real_jobs,
                },
            )


    def _username_from_kaggle_json(self) -> str | None:
        """Read ``username`` from the active credential ``kaggle.json`` when present."""
        roots: list[Path] = []
        if self._account_env is not None and "KAGGLE_CONFIG_DIR" in self._account_env.set:
            roots.append(Path(self._account_env.set["KAGGLE_CONFIG_DIR"]))
        env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        if env_dir:
            roots.append(Path(env_dir))
        # Detector environ may already include the account overlay.
        detector_env = getattr(self._detector, "_environ", None)
        if isinstance(detector_env, Mapping):
            overlay = detector_env.get("KAGGLE_CONFIG_DIR")
            if overlay:
                roots.append(Path(overlay))
        seen: set[Path] = set()
        for root in roots:
            resolved = root.expanduser()
            if resolved in seen:
                continue
            seen.add(resolved)
            path = resolved / "kaggle.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue
            value = data.get("username")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _persist_identity(self, quota: QuotaInfo) -> None:
        """Write non-secret Kaggle identity.json. Does not invent phone-verify."""
        if not self._account_id:
            return
        username = self._settings.providers.kaggle.username or self._username_from_kaggle_json()
        payload = {
            "provider": "kaggle",
            "username": username,
            "gpu_hours_total": quota.total,
            "gpu_hours_remaining": quota.remaining,
            "refreshed_at": utc_now_iso(),
            "source": quota.source,
        }
        try:
            if self._state_dir is not None:
                path = Path(self._state_dir) / "identity.json"
            else:
                path = identity_path(Path(self._settings.state_dir), self._account_id)
            write_identity(path, payload)
        except OSError:
            return

    def _require_username(self) -> str:
        username = self._settings.providers.kaggle.username or self._username_from_kaggle_json()
        if not username:
            raise ConfigError(
                "kaggle username is required to push a kernel "
                "(set providers.kaggle.username or put it in kaggle.json)",
                provider=self.name.value,
            )
        try:
            return validate_slug(username)
        except InvalidInputError as exc:
            raise ConfigError(
                "kaggle username must be a lowercase slug",
                provider=self.name.value,
            ) from exc

    def _executable(self) -> str:
        cli = self._settings.providers.kaggle.cli_path
        try:
            return validate_cli_arg(cli, "cli_path")
        except InvalidInputError as exc:
            raise ConfigError(
                "kaggle cli_path is not safe to execute",
                provider=self.name.value,
            ) from exc

    def _argv(self, *parts: str) -> list[str]:
        return [self._executable(), *parts]

    def _push_subprocess_timeout(self, spec: JobSpec) -> float:
        """How long to wait for ``kernels push`` itself.

        ``kernels push`` uploads and returns; it does not wait for the kernel
        run. ``timeout_seconds`` is the kernel run limit passed as ``-t`` and
        does not extend this wait. The subprocess is always bounded by
        ``kaggle.default_push_timeout_seconds``.
        """
        del spec
        return float(self._settings.providers.kaggle.default_push_timeout_seconds)

    def _reject_push(self, record: KaggleJobRecord, result: CommandResult) -> NoReturn:
        """Record a clear push rejection as FAILED, then raise the existing error."""
        try:
            self._raise_if_push_failed(result)
        except (QuotaExceededError, ProviderCLIError) as exc:
            self._save(_failed_record(record, exc.message))
            raise
        self._save(_failed_record(record, "kaggle kernels push failed"))
        raise ProviderCLIError(
            "kaggle kernels push failed",
            returncode=result.returncode,
            stderr_excerpt=redact_excerpt(f"{result.stdout}\n{result.stderr}"),
            provider=self.name.value,
        )

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

    def _auth_from_cli(
        self,
        detected: AuthStatus,
        state: AuthState,
        message: str,
        *,
        account_label: str | None = None,
    ) -> AuthStatus:
        details = dict(detected.details)
        if state is AuthState.error:
            details["reason"] = "permission_denied"
        return AuthStatus(
            provider=self.name,
            state=state,
            method=detected.method,
            account_label=account_label,
            is_mock=False,
            message=message,
            details=details,
        )

    def _raise_if_push_failed(self, result: CommandResult) -> None:
        combined = f"{result.stdout}\n{result.stderr}"
        lowered = combined.lower()
        if result.returncode == 0 and "error" not in lowered:
            return
        excerpt = redact_excerpt(combined)
        if "quota" in lowered:
            raise QuotaExceededError(
                excerpt or "kaggle kernel push was rejected for quota",
                provider=self.name.value,
            )
        raise ProviderCLIError(
            "kaggle kernels push failed",
            returncode=result.returncode,
            stderr_excerpt=excerpt,
            provider=self.name.value,
        )

    def _root(self) -> Path:
        if self._state_dir is not None:
            base = self._state_dir.expanduser().resolve()
        else:
            base = Path(self._settings.state_dir).expanduser().resolve()
        return base / "kaggle"

    def _record_path(self, job_id: str) -> Path:
        if _JOB_ID.fullmatch(job_id) is None:
            raise JobNotFoundError("unknown job id", provider=self.name.value)
        return safe_join(self._root() / "jobs", f"{job_id}.json")

    def _save(self, record: KaggleJobRecord) -> None:
        path = self._record_path(record.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def _load(self, job_id: str) -> KaggleJobRecord:
        path = self._record_path(job_id)
        if not path.is_file():
            raise JobNotFoundError("unknown job id", provider=self.name.value)
        try:
            return KaggleJobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise JobNotFoundError(
                "job record could not be read",
                provider=self.name.value,
            ) from exc
