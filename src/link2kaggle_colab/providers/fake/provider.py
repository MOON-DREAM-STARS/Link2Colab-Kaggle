"""Offline executor that runs a local ``.py`` script. It never calls the network."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import (
    InvalidInputError,
    NotSupportedError,
    PathSecurityError,
    ResourceUnavailableError,
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
from link2kaggle_colab.core.paths import safe_join
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.providers.base import Provider
from link2kaggle_colab.providers.mock_runtime import JobStore, StoredJob, to_result

FAKE_ACCOUNT = "mock-fake-user (TEST FIXTURE)"


def _copy_outputs(outputs: Path, dest_dir: Path) -> list[Artifact]:
    """Copy ``outputs`` into ``dest_dir``. Symlinks that leave ``outputs`` are refused."""
    root = Path(dest_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not outputs.is_dir():
        return []
    output_root = outputs.resolve()
    planned: list[tuple[Path, Path]] = []
    for dirpath, dirnames, filenames in os.walk(outputs, followlinks=False):
        current = Path(dirpath)
        kept: list[str] = []
        for dirname in dirnames:
            child = current / dirname
            if child.is_symlink():
                _require_inside(output_root, child)
                continue
            kept.append(dirname)
        dirnames[:] = kept
        for filename in filenames:
            source = current / filename
            if source.is_symlink():
                _require_inside(output_root, source)
            if not source.is_file():
                continue
            relative = source.relative_to(outputs)
            planned.append((source, safe_join(root, *relative.parts)))
    artifacts: list[Artifact] = []
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        artifacts.append(
            Artifact(
                name=destination.relative_to(root).as_posix(),
                local_path=destination,
                size_bytes=destination.stat().st_size,
            )
        )
    return artifacts


def _require_inside(root: Path, path: Path) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise PathSecurityError(
            "refusing a symlink that escapes the job outputs directory",
            provider=ProviderName.fake.value,
        ) from exc


def _captured_text(value: str | bytes | None) -> str:
    """Return process output as text.

    ``TimeoutExpired.stdout`` and ``stderr`` stay bytes on POSIX even when
    ``subprocess.run`` was called with ``text=True``.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_script(
    command: list[str],
    workdir: Path,
    timeout: int | None,
) -> subprocess.CompletedProcess[str]:
    # List arguments and shell=False. The executable is this interpreter, not a shell.
    return subprocess.run(  # noqa: S603
        command,
        cwd=workdir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
        shell=False,
    )


class FakeProvider(Provider):
    """Runs the user's script locally under ``state_dir/fake/<job_id>``.

    Notebooks are refused. GPU requests are refused. Only files under
    ``outputs/`` are downloadable. Job metadata is ``job.json`` so another
    process can read status and logs.
    """

    name = ProviderName.fake
    is_mock = True

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        state_dir: Path | None = None,
        account_id: str | None = None,
    ) -> None:
        self._settings = settings if settings is not None else Settings()
        self._state_dir = Path(state_dir) if state_dir is not None else self._settings.state_dir
        self._account_id = account_id
        self._jobs = JobStore(self._state_dir / "fake", nested=True)

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
                "Offline local executor. No GPU and no network.",
                "Quota is a TEST FIXTURE with an unknown unit.",
            ],
        )

    def auth_status(self) -> AuthStatus:
        return AuthStatus(
            provider=self.name,
            state=AuthState.authenticated,
            method=AuthMethod.mock,
            account_label=FAKE_ACCOUNT,
            account_id=self._account_id,
            is_mock=True,
            message="TEST FIXTURE: fake provider does not authenticate to a cloud account",
            details={"fixture": True},
        )

    def get_quota(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.name,
            unit=QuotaUnit.unknown,
            account_id=self._account_id,
            is_fixture=True,
            source="mock-fixture",
            message="TEST FIXTURE: fake provider does not track quota",
        )

    def list_resources(self) -> list[ResourceSpec]:
        return [
            ResourceSpec(
                gpu_type=GpuType.NONE,
                gpu_count=0,
                gpu_memory_gb_per_gpu=None,
                provider_resource_id="cpu",
                premium=False,
                label="CPU only (fake)",
            )
        ]

    def submit(self, spec: JobSpec) -> JobHandle:
        if not self._settings.providers.fake.allow_exec:
            raise InvalidInputError(
                "fake local exec is disabled (providers.fake.allow_exec is false)",
                provider=self.name.value,
            )
        if spec.script_path.suffix.lower() == ".ipynb":
            raise NotSupportedError(
                "FakeProvider cannot execute notebooks",
                provider=self.name.value,
            )
        if spec.gpu_type not in (None, GpuType.NONE):
            raise ResourceUnavailableError(
                "fake provider has no GPU",
                provider=self.name.value,
            )
        seen = {spec.script_path.name}
        for source in spec.input_files:
            if not source.is_file():
                raise InvalidInputError(
                    f"input file does not exist: {source.name}",
                    provider=self.name.value,
                )
            if source.name in seen:
                raise InvalidInputError(
                    "input file collides with the script name",
                    provider=self.name.value,
                )
            seen.add(source.name)
        job_id = uuid4().hex
        workdir = safe_join(self._jobs.root, job_id)
        workdir.mkdir(parents=True, exist_ok=False)
        script_dest = safe_join(workdir, spec.script_path.name)
        shutil.copyfile(spec.script_path, script_dest)
        for source in spec.input_files:
            shutil.copyfile(source, safe_join(workdir, source.name))
        started = datetime.now(UTC)
        command = [sys.executable, str(script_dest), *spec.args]
        log_parts: list[str] = []
        state = JobState.FAILED
        exit_code: int | None = None
        message = ""
        try:
            completed = _run_script(command, workdir, spec.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            log_parts.append(_captured_text(exc.stdout))
            log_parts.append(_captured_text(exc.stderr))
            log_parts.append(f"\n[timed out after {spec.timeout_seconds} seconds]\n")
            message = f"script timed out after {spec.timeout_seconds} seconds"
        except OSError as exc:
            message = redact(str(exc))
            log_parts.append(message)
        else:
            log_parts.append(completed.stdout or "")
            log_parts.append(completed.stderr or "")
            exit_code = completed.returncode
            if exit_code == 0:
                state = JobState.SUCCEEDED
                message = "exited 0"
            else:
                message = f"script exited with status {exit_code}"
        log_text = redact("".join(log_parts))
        (workdir / "log.txt").write_text(log_text, encoding="utf-8")
        finished = datetime.now(UTC)
        artifacts = _artifact_names(workdir / "outputs")
        record = StoredJob(
            job_id=job_id,
            provider=self.name,
            remote_id=None,
            state=state,
            message=message,
            submitted_at=started,
            started_at=started,
            finished_at=finished,
            exit_code=exit_code,
            artifacts=artifacts,
            is_mock=True,
            workdir=str(workdir),
        )
        self._jobs.save(record)
        return JobHandle(
            job_id=job_id,
            provider=self.name,
            remote_id=None,
            submitted_at=started,
        )

    def status(self, job_id: str) -> JobResult:
        return to_result(self._jobs.load(job_id))

    def logs(self, job_id: str) -> str:
        workdir = self._jobs.workdir(job_id)
        log_path = safe_join(workdir, "log.txt")
        if not log_path.is_file():
            return ""
        return redact(log_path.read_text(encoding="utf-8"))

    def download(self, job_id: str, dest_dir: Path) -> list[Artifact]:
        workdir = self._jobs.workdir(job_id)
        return _copy_outputs(workdir / "outputs", dest_dir)


def _artifact_names(outputs: Path) -> list[str]:
    if not outputs.is_dir():
        return []
    names: list[str] = []
    for dirpath, dirnames, filenames in os.walk(outputs, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if not (current / dirname).is_symlink()]
        for filename in filenames:
            source = current / filename
            if source.is_symlink() or not source.is_file():
                continue
            names.append(source.relative_to(outputs).as_posix())
    return sorted(names)
