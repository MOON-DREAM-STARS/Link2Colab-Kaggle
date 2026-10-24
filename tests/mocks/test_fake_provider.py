"""FakeProvider runs local scripts and never touches the network."""

import subprocess
import sys
from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import (
    InvalidInputError,
    NotSupportedError,
    PathSecurityError,
    ResourceUnavailableError,
)
from link2kaggle_colab.core.models import GpuType, JobSpec, JobState, ProviderName, QuotaUnit
from link2kaggle_colab.providers.fake.provider import FakeProvider
from link2kaggle_colab.storage.sqlite_store import SqliteJobStore, default_db_path

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "train_example.py"


def test_runs_example_script_and_downloads_outputs(state_dir: Path) -> None:
    provider = FakeProvider(state_dir=state_dir)
    assert provider.capabilities().supports_cancel is False
    resources = provider.list_resources()
    assert len(resources) == 1
    assert resources[0].cpu_only is True
    quota = provider.get_quota()
    assert quota.is_fixture is True
    assert quota.unit is QuotaUnit.unknown
    assert quota.known is False
    assert "TEST FIXTURE" in quota.message
    assert provider.auth_status().is_mock is True
    handle = provider.submit(JobSpec(script_path=_EXAMPLE, args=[], name="train-example"))
    result = provider.status(handle.job_id)
    assert result.state is JobState.SUCCEEDED
    assert result.exit_code == 0
    assert result.is_mock is True
    assert result.state.is_terminal()
    logs = provider.logs(handle.job_id)
    assert "step 1/3" in logs
    assert "wrote outputs/result.txt" in logs
    dest = state_dir / "downloaded"
    artifacts = provider.download(handle.job_id, dest)
    assert [item.name for item in artifacts] == ["result.txt"]
    assert (dest / "result.txt").read_text(encoding="utf-8") == "status=ok\n"
    with pytest.raises(NotSupportedError):
        provider.cancel(handle.job_id)


def test_args_inputs_and_ignores_files_outside_outputs(tmp_path: Path, state_dir: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "print(sys.argv[1])\n"
        "print(Path('data.txt').read_text())\n"
        "Path('notes.txt').write_text('not-an-output\\n')\n"
        "Path('outputs').mkdir()\n"
        "Path('outputs/result.txt').write_text('yes\\n')\n",
        encoding="utf-8",
    )
    data = tmp_path / "data.txt"
    data.write_text("payload-42\n", encoding="utf-8")
    provider = FakeProvider(state_dir=state_dir)
    handle = provider.submit(
        JobSpec(script_path=script, args=["hello world"], input_files=[data])
    )
    logs = provider.logs(handle.job_id)
    assert "hello world" in logs
    assert "payload-42" in logs
    dest = state_dir / "out"
    artifacts = provider.download(handle.job_id, dest)
    assert [item.name for item in artifacts] == ["result.txt"]
    assert not (dest / "notes.txt").exists()


def test_failing_script_records_exit_code(tmp_path: Path, state_dir: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\nsys.stderr.write('boom-failure\\n')\nsys.exit(3)\n",
        encoding="utf-8",
    )
    provider = FakeProvider(state_dir=state_dir)
    handle = provider.submit(JobSpec(script_path=script))
    result = provider.status(handle.job_id)
    assert result.state is JobState.FAILED
    assert result.exit_code == 3
    assert "boom-failure" in provider.logs(handle.job_id)


def test_printed_output_survives_a_real_timeout(tmp_path: Path, state_dir: Path) -> None:
    """A local script prints, flushes, then sleeps past the timeout.

    FakeProvider runs it with ``sys.executable``. The partial line and the
    timeout note are kept, and Client can read status and logs afterwards.
    """
    script = tmp_path / "slow.py"
    script.write_text(
        "import sys\n"
        "import time\n"
        "print(sys.executable, flush=True)\n"
        "print('partial-before-sleep', flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    settings = Settings(mode="mock", allow_real_jobs=False, state_dir=state_dir)
    client = Client(settings)
    handle = client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName.fake,
            gpu_type=GpuType.NONE,
            gpu_count=0,
            timeout_seconds=1,
        )
    )
    result = client.status(handle.job_id)
    assert result.state is JobState.FAILED
    assert result.exit_code is None
    assert "timed out" in result.message
    logs = client.logs(handle.job_id)
    assert sys.executable in logs
    assert "partial-before-sleep" in logs
    assert "timed out after 1 seconds" in logs
    stored = SqliteJobStore(default_db_path(settings.state_dir)).get(handle.job_id)
    assert stored.provider is ProviderName.fake
    assert stored.job_id == handle.job_id
    assert stored.provider_job_id
    assert stored.provider_job_id != handle.job_id
    assert (state_dir / "fake" / stored.provider_job_id / "job.json").is_file()
    again = Client(settings)
    assert again.status(handle.job_id).state is JobState.FAILED
    assert "partial-before-sleep" in again.logs(handle.job_id)


def test_timeout_expired_bytes_are_decoded(
    tmp_path: Path,
    state_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('unused')\n", encoding="utf-8")
    seen = {"count": 0}

    def explode(*_args: object, **_kwargs: object) -> None:
        seen["count"] += 1
        if seen["count"] == 1:
            raise subprocess.TimeoutExpired(
                cmd=["python", "ok.py"],
                timeout=1,
                output=b"partial-bytes\xff",
                stderr=b"err-bytes",
            )
        raise subprocess.TimeoutExpired(cmd=["python", "ok.py"], timeout=1)

    monkeypatch.setattr("link2kaggle_colab.providers.fake.provider._run_script", explode)
    provider = FakeProvider(state_dir=state_dir)
    handle = provider.submit(JobSpec(script_path=script, timeout_seconds=1))
    assert provider.status(handle.job_id).state is JobState.FAILED
    logs = provider.logs(handle.job_id)
    assert "partial-bytes" in logs
    assert "\ufffd" in logs
    assert "err-bytes" in logs
    assert "timed out after 1 seconds" in logs
    assert (state_dir / "fake" / handle.job_id / "job.json").is_file()

    second = provider.submit(JobSpec(script_path=script, timeout_seconds=1))
    assert provider.status(second.job_id).state is JobState.FAILED
    assert "timed out after 1 seconds" in provider.logs(second.job_id)


def test_timeout_is_a_failed_job(tmp_path: Path, state_dir: Path) -> None:
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    provider = FakeProvider(state_dir=state_dir)
    handle = provider.submit(JobSpec(script_path=script, timeout_seconds=1))
    result = provider.status(handle.job_id)
    assert result.state is JobState.FAILED
    assert result.exit_code is None
    assert "timed out" in result.message


def test_notebook_and_gpu_and_missing_input_are_rejected(tmp_path: Path, state_dir: Path) -> None:
    notebook = tmp_path / "job.ipynb"
    notebook.write_text("{}\n", encoding="utf-8")
    script = tmp_path / "ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    provider = FakeProvider(state_dir=state_dir)
    with pytest.raises(NotSupportedError):
        provider.submit(JobSpec(script_path=notebook))
    with pytest.raises(ResourceUnavailableError):
        provider.submit(JobSpec(script_path=script, gpu_type=GpuType.T4))
    with pytest.raises(InvalidInputError):
        provider.submit(JobSpec(script_path=script, input_files=[tmp_path / "missing.txt"]))


def test_log_redacts_tokens_and_status_persists(tmp_path: Path, state_dir: Path) -> None:
    marker = "KGAT_" + ("z" * 20)
    script = tmp_path / "leak.py"
    script.write_text(f"print({marker!r})\n", encoding="utf-8")
    first = FakeProvider(state_dir=state_dir)
    handle = first.submit(JobSpec(script_path=script))
    assert marker not in first.logs(handle.job_id)
    assert "[REDACTED]" in first.logs(handle.job_id)
    raw = (state_dir / "fake" / handle.job_id / "job.json").read_text(encoding="utf-8")
    second = FakeProvider(state_dir=state_dir)
    assert second.status(handle.job_id).state is JobState.SUCCEEDED
    assert (state_dir / "fake" / handle.job_id / "job.json").read_text(encoding="utf-8") == raw
    assert marker not in raw


def test_download_rejects_symlink_escape(tmp_path: Path, state_dir: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('outputs').mkdir()\n"
        "Path('outputs/result.txt').write_text('ok\\n')\n",
        encoding="utf-8",
    )
    provider = FakeProvider(state_dir=state_dir)
    handle = provider.submit(JobSpec(script_path=script))
    outside = tmp_path / "secret.txt"
    sentinel = "sentinel-secret-value"
    outside.write_text(sentinel, encoding="utf-8")
    outputs = state_dir / "fake" / handle.job_id / "outputs"
    (outputs / "escape.txt").symlink_to(outside)
    dest = state_dir / "dest"
    with pytest.raises(PathSecurityError):
        provider.download(handle.job_id, dest)
    copied = "".join(
        path.read_text(encoding="utf-8") for path in dest.rglob("*") if path.is_file()
    )
    assert sentinel not in copied
