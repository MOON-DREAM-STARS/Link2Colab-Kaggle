"""Artifact downloads stay inside Settings.artifact_root."""

from pathlib import Path

import pytest

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.errors import PathSecurityError
from link2kaggle_colab.core.models import JobSpec, ProviderName
from link2kaggle_colab.core.paths import resolve_artifact_dest


def test_resolve_default_uses_job_id(tmp_path: Path) -> None:
    root = tmp_path / "arts"
    root.mkdir()
    dest = resolve_artifact_dest(root, None, "abc123")
    assert dest == (root / "abc123").resolve()


def test_resolve_child_under_root(tmp_path: Path) -> None:
    root = tmp_path / "arts"
    root.mkdir()
    dest = resolve_artifact_dest(root, root / "out", "abc123")
    assert dest == (root / "out").resolve()


def test_resolve_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "arts"
    root.mkdir()
    with pytest.raises(PathSecurityError):
        resolve_artifact_dest(root, tmp_path / "outside", "abc123")
    with pytest.raises(PathSecurityError):
        resolve_artifact_dest(root, Path(".."), "abc123")


def test_client_download_rejects_outside_root(tmp_path: Path, py_script: Path) -> None:
    settings = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        artifact_root=tmp_path / "arts",
    )
    (tmp_path / "arts").mkdir()
    client = Client(settings)
    handle = client.submit(
        JobSpec(script_path=py_script, provider=ProviderName.fake, gpu_type=None, gpu_count=0)
    )
    with pytest.raises(PathSecurityError):
        client.download(handle.job_id, tmp_path / "evil")


def test_client_download_default_under_root(tmp_path: Path) -> None:
    root = tmp_path / "arts"
    script = tmp_path / "with_out.py"
    script.write_text(
        "from pathlib import Path\n"
        "out = Path('outputs')\n"
        "out.mkdir()\n"
        "(out / 'result.txt').write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    settings = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        artifact_root=root,
    )
    client = Client(settings)
    handle = client.submit(
        JobSpec(script_path=script, provider=ProviderName.fake, gpu_type=None, gpu_count=0)
    )
    artifacts = client.download(handle.job_id)
    assert artifacts
    assert artifacts[0].local_path.resolve().is_relative_to(root.resolve())
