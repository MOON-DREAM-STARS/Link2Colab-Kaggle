"""Offline flow: mock auth, quota, recommend, fake submit, download, recover.

Fake runs the script inside submit, so the first status read is already
SUCCEEDED. A Kaggle mock job in the same test is what walks QUEUED, RUNNING,
and SUCCEEDED. Nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path

from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.models import GpuType, JobSpec, JobState, ProviderName


def test_offline_mock_flow(tmp_path: Path) -> None:
    script = tmp_path / "train.py"
    script.write_text(
        "from pathlib import Path\n"
        "out = Path('outputs')\n"
        "out.mkdir()\n"
        "(out / 'result.txt').write_text('status=ok\\n', encoding='utf-8')\n"
        "print('step 1/1')\n",
        encoding="utf-8",
    )
    settings = Settings(
        mode="mock",
        allow_real_jobs=False,
        state_dir=tmp_path / "state",
        artifact_root=tmp_path,
    )
    client = Client(settings)

    auth = client.auth_status()
    assert {item.provider for item in auth} == {
        ProviderName.kaggle,
        ProviderName.colab,
        ProviderName.fake,
    }
    assert all(item.is_mock for item in auth)
    assert all("TEST FIXTURE" in item.message for item in auth)

    quotas = client.get_quota()
    assert any(item.is_fixture and item.provider is ProviderName.kaggle for item in quotas)
    assert all("TEST FIXTURE" in item.message for item in quotas)

    decision = client.recommend_backend(gpu_type=GpuType.NONE, gpu_count=0)
    assert decision["provider"] == "fake"
    assert any("TEST FIXTURE" in item for item in decision["warnings"])

    handle = client.submit(
        JobSpec(
            script_path=script,
            provider=ProviderName(decision["provider"]),
            gpu_type=GpuType.NONE,
            gpu_count=0,
        )
    )
    assert handle.provider is ProviderName.fake
    finished = client.status(handle.job_id)
    assert finished.state is JobState.SUCCEEDED
    assert "step 1/1" in client.logs(handle.job_id)

    kaggle = client.submit(
        JobSpec(script_path=script, provider=ProviderName.kaggle, gpu_type=GpuType.T4, gpu_count=2)
    )
    seen: list[JobState] = []
    for _ in range(6):
        current = client.status(kaggle.job_id)
        seen.append(current.state)
        if current.state is JobState.SUCCEEDED:
            break
    assert seen[0] is JobState.QUEUED
    assert JobState.RUNNING in seen
    assert seen[-1] is JobState.SUCCEEDED

    artifacts = client.download(handle.job_id, tmp_path / "out")
    assert len(artifacts) == 1
    assert artifacts[0].local_path.read_text(encoding="utf-8") == "status=ok\n"

    restored = Client(settings)
    again = restored.status(handle.job_id)
    assert again.job_id == handle.job_id
    assert again.provider is ProviderName.fake
    assert again.state is JobState.SUCCEEDED
    assert "step 1/1" in restored.logs(handle.job_id)
