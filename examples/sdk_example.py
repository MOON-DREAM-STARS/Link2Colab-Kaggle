"""Run examples/train_example.py through Client in mock mode.

Quota and auth values are TEST FIXTURES. This script does not call Kaggle,
Colab, or the network. Job files stay in a temporary directory.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from link2kaggle_colab import Client
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.models import GpuType, JobSpec, ProviderName


def main() -> None:
    script = Path(__file__).resolve().parent / "train_example.py"
    with tempfile.TemporaryDirectory(prefix="l2kc-example-") as raw:
        state = Path(raw)
        client = Client(Settings(mode="mock", allow_real_jobs=False, state_dir=state))
        decision = client.recommend_backend(gpu_type=GpuType.NONE, gpu_count=0)
        print(f"provider: {decision['provider']}")
        for warning in decision["warnings"]:
            print(f"warning: {warning}")
        provider = decision["provider"]
        if not isinstance(provider, str):
            raise SystemExit("no backend selected")
        handle = client.submit(
            JobSpec(
                script_path=script,
                provider=ProviderName(provider),
                gpu_type=GpuType.NONE,
                gpu_count=0,
            )
        )
        result = client.status(handle.job_id)
        logs = client.logs(handle.job_id)
        print(f"job_id: {handle.job_id}")
        print(f"state: {result.state.value}")
        print(logs, end="" if logs.endswith("\n") else "\n")
        for artifact in client.download(handle.job_id, state / "artifacts"):
            print(f"artifact: {artifact.name}")


if __name__ == "__main__":
    main()
