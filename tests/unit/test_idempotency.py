"""Spec hashes use digests and ignore the client idempotency key."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from link2kaggle_colab.core.errors import InvalidInputError
from link2kaggle_colab.core.models import GpuType, JobSpec, ProviderName
from link2kaggle_colab.jobs.idempotency import canonical_spec_json, spec_hash


def _script(tmp_path: Path, body: str = "print('ok')\n") -> Path:
    path = tmp_path / "train.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_hash_ignores_idempotency_key_and_raw_file_bytes(tmp_path: Path) -> None:
    script = _script(tmp_path, "print('marker-BODY-xyz')\n")
    data = tmp_path / "data.txt"
    data.write_text("raw-file-body-SECRET", encoding="utf-8")
    spec = JobSpec(
        script_path=script,
        args=["--epochs", "1"],
        provider=ProviderName.kaggle,
        gpu_type=GpuType.T4,
        gpu_count=2,
        min_gpu_memory_gb_per_gpu=16,
        timeout_seconds=30,
        name="train-1",
        input_files=[data],
        output_dir=tmp_path / "out",
        allow_premium=False,
        idempotency_key="client-key.1",
    )
    renamed = spec.model_copy(update={"idempotency_key": "other-key"})
    assert spec_hash(spec) == spec_hash(renamed)
    assert spec_hash(spec) == spec_hash(spec.model_copy(update={"idempotency_key": None}))
    document = canonical_spec_json(spec)
    assert "marker-BODY-xyz" not in document
    assert "raw-file-body-SECRET" not in document
    assert "client-key.1" not in document
    assert "idempotency_key" not in document
    payload = json.loads(document)
    assert payload["inputs"][0]["name"] == "data.txt"
    assert payload["inputs"][0]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert payload["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert payload["args"] == ["--epochs", "1"]
    assert payload["provider"] == "kaggle"
    assert payload["output_dir"] == str(tmp_path / "out")


def test_directory_entries_are_sorted_relative_paths(tmp_path: Path) -> None:
    script = _script(tmp_path)
    folder = tmp_path / "inputs"
    nested = folder / "sub"
    nested.mkdir(parents=True)
    (folder / "b.txt").write_bytes(b"b")
    (nested / "c.txt").write_bytes(b"c")
    (folder / "a.txt").write_bytes(b"a")
    spec = JobSpec(script_path=script, input_files=[folder])
    payload = json.loads(canonical_spec_json(spec))
    files = payload["inputs"][0]["files"]
    assert [item["path"] for item in files] == ["a.txt", "b.txt", "sub/c.txt"]
    assert payload["inputs"][0]["name"] == "inputs"
    first = spec_hash(spec)
    (folder / "a.txt").write_bytes(b"a")
    assert spec_hash(spec) == first
    (folder / "a.txt").write_bytes(b"changed")
    assert spec_hash(spec) != first


def test_script_bytes_args_and_provider_change_the_hash(tmp_path: Path) -> None:
    script = _script(tmp_path, "print(1)\n")
    base = JobSpec(script_path=script, args=["--epochs", "1"])
    original = spec_hash(base)
    script.write_text("print(2)\n", encoding="utf-8")
    assert spec_hash(base) != original
    script.write_text("print(1)\n", encoding="utf-8")
    assert spec_hash(base) == original
    assert spec_hash(base.model_copy(update={"args": ["--epochs", "2"]})) != original
    assert spec_hash(base.model_copy(update={"provider": ProviderName.colab})) != original
    assert spec_hash(base.model_copy(update={"allow_premium": True})) != original
    assert spec_hash(base.model_copy(update={"gpu_type": GpuType.T4})) != original
    assert spec_hash(base.model_copy(update={"high_mem": True})) != original


def test_missing_input_is_rejected(tmp_path: Path) -> None:
    script = _script(tmp_path)
    spec = JobSpec(script_path=script, input_files=[tmp_path / "missing.txt"])
    with pytest.raises(InvalidInputError):
        spec_hash(spec)


def test_idempotency_key_pattern(tmp_path: Path) -> None:
    script = _script(tmp_path)
    assert JobSpec(script_path=script, idempotency_key="A:b.1_ok-key").idempotency_key == (
        "A:b.1_ok-key"
    )
    assert JobSpec(script_path=script, idempotency_key="a" * 128).idempotency_key is not None
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, idempotency_key="a" * 129)
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, idempotency_key="has space")
    with pytest.raises(ValidationError):
        JobSpec(script_path=script, idempotency_key="")
