"""Spec hash for dedupe. The idempotency key is not part of the hash."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from link2kaggle_colab.core.errors import InvalidInputError
from link2kaggle_colab.core.models import JobSpec

_CHUNK = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise InvalidInputError(
            "input path could not be read",
            details={"name": path.name},
        ) from exc
    with handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise InvalidInputError(
            "input path does not exist",
            details={"name": path.name},
        )
    if path.is_file():
        return {"name": path.name, "sha256": _sha256_file(path)}
    if path.is_dir():
        files: list[dict[str, str]] = []
        for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
            dirnames.sort()
            current = Path(dirpath)
            for name in sorted(filenames):
                file_path = current / name
                if not file_path.is_file():
                    raise InvalidInputError(
                        "input directory contains an entry that is not a file",
                        details={"name": file_path.name},
                    )
                relative = file_path.relative_to(path).as_posix()
                files.append({"path": relative, "sha256": _sha256_file(file_path)})
        files.sort(key=lambda item: item["path"])
        return {"name": path.name, "files": files}
    raise InvalidInputError(
        "input path is not a file or directory",
        details={"name": path.name},
    )


def canonical_spec_json(spec: JobSpec) -> str:
    """Canonical JSON hashed by :func:`spec_hash`.

    File bodies are represented only by sha256 digests. ``idempotency_key``
    is omitted. ``args`` are included because they change the job, and are
    not written to logs.
    """
    provider = spec.provider.value if spec.provider is not None else None
    gpu_type = spec.gpu_type.value if spec.gpu_type is not None else None
    payload: dict[str, Any] = {
        "provider": provider,
        "gpu_type": gpu_type,
        "gpu_count": spec.gpu_count,
        "min_gpu_memory_gb_per_gpu": spec.min_gpu_memory_gb_per_gpu,
        "timeout_seconds": spec.timeout_seconds,
        "name": spec.name,
        "args": list(spec.args),
        "allow_premium": spec.allow_premium,
        "high_mem": spec.high_mem,
        "output_dir": None if spec.output_dir is None else str(spec.output_dir),
        "script_sha256": _sha256_file(spec.script_path),
        "inputs": [_file_entry(path) for path in spec.input_files],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_hash(spec: JobSpec) -> str:
    """Return the sha256 hex digest of :func:`canonical_spec_json`."""
    document = canonical_spec_json(spec)
    return hashlib.sha256(document.encode("utf-8")).hexdigest()
