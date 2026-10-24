"""Path helpers that refuse traversal outside a base directory."""

from __future__ import annotations

from pathlib import Path

from link2kaggle_colab.core.errors import PathSecurityError


def _reject_part(part: str) -> None:
    if part in {"", ".", ".."} or "\x00" in part or "\\" in part:
        raise PathSecurityError("unsafe path component")
    piece = Path(part)
    if piece.is_absolute() or ".." in piece.parts:
        raise PathSecurityError("unsafe path component")


def ensure_within(base: Path, path: Path) -> Path:
    """Resolve ``path`` and require it to stay inside ``base``.

    Symlinks are resolved. A link whose target leaves ``base`` raises
    ``PathSecurityError``.
    """
    root = Path(base).expanduser().resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PathSecurityError(
            "path escapes the base directory",
            details={"reason": "resolved path is outside the base directory"},
        ) from exc
    return resolved


def safe_join(base: Path, *parts: str) -> Path:
    """Join ``parts`` onto ``base`` and require the result to stay inside it."""
    root = Path(base).expanduser().resolve()
    if not parts:
        return root
    current = root
    for part in parts:
        _reject_part(part)
        current = current / part
    return ensure_within(root, current)


def resolve_artifact_dest(root: Path, dest_dir: Path | None, job_id: str) -> Path:
    """Resolve download dest under ``root``. ``None`` → ``root/job_id``."""
    base = Path(root).expanduser().resolve()
    if dest_dir is None:
        return safe_join(base, job_id)
    _reject_part(job_id)
    return ensure_within(base, Path(dest_dir))
