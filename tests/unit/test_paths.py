"""Path helpers refuse traversal and escaping symlinks."""

from pathlib import Path

import pytest

from link2kaggle_colab.core.errors import PathSecurityError
from link2kaggle_colab.core.paths import ensure_within, safe_join


def test_safe_join_keeps_children_and_rejects_escape(tmp_path: Path) -> None:
    child = safe_join(tmp_path, "sub", "file.txt")
    assert child == (tmp_path / "sub" / "file.txt").resolve()
    assert ensure_within(tmp_path, child) == child
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "..", "etc")
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "/etc/passwd")
    with pytest.raises(PathSecurityError):
        safe_join(tmp_path, "foo/../../etc")


def test_symlink_that_leaves_the_base_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    sentinel = outside / "secret.txt"
    sentinel.write_text("sentinel-secret", encoding="utf-8")
    (base / "link").symlink_to(outside)
    with pytest.raises(PathSecurityError):
        ensure_within(base, base / "link" / "secret.txt")
    with pytest.raises(PathSecurityError):
        safe_join(base, "link", "secret.txt")
