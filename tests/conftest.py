"""Shared pytest fixtures. Live tests are opt-in via the ``live`` marker."""

from pathlib import Path

import pytest


@pytest.fixture
def py_script(tmp_path: Path) -> Path:
    path = tmp_path / "script.py"
    path.write_text("print('hello-from-script')\n", encoding="utf-8")
    return path


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    path = tmp_path / "state"
    path.mkdir()
    return path
