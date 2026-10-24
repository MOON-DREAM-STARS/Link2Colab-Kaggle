"""The declared MCP pin must match the SDK API the server imports."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]


def _mcp_requirement() -> Requirement:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = [Requirement(r) for r in data["project"]["dependencies"]]
    (mcp,) = [r for r in reqs if r.name == "mcp"]
    return mcp


def test_mcp_pin_excludes_1x_and_3x() -> None:
    spec = _mcp_requirement().specifier
    assert Version("1.27.0") not in spec
    assert Version("2.0.0") in spec
    assert Version("3.0.0") not in spec


def test_mcpserver_import_available() -> None:
    from mcp.server.mcpserver import MCPServer

    assert MCPServer is not None
