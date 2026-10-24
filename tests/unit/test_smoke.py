"""Smoke tests for package import and the CLI version command."""

from typer.testing import CliRunner

from link2kaggle_colab import __version__
from link2kaggle_colab.cli.main import app

runner = CliRunner()


def test_package_imports_and_version_exists() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_cli_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
