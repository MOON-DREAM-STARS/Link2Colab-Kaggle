"""Offline command runner and CLI argument checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from link2kaggle_colab.core.envspec import AccountEnv
from link2kaggle_colab.core.errors import InvalidInputError, ProviderCLIError
from link2kaggle_colab.core.process import (
    SubprocessRunner,
    absolute_cli_path,
    validate_cli_arg,
    validate_kernel_ref,
    validate_remote_relpath,
    validate_session_name,
    validate_slug,
)


def test_subprocess_runner_captures_local_python(tmp_path: Path) -> None:
    result = SubprocessRunner().run(
        [sys.executable, "-c", "print(1)"],
        cwd=tmp_path,
        timeout=30,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "1"
    assert result.stderr == ""
    assert result.args == [sys.executable, "-c", "print(1)"]
    assert result.duration >= 0


def test_subprocess_runner_uses_list_args_and_does_not_pass_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = SubprocessRunner().run(
        [sys.executable, "-c", "print(1)"],
        cwd=tmp_path,
        timeout=5,
        input="ignored",
    )
    assert result.stdout == "ok\n"
    assert captured["args"] == [sys.executable, "-c", "print(1)"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert kwargs["cwd"] == tmp_path
    assert kwargs["timeout"] == 5
    assert kwargs["input"] == "ignored"
    assert "env" not in kwargs


def test_subprocess_runner_applies_account_env_without_touching_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "KGAT_" + ("e" * 20)
    monkeypatch.setenv("L2KC_PARENT_ONLY", "parent-marker")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("L2KC_CHILD_MARKER", raising=False)
    before = dict(os.environ)
    result = SubprocessRunner().run(
        [sys.executable, "-c", "import os,json;print(json.dumps(dict(os.environ)))"],
        timeout=30,
        env=AccountEnv(
            set={"KAGGLE_API_TOKEN": secret, "L2KC_CHILD_MARKER": "from-account"},
            unset=frozenset({"L2KC_PARENT_ONLY"}),
        ),
    )
    assert dict(os.environ) == before
    assert result.returncode == 0
    assert not hasattr(result, "env")
    assert "env" not in result.__dataclass_fields__
    child = json.loads(result.stdout)
    assert child["L2KC_CHILD_MARKER"] == "from-account"
    assert "L2KC_PARENT_ONLY" not in child
    assert "KAGGLE_API_TOKEN" in child
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert secret not in str(result)
    assert child["KAGGLE_API_TOKEN"] != secret
    assert os.environ["L2KC_PARENT_ONLY"] == "parent-marker"
    assert "KAGGLE_API_TOKEN" not in os.environ


def test_subprocess_runner_rejects_non_list_and_redacts_stderr() -> None:
    with pytest.raises(InvalidInputError):
        SubprocessRunner().run("print")  # type: ignore[arg-type]
    token = "KGAT_" + ("b" * 16)
    script = f"import sys; sys.stderr.write({token!r}); raise SystemExit(3)"
    result = SubprocessRunner().run([sys.executable, "-c", script])
    assert result.returncode == 3
    assert token not in result.stderr
    assert "REDACTED" in result.stderr


def test_subprocess_runner_missing_binary_and_timeout(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-l2kc-cli"
    with pytest.raises(ProviderCLIError) as missing_exc:
        SubprocessRunner().run([str(missing)])
    assert missing_exc.value.code == "cli_missing"

    with pytest.raises(ProviderCLIError) as timeout_exc:
        SubprocessRunner().run(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.2,
        )
    assert timeout_exc.value.code == "cli_timeout"


def test_subprocess_runner_timeout_excerpt_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "KGAT_" + ("d" * 12)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["sleep"], timeout=0.1, stderr=token.encode())

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ProviderCLIError) as caught:
        SubprocessRunner().run([sys.executable, "-c", "pass"], timeout=0.1)
    assert caught.value.code == "cli_timeout"
    blob = json.dumps(caught.value.to_dict())
    assert token not in blob
    assert token not in str(caught.value)


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("plain", True),
        ("owner", True),
        ("-rf", False),
        ("--help", False),
        ("a\nb", False),
        ("a\rb", False),
        ("a\x00b", False),
        ("", False),
    ],
)
def test_validate_cli_arg_rejects_option_injection(value: str, ok: bool) -> None:
    if ok:
        assert validate_cli_arg(value, "flag") == value
    else:
        with pytest.raises(InvalidInputError):
            validate_cli_arg(value, "flag")


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("abc", True),
        ("a" * 50, True),
        ("a--b", True),
        ("ab", False),
        ("a" * 51, False),
        ("Abc", False),
        ("a_b", False),
        ("-abc", False),
        ("ab\n", False),
    ],
)
def test_validate_slug(value: str, ok: bool) -> None:
    if ok:
        assert validate_slug(value) == value
    else:
        with pytest.raises(InvalidInputError):
            validate_slug(value)


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("owner/kernel", True),
        ("Owner_1/abc", True),
        ("owner/kernel/3", True),
        ("owner/kernel/12", True),
        ("-owner/kernel", False),
        ("owner/Kernel", False),
        ("owner/kernel/x", False),
        ("../x", False),
        ("owner/slug\n", False),
    ],
)
def test_validate_kernel_ref(value: str, ok: bool) -> None:
    if ok:
        assert validate_kernel_ref(value) == value
    else:
        with pytest.raises(InvalidInputError):
            validate_kernel_ref(value)


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("l2kc-a", True),
        ("l2kc-" + ("a" * 40), True),
        ("l2kc-job-1", True),
        ("l2kc-", False),
        ("l2kc-" + ("a" * 41), False),
        ("session", False),
        ("L2kc-abc", False),
        ("-l2kc-abc", False),
    ],
)
def test_validate_session_name(value: str, ok: bool) -> None:
    if ok:
        assert validate_session_name(value) == value
    else:
        with pytest.raises(InvalidInputError):
            validate_session_name(value)


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("out.txt", True),
        ("outputs/out.txt", True),
        ("../x", False),
        ("foo/../x", False),
        ("-rf", False),
        ("/etc/passwd", False),
        ("a\\b", False),
        ("a\nb", False),
        ("foo/-rf", False),
        ("", False),
    ],
)
def test_validate_remote_relpath(value: str, ok: bool) -> None:
    if ok:
        assert validate_remote_relpath(value) == value
    else:
        with pytest.raises(InvalidInputError):
            validate_remote_relpath(value)


def test_absolute_cli_path_resolves(tmp_path: Path) -> None:
    target = tmp_path / "script.py"
    target.write_text("print(1)\n", encoding="utf-8")
    text = absolute_cli_path(target)
    assert text == str(target.resolve())
    assert Path(text).is_absolute()
    with pytest.raises(InvalidInputError):
        absolute_cli_path(Path("a\nb"))
