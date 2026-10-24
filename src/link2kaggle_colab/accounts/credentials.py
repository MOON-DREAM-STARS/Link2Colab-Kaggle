"""Resolve a credential reference into a child environment.

File checks use ``Path.is_file`` only. Environment values are read when
:func:`resolve_credential` is called, then kept only on :class:`AccountEnv`.
They are not logged. ``repr`` of that overlay masks the values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from link2kaggle_colab.accounts.models import (
    AdcFileCredential,
    ColabHomeCredential,
    CredentialRefConfig,
    EnvCredential,
    KaggleConfigDirCredential,
    KeyringCredential,
)
from link2kaggle_colab.core.config import Settings
from link2kaggle_colab.core.envspec import AccountEnv
from link2kaggle_colab.core.errors import CredentialRefError, PathSecurityError
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.core.paths import safe_join

_KAGGLE_TOKEN_VARS = frozenset({"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"})
_KAGGLE_CONFIG_UNSET = frozenset({"KAGGLE_CONFIG_DIR", "KAGGLE_USERNAME", "KAGGLE_KEY"})


@dataclass(frozen=True, slots=True)
class ResolvedCredential:
    """Where an account's secret lives, expressed as process environment changes."""

    env: AccountEnv
    home: Path | None
    state_root: Path | None
    source: str


@dataclass(frozen=True, slots=True)
class _View:
    kind: str
    path: Path | None = None
    env_name: str | None = None
    service: str | None = None
    username: str | None = None
    home: Path | None = None
    state_root: Path | None = None


def _empty(source: str) -> ResolvedCredential:
    return ResolvedCredential(
        env=AccountEnv.empty(),
        home=None,
        state_root=None,
        source=source,
    )


def _convention_root(settings: Settings, ref_name: str) -> Path:
    try:
        return safe_join(settings.state_dir, "credentials", ref_name)
    except PathSecurityError as exc:
        raise CredentialRefError(
            "credential reference name is not a safe path",
            details={"field": "credential_ref", "account_id": None},
        ) from exc


def _describe(
    ref_name: str | None,
    ref_config: CredentialRefConfig | None,
    provider: ProviderName,
    settings: Settings,
) -> _View:
    if ref_config is None:
        if ref_name is None:
            return _View(kind="ambient")
        if provider is ProviderName.fake:
            return _View(kind="none")
        root = _convention_root(settings, ref_name)
        if provider is ProviderName.kaggle:
            return _View(kind="kaggle_config_dir", path=root, state_root=root)
        if provider is ProviderName.colab and settings.providers.colab.auth == "oauth2":
            return _View(kind="colab_home", path=root, home=root, state_root=root)
        if provider is ProviderName.colab:
            adc = root / "application_default_credentials.json"
            return _View(kind="adc_file", path=adc, state_root=root)
        return _View(kind="none")

    if isinstance(ref_config, KaggleConfigDirCredential):
        path = Path(ref_config.path).expanduser()
        return _View(kind="kaggle_config_dir", path=path, state_root=path)
    if isinstance(ref_config, EnvCredential):
        return _View(kind="env", env_name=ref_config.env)
    if isinstance(ref_config, AdcFileCredential):
        path = Path(ref_config.path).expanduser()
        return _View(kind="adc_file", path=path, state_root=path.parent)
    if isinstance(ref_config, ColabHomeCredential):
        path = Path(ref_config.path).expanduser()
        return _View(kind="colab_home", path=path, home=path, state_root=path)
    if isinstance(ref_config, KeyringCredential):
        return _View(kind="keyring", service=ref_config.service, username=ref_config.username)
    raise CredentialRefError(
        "credential source is not supported",
        details={"field": "source", "account_id": None},
    )


def _load_keyring() -> object | None:
    try:
        import keyring
    except ImportError:
        return None
    return keyring


def _keyring_password(service: str, username: str) -> str | None:
    """Return the stored password, or None when the entry is missing.

    Import failure is reported by the caller as unsupported. Exceptions from
    the backend are not chained: their text can contain the secret.
    """
    module = _load_keyring()
    if module is None:
        return None
    getter = getattr(module, "get_password", None)
    if not callable(getter):
        raise CredentialRefError(
            "keyring credential source is unavailable",
            details={"source": f"keyring:{service}", "reason": "unsupported", "account_id": None},
        )
    try:
        password = getter(service, username)
    except Exception:
        raise CredentialRefError(
            "keyring credential source is unavailable",
            details={"source": f"keyring:{service}", "reason": "unsupported", "account_id": None},
        ) from None
    if password is None or password == "":
        return None
    if not isinstance(password, str):
        raise CredentialRefError(
            "keyring credential source is unavailable",
            details={"source": f"keyring:{service}", "reason": "unsupported", "account_id": None},
        )
    return password


def resolve_credential(
    ref_name: str | None,
    ref_config: CredentialRefConfig | None,
    provider: ProviderName,
    settings: Settings,
) -> ResolvedCredential:
    """Build the child environment for one credential reference.

    ``ref_config is None`` with a name uses ``state_dir/credentials/<name>/``.
    Kaggle uses that directory as ``KAGGLE_CONFIG_DIR``. Colab uses it as
    ``HOME`` when ``providers.colab.auth`` is ``oauth2``, and otherwise as an
    ADC file at ``<dir>/application_default_credentials.json``. Fake accounts
    have no credentials. A null name is ambient: an empty overlay.

    An ``env`` source reads ``os.environ`` at this call and passes the value
    to the child as ``KAGGLE_API_TOKEN``. The value is not stored on the
    reference and does not appear in ``repr`` or ``str``.
    """
    view = _describe(ref_name, ref_config, provider, settings)
    if view.kind in {"ambient", "none"}:
        return _empty(view.kind)
    if view.kind == "kaggle_config_dir":
        path = view.path
        if path is None:
            raise CredentialRefError(
                "credential path is missing",
                details={"field": "path", "account_id": None},
            )
        return ResolvedCredential(
            env=AccountEnv(
                set={"KAGGLE_CONFIG_DIR": str(path)},
                unset=_KAGGLE_TOKEN_VARS,
            ),
            home=None,
            state_root=view.state_root,
            source=f"file:{path}",
        )
    if view.kind == "env":
        name = view.env_name or ""
        raw = os.environ.get(name)
        if raw is None or raw == "":
            raise CredentialRefError(
                "credential environment variable is not set",
                details={"field": "env", "source": f"env:{name}", "account_id": None},
            )
        return ResolvedCredential(
            env=AccountEnv(set={"KAGGLE_API_TOKEN": raw}, unset=_KAGGLE_CONFIG_UNSET),
            home=None,
            state_root=None,
            source=f"env:{name}",
        )
    if view.kind == "adc_file":
        path = view.path
        if path is None:
            raise CredentialRefError(
                "credential path is missing",
                details={"field": "path", "account_id": None},
            )
        return ResolvedCredential(
            env=AccountEnv(
                set={"GOOGLE_APPLICATION_CREDENTIALS": str(path)},
                unset=frozenset(),
            ),
            home=None,
            state_root=view.state_root,
            source=f"file:{path}",
        )
    if view.kind == "colab_home":
        path = view.path
        if path is None:
            raise CredentialRefError(
                "credential path is missing",
                details={"field": "path", "account_id": None},
            )
        return ResolvedCredential(
            env=AccountEnv(
                set={"HOME": str(path)},
                unset=frozenset({"GOOGLE_APPLICATION_CREDENTIALS"}),
            ),
            home=view.home,
            state_root=view.state_root,
            source=f"file:{path}",
        )
    if view.kind == "keyring":
        service = view.service or ""
        username = view.username or ""
        source = f"keyring:{service}"
        if _load_keyring() is None:
            raise CredentialRefError(
                "keyring credential source is unavailable",
                details={"source": source, "reason": "unsupported", "account_id": None},
            )
        if provider is not ProviderName.kaggle:
            raise CredentialRefError(
                "keyring credential source is not supported for this provider",
                details={"source": source, "account_id": None},
            )
        password = _keyring_password(service, username)
        if password is None:
            raise CredentialRefError(
                "keyring credential is missing",
                details={"source": source, "reason": "missing", "account_id": None},
            )
        return ResolvedCredential(
            env=AccountEnv(set={"KAGGLE_API_TOKEN": password}, unset=_KAGGLE_CONFIG_UNSET),
            home=None,
            state_root=None,
            source=source,
        )
    raise CredentialRefError(
        "credential source is not supported",
        details={"field": "source", "account_id": None},
    )


def _file_source(path: Path, present: bool) -> dict[str, str]:
    return {"state": "present" if present else "missing", "source": f"file:{path}"}


def credential_health(
    ref_name: str | None,
    ref_config: CredentialRefConfig | None,
    provider: ProviderName,
    settings: Settings,
) -> dict[str, str]:
    """Report whether a credential source is present.

    Presence only. File contents are never read. No network. A missing
    ``keyring`` package is ``unsupported``.
    """
    view = _describe(ref_name, ref_config, provider, settings)
    if view.kind == "ambient":
        return {"state": "present", "source": "ambient"}
    if view.kind == "none":
        return {"state": "present", "source": "none"}
    if view.kind == "kaggle_config_dir" and view.path is not None:
        for name in ("access_token", "credentials.json", "kaggle.json"):
            candidate = view.path / name
            if candidate.is_file():
                return _file_source(candidate, True)
        return _file_source(view.path, False)
    if view.kind == "env":
        name = view.env_name or ""
        raw = os.environ.get(name)
        present = bool(raw and raw.strip())
        return {"state": "present" if present else "missing", "source": f"env:{name}"}
    if view.kind == "adc_file" and view.path is not None:
        return _file_source(view.path, view.path.is_file())
    if view.kind == "colab_home" and view.path is not None:
        token = view.path / ".config" / "colab-cli" / "token.json"
        return _file_source(token, token.is_file())
    if view.kind == "keyring":
        service = view.service or ""
        source = f"keyring:{service}"
        if _load_keyring() is None:
            return {"state": "unsupported", "source": source}
        try:
            password = _keyring_password(service, view.username or "")
        except CredentialRefError:
            return {"state": "unsupported", "source": source}
        return {"state": "present" if password else "missing", "source": source}
    return {"state": "missing", "source": view.kind}
