"""Detect credential *sources* without reading secret values into results.

File checks use ``Path.is_file`` only. Environment checks test that a variable
is non-empty. The secret bytes are never copied into ``AuthStatus``, logs, or
error details. Finding a file is ``unknown``, never ``authenticated``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from link2kaggle_colab.auth.base import AuthProvider
from link2kaggle_colab.core.models import AuthMethod, AuthState, AuthStatus, ProviderName

CREDENTIAL_FOUND_MESSAGE = "credential source found; validity not verified until a real call"
CREDENTIAL_MISSING_MESSAGE = "no credential source found"


def _nonempty(environ: Mapping[str, str], name: str) -> bool:
    value = environ.get(name)
    return bool(value and value.strip())


class KaggleCredentialDetector(AuthProvider):
    """Detect a Kaggle credential source without opening the file.

    Order: ``KAGGLE_API_TOKEN``, then ``access_token``, ``credentials.json``
    (OAuth cache written by ``kaggle auth login``), then legacy ``kaggle.json``.
    ``KAGGLE_CONFIG_DIR``, when set, replaces ``~/.kaggle`` for those three files.
    A hit is ``unknown``, never ``authenticated``.
    """

    provider = ProviderName.kaggle

    def __init__(
        self,
        *,
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._home = Path(home) if home is not None else Path.home()
        self._environ = os.environ if environ is None else environ

    def _config_dir(self) -> tuple[Path, str]:
        if _nonempty(self._environ, "KAGGLE_CONFIG_DIR"):
            return Path(self._environ["KAGGLE_CONFIG_DIR"]).expanduser(), "KAGGLE_CONFIG_DIR"
        return self._home / ".kaggle", "~/.kaggle"

    def status(self) -> AuthStatus:
        if _nonempty(self._environ, "KAGGLE_API_TOKEN"):
            return self._found(AuthMethod.api_token, "env:KAGGLE_API_TOKEN")
        root, label = self._config_dir()
        if (root / "access_token").is_file():
            return self._found(AuthMethod.api_token, f"file:{label}/access_token")
        # Presence only. The OAuth cache is never opened.
        if (root / "credentials.json").is_file():
            return self._found(AuthMethod.oauth, f"file:{label}/credentials.json")
        if (root / "kaggle.json").is_file():
            return self._found(AuthMethod.legacy_key, f"file:{label}/kaggle.json")
        return self._missing()

    def _found(self, method: AuthMethod, source: str) -> AuthStatus:
        return AuthStatus(
            provider=self.provider,
            state=AuthState.unknown,
            method=method,
            account_label=None,
            is_mock=False,
            message=CREDENTIAL_FOUND_MESSAGE,
            details={"source": source},
        )

    def _missing(self) -> AuthStatus:
        return AuthStatus(
            provider=self.provider,
            state=AuthState.unauthenticated,
            method=AuthMethod.none,
            account_label=None,
            is_mock=False,
            message=CREDENTIAL_MISSING_MESSAGE,
            details={},
        )


class ColabCredentialDetector(AuthProvider):
    """Detect the Colab CLI token file or Application Default Credentials.

    ADC counts when ``GOOGLE_APPLICATION_CREDENTIALS`` names an existing file,
    or when ``~/.config/gcloud/application_default_credentials.json`` exists.
    """

    provider = ProviderName.colab

    def __init__(
        self,
        *,
        home: Path | None = None,
        environ: Mapping[str, str] | None = None,
        auth: str | None = None,
    ) -> None:
        self._home = Path(home) if home is not None else Path.home()
        self._environ = os.environ if environ is None else environ
        # None keeps the historical order (token file, then ADC). ``oauth2`` and
        # ``adc`` look at only the source that matches ``colab --auth``.
        self._auth = auth

    def status(self) -> AuthStatus:
        if self._auth == "oauth2":
            return self._from_token()
        if self._auth == "adc":
            return self._from_adc()
        token = self._from_token()
        if token.state is not AuthState.unauthenticated:
            return token
        return self._from_adc()

    def _from_token(self) -> AuthStatus:
        token = self._home / ".config" / "colab-cli" / "token.json"
        if token.is_file():
            return self._found(AuthMethod.oauth, "file:~/.config/colab-cli/token.json")
        return self._missing()

    def _from_adc(self) -> AuthStatus:
        adc_env = self._environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if adc_env and Path(adc_env).expanduser().is_file():
            return self._found(AuthMethod.adc, "env:GOOGLE_APPLICATION_CREDENTIALS")
        adc_file = self._home / ".config" / "gcloud" / "application_default_credentials.json"
        if adc_file.is_file():
            return self._found(
                AuthMethod.adc,
                "file:~/.config/gcloud/application_default_credentials.json",
            )
        return self._missing()

    def _missing(self) -> AuthStatus:
        return AuthStatus(
            provider=self.provider,
            state=AuthState.unauthenticated,
            method=AuthMethod.none,
            account_label=None,
            is_mock=False,
            message=CREDENTIAL_MISSING_MESSAGE,
            details={},
        )

    def _found(self, method: AuthMethod, source: str) -> AuthStatus:
        return AuthStatus(
            provider=self.provider,
            state=AuthState.unknown,
            method=method,
            account_label=None,
            is_mock=False,
            message=CREDENTIAL_FOUND_MESSAGE,
            details={"source": source},
        )
