"""Mock auth scenarios and real credential-source detection."""

import inspect
from datetime import UTC, datetime
from pathlib import Path

import pytest

from link2kaggle_colab.auth.mock import (
    COLAB_MOCK_ACCOUNT,
    KAGGLE_MOCK_ACCOUNT,
    MockColabAuth,
    MockKaggleAuth,
)
from link2kaggle_colab.auth.real import (
    CREDENTIAL_FOUND_MESSAGE,
    CREDENTIAL_MISSING_MESSAGE,
    ColabCredentialDetector,
    KaggleCredentialDetector,
)
from link2kaggle_colab.core.errors import AuthError, ConfigError, NotSupportedError
from link2kaggle_colab.core.models import AuthMethod, AuthState, ProviderName


@pytest.mark.parametrize(
    ("scenario", "state", "method"),
    [
        ("token_valid", AuthState.authenticated, AuthMethod.api_token),
        ("token_invalid", AuthState.invalid, AuthMethod.api_token),
        ("token_missing", AuthState.unauthenticated, AuthMethod.api_token),
        ("oauth_ok", AuthState.authenticated, AuthMethod.oauth),
        ("oauth_fail", AuthState.error, AuthMethod.oauth),
        ("permission_error", AuthState.error, AuthMethod.api_token),
    ],
)
def test_kaggle_auth_scenarios(scenario: str, state: AuthState, method: AuthMethod) -> None:
    status = MockKaggleAuth(scenario).status()
    assert status.provider is ProviderName.kaggle
    assert status.state is state
    assert status.method is method
    assert status.is_mock is True
    assert status.account_label == KAGGLE_MOCK_ACCOUNT
    assert "TEST FIXTURE" in status.message


@pytest.mark.parametrize(
    ("scenario", "state", "method"),
    [
        ("oauth_ok", AuthState.authenticated, AuthMethod.oauth),
        ("oauth_fail", AuthState.error, AuthMethod.oauth),
        ("token_expired", AuthState.expired, AuthMethod.oauth),
        ("refresh_fail", AuthState.expired, AuthMethod.oauth),
        ("adc_valid", AuthState.authenticated, AuthMethod.adc),
        ("adc_missing", AuthState.unauthenticated, AuthMethod.adc),
    ],
)
def test_colab_auth_scenarios(scenario: str, state: AuthState, method: AuthMethod) -> None:
    status = MockColabAuth(scenario).status()
    assert status.provider is ProviderName.colab
    assert status.state is state
    assert status.method is method
    assert status.is_mock is True
    assert status.account_label == COLAB_MOCK_ACCOUNT
    assert "TEST FIXTURE" in status.message


def test_colab_refresh_succeeds_only_for_expired_token() -> None:
    auth = MockColabAuth("token_expired")
    before = auth.status()
    assert before.state is AuthState.expired
    assert before.expires_at is not None
    assert before.expires_at < datetime.now(UTC)
    refreshed = auth.refresh()
    assert refreshed.state is AuthState.authenticated
    assert refreshed.is_mock is True
    assert refreshed.expires_at is not None
    assert refreshed.expires_at > datetime.now(UTC)
    assert auth.status().state is AuthState.authenticated

    failing = MockColabAuth("refresh_fail")
    assert failing.status().state is AuthState.expired
    with pytest.raises(AuthError):
        failing.refresh()
    with pytest.raises(NotSupportedError):
        MockColabAuth("oauth_ok").refresh()
    with pytest.raises(NotSupportedError):
        MockKaggleAuth("token_valid").refresh()


def test_unknown_auth_scenario_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        MockKaggleAuth("nope")
    with pytest.raises(ConfigError):
        MockColabAuth("nope")


def test_detector_source_does_not_read_secret_files() -> None:
    for detector in (KaggleCredentialDetector, ColabCredentialDetector):
        text = Path(inspect.getfile(detector)).read_text(encoding="utf-8")
        assert "read_text" not in text
        assert "read_bytes" not in text
        assert ".read(" not in text


def test_kaggle_detector_presence_is_unknown_and_never_leaks(tmp_path: Path) -> None:
    home = tmp_path / "home"
    kaggle_dir = home / ".kaggle"
    kaggle_dir.mkdir(parents=True)
    file_marker = "KGAT_" + ("f" * 24)
    (kaggle_dir / "access_token").write_text(file_marker, encoding="utf-8")
    (kaggle_dir / "kaggle.json").write_text('{"key": "json-marker-value"}', encoding="utf-8")
    env_marker = "KGAT_" + ("e" * 24)

    from_env = KaggleCredentialDetector(
        home=home,
        environ={"KAGGLE_API_TOKEN": env_marker},
    ).status()
    blob = from_env.model_dump_json()
    assert from_env.state is AuthState.unknown
    assert from_env.state is not AuthState.authenticated
    assert from_env.method is AuthMethod.api_token
    assert from_env.is_mock is False
    assert from_env.message == CREDENTIAL_FOUND_MESSAGE
    assert from_env.details["source"] == "env:KAGGLE_API_TOKEN"
    assert env_marker not in blob
    assert file_marker not in blob
    assert "json-marker-value" not in blob

    from_file = KaggleCredentialDetector(home=home, environ={}).status()
    assert from_file.state is AuthState.unknown
    assert from_file.method is AuthMethod.api_token
    assert from_file.details["source"] == "file:~/.kaggle/access_token"
    assert file_marker not in from_file.model_dump_json()

    (kaggle_dir / "access_token").unlink()
    from_json = KaggleCredentialDetector(home=home, environ={}).status()
    assert from_json.method is AuthMethod.legacy_key
    assert from_json.state is AuthState.unknown
    assert "json-marker-value" not in from_json.model_dump_json()

    (kaggle_dir / "kaggle.json").unlink()
    missing = KaggleCredentialDetector(home=home, environ={}).status()
    assert missing.state is AuthState.unauthenticated
    assert missing.method is AuthMethod.none
    assert missing.message == CREDENTIAL_MISSING_MESSAGE
    with pytest.raises(NotSupportedError):
        KaggleCredentialDetector(home=home, environ={}).refresh()


def test_colab_detector_token_and_adc(tmp_path: Path) -> None:
    home = tmp_path / "home"
    token = home / ".config" / "colab-cli" / "token.json"
    token.parent.mkdir(parents=True)
    token.write_text('{"access_token": "ya29.should-not-leak"}', encoding="utf-8")
    adc = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text('{"refresh_token": "1//should-not-leak"}', encoding="utf-8")

    from_token = ColabCredentialDetector(home=home, environ={}).status()
    assert from_token.state is AuthState.unknown
    assert from_token.method is AuthMethod.oauth
    assert from_token.is_mock is False
    assert from_token.message == CREDENTIAL_FOUND_MESSAGE
    dumped = from_token.model_dump_json()
    assert "should-not-leak" not in dumped
    assert "ya29." not in dumped

    token.unlink()
    from_adc = ColabCredentialDetector(home=home, environ={}).status()
    assert from_adc.state is AuthState.unknown
    assert from_adc.method is AuthMethod.adc
    assert "should-not-leak" not in from_adc.model_dump_json()

    env_file = tmp_path / "adc.json"
    env_file.write_text('{"private_key": "pem-marker-value"}', encoding="utf-8")
    from_env = ColabCredentialDetector(
        home=home,
        environ={"GOOGLE_APPLICATION_CREDENTIALS": str(env_file)},
    ).status()
    assert from_env.method is AuthMethod.adc
    assert from_env.details["source"] == "env:GOOGLE_APPLICATION_CREDENTIALS"
    assert "pem-marker-value" not in from_env.model_dump_json()

    adc.unlink()
    missing_target = ColabCredentialDetector(
        home=home,
        environ={"GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "missing.json")},
    ).status()
    assert missing_target.state is AuthState.unauthenticated
    assert missing_target.message == CREDENTIAL_MISSING_MESSAGE
    empty = ColabCredentialDetector(home=home, environ={"KAGGLE_API_TOKEN": ""}).status()
    assert empty.state is AuthState.unauthenticated


def test_kaggle_config_dir_and_oauth_cache_are_presence_only(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".kaggle").mkdir(parents=True)
    (home / ".kaggle" / "kaggle.json").write_text(
        '{"key": "home-json-marker"}', encoding="utf-8"
    )
    override = tmp_path / "cfg"
    override.mkdir()
    marker = "oauth-cache-marker"
    (override / "credentials.json").write_text(
        '{"token": "' + marker + '"}', encoding="utf-8"
    )
    (override / "kaggle.json").write_text('{"key": "override-json-marker"}', encoding="utf-8")

    found = KaggleCredentialDetector(
        home=home,
        environ={"KAGGLE_CONFIG_DIR": str(override)},
    ).status()
    blob = found.model_dump_json()
    assert found.state is AuthState.unknown
    assert found.state is not AuthState.authenticated
    assert found.method is AuthMethod.oauth
    assert found.details["source"] == "file:KAGGLE_CONFIG_DIR/credentials.json"
    assert marker not in blob
    assert "home-json-marker" not in blob
    assert "override-json-marker" not in blob
    assert str(home) not in blob
    assert str(override) not in blob

    (override / "access_token").write_text("KGAT_" + ("a" * 24), encoding="utf-8")
    from_token = KaggleCredentialDetector(
        home=home,
        environ={"KAGGLE_CONFIG_DIR": str(override)},
    ).status()
    assert from_token.method is AuthMethod.api_token
    assert from_token.details["source"] == "file:KAGGLE_CONFIG_DIR/access_token"
    assert "KGAT_" not in from_token.model_dump_json()

    from_env = KaggleCredentialDetector(
        home=home,
        environ={
            "KAGGLE_CONFIG_DIR": str(override),
            "KAGGLE_API_TOKEN": "KGAT_" + ("b" * 24),
        },
    ).status()
    assert from_env.method is AuthMethod.api_token
    assert from_env.details["source"] == "env:KAGGLE_API_TOKEN"

    (override / "access_token").unlink()
    (override / "credentials.json").unlink()
    from_json = KaggleCredentialDetector(
        home=home,
        environ={"KAGGLE_CONFIG_DIR": str(override)},
    ).status()
    assert from_json.method is AuthMethod.legacy_key
    assert from_json.details["source"] == "file:KAGGLE_CONFIG_DIR/kaggle.json"
    assert "override-json-marker" not in from_json.model_dump_json()

    blank = KaggleCredentialDetector(
        home=home,
        environ={"KAGGLE_CONFIG_DIR": "   "},
    ).status()
    assert blank.method is AuthMethod.legacy_key
    assert blank.details["source"] == "file:~/.kaggle/kaggle.json"


def test_colab_detector_auth_filter(tmp_path: Path) -> None:
    home = tmp_path / "home"
    token = home / ".config" / "colab-cli" / "token.json"
    token.parent.mkdir(parents=True)
    token.write_text('{"access_token": "ya29.filter-marker"}', encoding="utf-8")
    adc = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text('{"refresh_token": "1//filter-marker"}', encoding="utf-8")

    oauth_only = ColabCredentialDetector(home=home, environ={}, auth="oauth2").status()
    assert oauth_only.method is AuthMethod.oauth
    assert oauth_only.state is AuthState.unknown
    assert "filter-marker" not in oauth_only.model_dump_json()

    token.unlink()
    assert (
        ColabCredentialDetector(home=home, environ={}, auth="oauth2").status().state
        is AuthState.unauthenticated
    )
    adc_only = ColabCredentialDetector(home=home, environ={}, auth="adc").status()
    assert adc_only.method is AuthMethod.adc
    assert adc_only.state is not AuthState.authenticated
    token.write_text("{}", encoding="utf-8")
    still_adc = ColabCredentialDetector(home=home, environ={}, auth="adc").status()
    assert still_adc.method is AuthMethod.adc
