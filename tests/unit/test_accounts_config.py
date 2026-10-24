"""Account and credential settings reject secrets and inconsistent rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from link2kaggle_colab.accounts.registry import AccountRegistry
from link2kaggle_colab.core.config import load_config
from link2kaggle_colab.core.errors import ConfigError

_DESIGN_YAML = """\
accounts:
  - id: kaggle-primary
    provider: kaggle
    credential_ref: kaggle-main
    enabled: true
    default: false
    labels:
      team: a
    mock_scenario: quota_45h
  - id: kaggle-alt
    provider: kaggle
    credential_ref: kaggle-alt
    enabled: true
    default: true
  - id: colab-primary
    provider: colab
    credential_ref: colab-main
    enabled: true
  - id: colab-alt
    provider: colab
    credential_ref: colab-alt
    enabled: true
credentials:
  kaggle-main:
    source: kaggle_config_dir
    path: ~/.l2kc/credentials/kaggle-main
  kaggle-alt:
    source: env
    env: KAGGLE_ALT_API_TOKEN
  colab-main:
    source: adc_file
    path: ~/creds/colab-main-adc.json
  colab-alt:
    source: colab_home
    path: ~/.l2kc/credentials/colab-alt
  vault-x:
    source: keyring
    service: l2kc
    username: kaggle-x
"""


def _write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Path:
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    *banned: str,
) -> ConfigError:
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    payload = {"state_dir": str(tmp_path / "state"), **body}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    error = caught.value
    assert error.code == "config_error"
    rendered = str(error)
    blob = json.dumps(error.to_dict())
    for secret in banned:
        assert secret not in rendered
        assert secret not in blob
    return error


def test_design_yaml_example_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("L2KC_STATE_DIR", raising=False)
    text = f"state_dir: {tmp_path / 'state'}\n{_DESIGN_YAML}"
    loaded = load_config(_write(tmp_path, monkeypatch, text))
    assert loaded.accounts is not None
    assert [account.id for account in loaded.accounts] == [
        "kaggle-primary",
        "kaggle-alt",
        "colab-primary",
        "colab-alt",
    ]
    primary = loaded.accounts[0]
    assert primary.provider.value == "kaggle"
    assert primary.credential_ref == "kaggle-main"
    assert primary.enabled is True
    assert primary.default is False
    assert primary.labels == {"team": "a"}
    assert primary.mock_scenario == "quota_45h"
    assert loaded.accounts[1].default is True
    assert loaded.credentials["kaggle-main"].source == "kaggle_config_dir"
    assert loaded.credentials["kaggle-main"].path == "~/.l2kc/credentials/kaggle-main"
    assert loaded.credentials["kaggle-alt"].source == "env"
    assert loaded.credentials["kaggle-alt"].env == "KAGGLE_ALT_API_TOKEN"
    assert loaded.credentials["colab-main"].source == "adc_file"
    assert loaded.credentials["colab-alt"].source == "colab_home"
    assert loaded.credentials["vault-x"].source == "keyring"
    assert loaded.credentials["vault-x"].service == "l2kc"
    assert loaded.credentials["vault-x"].username == "kaggle-x"


def test_implicit_defaults_when_accounts_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = (
        f"state_dir: {tmp_path / 'state'}\n"
        "providers:\n"
        "  kaggle:\n"
        "    enabled: false\n"
        "  fake:\n"
        "    enabled: false\n"
    )
    loaded = load_config(_write(tmp_path, monkeypatch, text))
    assert loaded.accounts is None
    registry = AccountRegistry.from_settings(loaded)
    accounts = registry.list()
    assert [account.id for account in accounts] == [
        "kaggle-default",
        "colab-default",
        "fake-default",
    ]
    assert [account.enabled for account in accounts] == [False, True, False]
    assert all(account.implicit for account in accounts)
    assert all(account.credential_ref is None for account in accounts)
    assert all(account.is_default for account in accounts)
    assert not (tmp_path / "state" / "accounts_state.json").exists()


def test_literal_kaggle_key_credential_ref_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "cafebabe" * 4
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": key},
            ]
        },
        key,
    )


def test_kgat_in_env_field_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "KGAT_" + ("A" * 16)
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "credentials": {"kaggle-alt": {"source": "env", "env": token}},
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": "kaggle-alt"},
            ],
        },
        token,
    )


def test_jwt_and_ya29_in_path_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jwt = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0.signature"
    ya = "ya29." + ("a" * 24)
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "credentials": {
                "colab-main": {"source": "adc_file", "path": f"/creds/{jwt}/{ya}.json"},
            }
        },
        jwt,
        ya,
    )


def test_pem_in_label_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    header = "-----BEGIN PRIVATE KEY-----"
    body = "MIIEPEMVALUE1234"
    pem = f"{header}\n{body}\n-----END PRIVATE KEY-----"
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {
                    "id": "kaggle-a",
                    "provider": "kaggle",
                    "labels": {"note": pem},
                }
            ]
        },
        header,
        body,
        pem,
    )


def test_kaggle_json_credential_ref_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "0123456789abcdef" * 2
    raw = json.dumps({"username": "u", "key": key})
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": raw},
            ]
        },
        raw,
        key,
    )


def test_secret_keys_inside_accounts_credentials_and_labels_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_value = "deadbeef" * 4
    api_key_value = "ab" * 16
    label_note = "label-password-value-xyz"
    nested_note = "nested-secret-value-xyz"
    cases = [
        (
            {
                "accounts": [
                    {"id": "kaggle-a", "provider": "kaggle", "token": token_value},
                ]
            },
            token_value,
        ),
        (
            {
                "credentials": {
                    "kaggle-main": {
                        "source": "env",
                        "env": "KAGGLE_ALT_API_TOKEN",
                        "api_key": api_key_value,
                    }
                }
            },
            api_key_value,
        ),
        (
            {
                "accounts": [
                    {
                        "id": "kaggle-a",
                        "provider": "kaggle",
                        "labels": {"password": label_note},
                    }
                ]
            },
            label_note,
        ),
        (
            {
                "credentials": {
                    "kaggle-main": {
                        "source": "kaggle_config_dir",
                        "path": "~/.l2kc/credentials/kaggle-main",
                        "secret": nested_note,
                    }
                }
            },
            nested_note,
        ),
    ]
    for body, secret in cases:
        _rejected(tmp_path, monkeypatch, body, secret)


def test_duplicate_account_id_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-dup", "provider": "kaggle", "credential_ref": "ref-a"},
                {"id": "kaggle-dup", "provider": "kaggle", "credential_ref": "ref-b"},
            ],
            "credentials": {
                "ref-a": {"source": "env", "env": "KAGGLE_ONE_API_TOKEN"},
                "ref-b": {"source": "env", "env": "KAGGLE_TWO_API_TOKEN"},
            },
        },
        "kaggle-dup",
    )


def test_two_defaults_for_one_provider_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {
                    "id": "kaggle-primary",
                    "provider": "kaggle",
                    "credential_ref": "ref-a",
                    "default": True,
                },
                {
                    "id": "kaggle-alt",
                    "provider": "kaggle",
                    "credential_ref": "ref-b",
                    "default": True,
                },
            ],
            "credentials": {
                "ref-a": {"source": "env", "env": "KAGGLE_ONE_API_TOKEN"},
                "ref-b": {"source": "env", "env": "KAGGLE_TWO_API_TOKEN"},
            },
        },
        "kaggle-primary",
        "kaggle-alt",
    )


def test_shared_expanded_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    absolute = str((home / "same" / "kaggle").resolve())
    error = _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": "ref-a"},
                {"id": "kaggle-b", "provider": "kaggle", "credential_ref": "ref-b"},
            ],
            "credentials": {
                "ref-a": {"source": "kaggle_config_dir", "path": "~/same/kaggle"},
                "ref-b": {"source": "kaggle_config_dir", "path": absolute},
            },
        },
        "~/same/kaggle",
        absolute,
        "same/kaggle",
    )
    assert error.message == "accounts must not share credentials"
    assert error.details["field"] == "accounts.credential_ref"


def test_shared_env_var_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": "ref-a"},
                {"id": "kaggle-b", "provider": "kaggle", "credential_ref": "ref-b"},
            ],
            "credentials": {
                "ref-a": {"source": "env", "env": "KAGGLE_SHARED_TOKEN"},
                "ref-b": {"source": "env", "env": "KAGGLE_SHARED_TOKEN"},
            },
        },
        "KAGGLE_SHARED_TOKEN",
    )
    assert error.message == "accounts must not share credentials"


def test_shared_keyring_identity_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": "ref-a"},
                {"id": "kaggle-b", "provider": "kaggle", "credential_ref": "ref-b"},
            ],
            "credentials": {
                "ref-a": {
                    "source": "keyring",
                    "service": "l2kc-shared-vault",
                    "username": "same-user",
                },
                "ref-b": {
                    "source": "keyring",
                    "service": "l2kc-shared-vault",
                    "username": "same-user",
                },
            },
        },
        "l2kc-shared-vault",
        "same-user",
    )
    assert error.message == "accounts must not share credentials"


def test_convention_path_colliding_with_declared_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = (tmp_path / "state").resolve()
    convention = (state / "credentials" / "orphan").resolve()
    error = _rejected(
        tmp_path,
        monkeypatch,
        {
            "state_dir": str(state),
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": "orphan"},
                {"id": "kaggle-b", "provider": "kaggle", "credential_ref": "declared"},
            ],
            "credentials": {
                "declared": {"source": "kaggle_config_dir", "path": str(convention)},
            },
        },
        str(convention),
        "orphan",
    )
    assert error.message == "accounts must not share credentials"
    assert error.details == {"field": "accounts.credential_ref"}


def test_shared_credential_ref_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle", "credential_ref": "shared-ref"},
                {"id": "kaggle-b", "provider": "kaggle", "credential_ref": "shared-ref"},
            ],
            "credentials": {
                "shared-ref": {"source": "env", "env": "KAGGLE_ALT_API_TOKEN"},
            },
        },
        "shared-ref",
    )


def test_two_ambient_kaggle_accounts_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rejected(
        tmp_path,
        monkeypatch,
        {
            "accounts": [
                {"id": "kaggle-a", "provider": "kaggle"},
                {"id": "kaggle-b", "provider": "kaggle"},
            ]
        },
        "kaggle-a",
        "kaggle-b",
    )


def test_bad_env_name_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _rejected(
        tmp_path,
        monkeypatch,
        {"credentials": {"kaggle-alt": {"source": "env", "env": "not-a-name"}}},
        "not-a-name",
    )


def test_account_id_regex_and_secret_ids_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_id = "ab" * 16
    _rejected(
        tmp_path,
        monkeypatch,
        {"accounts": [{"id": secret_id, "provider": "kaggle"}]},
        secret_id,
    )
    _rejected(
        tmp_path,
        monkeypatch,
        {"accounts": [{"id": "Bad_Id", "provider": "kaggle"}]},
        "Bad_Id",
    )
