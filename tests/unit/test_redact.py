"""Redaction rules for tokens that must never reach logs or errors."""

from link2kaggle_colab.core.redact import redact, redact_value


def test_masks_kaggle_google_and_refresh_tokens() -> None:
    kaggle = "KGAT_" + ("a" * 20)
    access = "ya29." + ("b" * 24)
    refresh = "1//" + ("c" * 16)
    text = f"prefix {kaggle} mid {access} end {refresh}"
    masked = redact(text)
    assert kaggle not in masked
    assert access not in masked
    assert refresh not in masked
    assert "[REDACTED]" in masked
    assert "prefix" in masked


def test_masks_long_assignments_and_json_fields() -> None:
    assigned = "d" * 32
    short = "e" * 31
    json_token = "f" * 16
    json_key = "g" * 12
    text = (
        f"token={assigned} token={short} "
        f'{{"token": "{json_token}", "key": "{json_key}", "label": "T4"}}'
    )
    masked = redact(text)
    assert assigned not in masked
    assert f"token={short}" in masked
    assert json_token not in masked
    assert json_key not in masked
    assert "T4" in masked


def test_masks_authorization_header_and_leaves_prose() -> None:
    header = "Authorization: Bearer " + ("h" * 20)
    masked = redact(f"step 1/3\n{header}\nwrote outputs/result.txt")
    assert "h" * 20 not in masked
    assert "step 1/3" in masked
    assert "wrote outputs/result.txt" in masked


def test_redact_value_walks_nested_data() -> None:
    marker = "KGAT_" + ("z" * 12)
    cleaned = redact_value({"outer": [{"token": marker}], "count": 2})
    assert isinstance(cleaned, dict)
    assert marker not in str(cleaned)
    assert cleaned["count"] == 2


def test_empty_text_is_unchanged() -> None:
    assert redact("") == ""


def test_masks_env_style_secret_assignments() -> None:
    kaggle = "0123456789abcdef0123456789abcdef"
    hf = "hf_abcdefghijklmnop"
    db_value = "hunter2hunter2"
    aws = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    long_id = "job-42-abcdef"
    text = (
        f"KAGGLE_KEY={kaggle} HF_TOKEN={hf} "
        f'export DB_PASSWORD="{db_value}" AWS_SECRET_ACCESS_KEY={aws} '
        f"gpu: T4 idempotency_key=job-42 idempotency_key={long_id} key=short"
    )
    masked = redact(text)
    assert kaggle not in masked
    assert hf not in masked
    assert db_value not in masked
    assert aws not in masked
    assert "KAGGLE_KEY=[REDACTED]" in masked
    assert "HF_TOKEN=[REDACTED]" in masked
    assert 'DB_PASSWORD="[REDACTED]"' in masked
    assert "AWS_SECRET_ACCESS_KEY=[REDACTED]" in masked
    assert "gpu: T4" in masked
    assert "idempotency_key=job-42" in masked
    assert f"idempotency_key={long_id}" in masked
    assert "key=short" in masked


def test_redact_value_masks_secret_mapping_keys() -> None:
    cleaned = redact_value(
        {
            "api_key": "plainvalue123",
            "nested": {"client_secret": "abcdefgh"},
            "idempotency_key": "k-1",
            "gpu_type": "T4",
            "count": 3,
            "API_KEY": "",
            "token": None,
            "password": False,
            "secret": 0,
            "credentials": {"user": "alice", "n": 1, "items": ["secret-value"]},
            "key": "short-value",
        }
    )
    assert isinstance(cleaned, dict)
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["nested"] == {"client_secret": "[REDACTED]"}
    assert cleaned["idempotency_key"] == "k-1"
    assert cleaned["gpu_type"] == "T4"
    assert cleaned["count"] == 3
    assert cleaned["API_KEY"] == ""
    assert cleaned["token"] is None
    assert cleaned["password"] is False
    assert cleaned["secret"] == 0
    assert cleaned["credentials"] == {
        "user": "[REDACTED]",
        "n": 1,
        "items": ["[REDACTED]"],
    }
    assert cleaned["key"] == "short-value"
    assert redact_value({"key": "k" * 16}) == {"key": "[REDACTED]"}
    assert redact_value({"hf_token": "hf_abcdefghijklmnop"}) == {"hf_token": "[REDACTED]"}
