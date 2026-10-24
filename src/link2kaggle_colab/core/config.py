"""YAML settings. Secrets stay in the environment or in credential files, not here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from link2kaggle_colab.accounts.models import (
    AccountConfig,
    AdcFileCredential,
    ColabHomeCredential,
    CredentialRefConfig,
    EnvCredential,
    KaggleConfigDirCredential,
    KeyringCredential,
    reject_ref_name,
)
from link2kaggle_colab.core.errors import ConfigError, PathSecurityError, RealModeDisabledError
from link2kaggle_colab.core.models import ProviderName
from link2kaggle_colab.core.paths import safe_join
from link2kaggle_colab.core.redact import redact
from link2kaggle_colab.scheduler.candidates import CANDIDATES, DEFAULT_COST_PRIORITY

# Third-party observation. Google publishes no fixed consumer Colab CCU rate.
_UNVERIFIED_CCU_SOURCE = (
    "unverified estimate: third-party report "
    "https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/ "
    "(checked 2026-03-10); Google publishes no fixed consumer Colab CCU rate; "
    "confirm with `colab usage`"
)
_DEFAULT_CCU_PER_HOUR: dict[str, float] = {
    "colab:T4": 1.19,
    "colab:L4": 1.71,
    "colab:A100-40GB": 5.40,
    "colab:A100-80GB": 7.52,
    # High-mem shape: fixture uplift over the base accelerator rate.
    "colab:T4+high_mem": 1.45,
    "colab:A100-40GB+high_mem": 6.50,
    "colab:A100-80GB+high_mem": 9.00,
    # TPU: fixture estimates only; confirm with `colab usage` on a live account.
    "colab:TPU-v5e1": 4.50,
    "colab:TPU-v6e1": 6.00,
}

_HIGH_MEM_CCU_SOURCE = (
    "unverified fixture estimate: high-mem shape uplift over the base "
    "accelerator rate; Google publishes no fixed consumer Colab CCU rate; "
    "confirm with `colab usage`"
)
_TPU_CCU_SOURCE = (
    "unverified fixture estimate: Colab TPU CCU rate; "
    "Google publishes no fixed consumer Colab CCU rate; confirm with `colab usage`"
)

_SECRET_FRAGMENTS = ("token", "secret", "password", "key")
_ALLOWED_SECRETISH_KEYS = frozenset({"credential_ref", "api_key_env"})
# Allowed only on a single ``credentials:`` entry. ``token`` / ``api_key`` /
# ``password`` / ``secret`` are not in this set and stay rejected everywhere.
_CREDENTIAL_ENTRY_KEYS = frozenset({"source", "env", "path", "service", "username"})


class KaggleProviderSettings(BaseModel):
    """Kaggle CLI settings. ``username`` is the kernel owner slug, not a secret."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cli_path: str = "kaggle"
    username: str | None = None
    credential_ref: str | None = None
    api_key_env: str | None = None
    # Bound for ``kaggle kernels push`` when the job leaves ``timeout_seconds``
    # unset. This limits the push command, not the kernel run (``-t``).
    default_push_timeout_seconds: int = Field(default=300, gt=0)
    # Bound for ``kaggle kernels output``. 600 seconds covers large logs and
    # artifacts; a fixed 120 second wait was too short.
    download_timeout_seconds: int = Field(default=600, gt=0)

    @field_validator("cli_path", "username", "credential_ref", "api_key_env")
    @classmethod
    def _value_is_not_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if redact(value) != value:
            raise ValueError(
                "value looks like a secret; use a credential ref or an environment variable name"
            )
        return value


class ColabProviderSettings(BaseModel):
    """Colab CLI settings. ``auth`` matches ``colab --auth {oauth2,adc}``."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    cli_path: str = "colab"
    auth: Literal["oauth2", "adc"] = "adc"
    credential_ref: str | None = None
    # Submit calls ``colab stop`` unless this is true, so a runtime cannot keep
    # spending compute units after the job returns.
    keep_runtime: bool = False
    # Bound for ``colab exec`` when the job leaves ``timeout_seconds`` unset.
    # 21600 seconds is 6 hours, so a hung exec cannot hold a runtime open.
    default_exec_timeout_seconds: int = Field(default=21600, gt=0)
    # Undocumented Colab ``ccu-info`` HTTP fetch. Default OFF; prefer ``colab usage``.
    # Also enabled when env ``L2KC_COLAB_EXPERIMENTAL_CCU_INFO`` is 1/true/yes/on.
    experimental_ccu_info: bool = False

    @field_validator("cli_path", "credential_ref")
    @classmethod
    def _value_is_not_secret(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if redact(value) != value:
            raise ValueError(
                "value looks like a secret; use a credential ref or an environment variable name"
            )
        return value


class FakeProviderSettings(BaseModel):
    """Local executor used by mock mode and tests."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # When false, FakeProvider.submit refuses to run local scripts.
    allow_exec: bool = True


class ProvidersSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kaggle: KaggleProviderSettings = Field(default_factory=KaggleProviderSettings)
    colab: ColabProviderSettings = Field(default_factory=ColabProviderSettings)
    fake: FakeProviderSettings = Field(default_factory=FakeProviderSettings)


class MockSettings(BaseModel):
    """Scenario names for mock providers. Values are test-fixture selectors."""

    model_config = ConfigDict(extra="forbid")

    kaggle_scenario: str = "ok"
    colab_scenario: str = "ok"


class CcuRate(BaseModel):
    """Hourly Colab compute-unit price for one paid candidate.

    ``verified`` is true only when the number comes from an official Google
    source. The live consumer rate is shown in the Colab UI or ``colab usage``.
    There is no published fixed per-GPU rate to copy.
    """

    model_config = ConfigDict(extra="forbid")

    ccu_per_hour: float = Field(gt=0)
    source: str = Field(min_length=1)
    verified: bool = False


class QuotaSettings(BaseModel):
    """How long one quota snapshot may be reused. 0 disables caching."""

    model_config = ConfigDict(extra="forbid")

    cache_ttl_seconds: float = Field(default=300, ge=0)


class JobsSettings(BaseModel):
    """Timeouts and retry bounds for the async job manager.

    ``submit_timeout_seconds`` is added to the job's ``timeout_seconds``
    (treated as 0 when unset). The Colab adapter's submit runs the script
    synchronously, so the manager wait has to cover that run plus overhead.

    A ``SUBMITTING`` row with no provider id, and no live submit in this
    process, is an interrupted submit once it is older than
    ``stale_submitting_seconds``. Reconcile marks it ``UNKNOWN`` and does
    not submit it again.
    """

    model_config = ConfigDict(extra="forbid")

    submit_timeout_seconds: float = Field(default=300, gt=0)
    status_timeout_seconds: float = Field(default=60, gt=0)
    logs_timeout_seconds: float = Field(default=60, gt=0)
    download_timeout_seconds: float = Field(default=600, gt=0)
    cancel_timeout_seconds: float = Field(default=120, gt=0)
    retry_max_attempts: int = Field(default=4, ge=1)
    retry_base_delay_seconds: float = Field(default=0.5, gt=0)
    retry_max_delay_seconds: float = Field(default=8.0, gt=0)
    reconcile_concurrency: int = Field(default=4, ge=1)
    stale_submitting_seconds: float = Field(default=900, gt=0)


def _default_cost_priority() -> list[str]:
    return list(DEFAULT_COST_PRIORITY)


def _default_ccu_rates() -> dict[str, CcuRate]:
    """Default Colab rates. All unverified. No rate for ``colab:H100``."""
    rates: dict[str, CcuRate] = {}
    for candidate_id, ccu_per_hour in _DEFAULT_CCU_PER_HOUR.items():
        if candidate_id.endswith("+high_mem"):
            source = _HIGH_MEM_CCU_SOURCE
        elif candidate_id.startswith("colab:TPU-"):
            source = _TPU_CCU_SOURCE
        else:
            source = _UNVERIFIED_CCU_SOURCE
        rates[candidate_id] = CcuRate(
            ccu_per_hour=ccu_per_hour,
            source=source,
            verified=False,
        )
    return rates


class SchedulerSettings(BaseModel):
    """Inputs for placement. ``cost_priority`` is a cost order, not a speed rank.

    ``unknown_kaggle_quota`` applies only to the free Kaggle backend. An unknown
    paid (Colab) balance is always skipped and is not configurable.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["kaggle_first"] = "kaggle_first"
    allow_colab_fallback: bool = True
    allow_premium_gpu: bool = False
    require_premium_confirmation: bool = True
    min_colab_ccu_reserve: float = Field(default=50, ge=0)
    max_ccu_per_job: float = Field(default=20, gt=0)
    cost_priority: list[str] = Field(default_factory=_default_cost_priority)
    unknown_kaggle_quota: Literal["warn", "skip"] = "warn"
    ccu_rates: dict[str, CcuRate] = Field(default_factory=_default_ccu_rates)

    @field_validator("cost_priority")
    @classmethod
    def _cost_priority_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("cost_priority must not be empty")
        seen: set[str] = set()
        for candidate_id in value:
            if candidate_id in seen:
                raise ValueError(f"duplicate cost_priority id: {candidate_id}")
            seen.add(candidate_id)
            if candidate_id not in CANDIDATES:
                raise ValueError(f"unknown cost_priority id: {candidate_id}")
        return value

    @field_validator("ccu_rates")
    @classmethod
    def _paid_ccu_rate_keys(cls, value: dict[str, CcuRate]) -> dict[str, CcuRate]:
        for rate_key in value:
            base_id = rate_key.removesuffix("+high_mem")
            candidate = CANDIDATES.get(base_id)
            if candidate is None or not candidate.paid:
                raise ValueError(
                    "ccu_rates keys must be paid candidate ids "
                    f"(optionally with +high_mem); got {rate_key}"
                )
            if rate_key.endswith("+high_mem") and not candidate.supports_high_mem:
                raise ValueError(
                    f"ccu_rates key {rate_key} is invalid: "
                    f"{base_id} does not support high_mem"
                )
        return value


def _default_state_dir() -> Path:
    override = os.environ.get("L2KC_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".l2kc"


class Settings(BaseModel):
    """Process settings loaded from YAML.

    Missing ``mode`` means mock. Real GPU work needs both ``mode: real`` and
    ``allow_real_jobs: true``.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "real"] = "mock"
    allow_real_jobs: bool = False
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    mock: MockSettings = Field(default_factory=MockSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    quota: QuotaSettings = Field(default_factory=QuotaSettings)
    jobs: JobsSettings = Field(default_factory=JobsSettings)
    state_dir: Path = Field(default_factory=_default_state_dir)
    # None → ``effective_artifact_root`` uses ``./l2kc-artifacts``.
    artifact_root: Path | None = None
    # None means the key was absent. The registry then synthesizes one implicit
    # ambient account per provider (``<provider>-default``).
    accounts: list[AccountConfig] | None = None
    credentials: dict[str, CredentialRefConfig] = Field(default_factory=dict)

    @field_validator("state_dir")
    @classmethod
    def _expand_state_dir(cls, value: Path) -> Path:
        return Path(value).expanduser()

    @field_validator("artifact_root")
    @classmethod
    def _expand_artifact_root(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    @model_validator(mode="after")
    def _accounts_are_consistent(self) -> Settings:
        for name in self.credentials:
            reject_ref_name(name, "credentials")
        accounts = self.accounts
        if not accounts:
            return self
        seen: set[str] = set()
        defaults: dict[str, int] = {}
        ambient: dict[str, int] = {}
        shared: dict[str, int] = {}
        for account in accounts:
            if account.id in seen:
                raise ConfigError(
                    "duplicate account id",
                    details={"field": "accounts.id"},
                )
            seen.add(account.id)
            provider = account.provider.value
            if account.default:
                defaults[provider] = defaults.get(provider, 0) + 1
                if defaults[provider] > 1:
                    raise ConfigError(
                        "more than one default account for a provider",
                        details={"field": "accounts.default", "provider": provider},
                    )
            if account.credential_ref is None:
                ambient[provider] = ambient.get(provider, 0) + 1
                if ambient[provider] > 1:
                    raise ConfigError(
                        "more than one ambient account for a provider",
                        details={"field": "accounts.credential_ref", "provider": provider},
                    )
            else:
                shared[account.credential_ref] = shared.get(account.credential_ref, 0) + 1
                if shared[account.credential_ref] > 1:
                    raise ConfigError(
                        "accounts must not share a credential_ref",
                        details={"field": "accounts.credential_ref"},
                    )
        _reject_shared_credential_locations(self)
        return self

    @property
    def real_enabled(self) -> bool:
        return self.mode == "real"

    def effective_artifact_root(self) -> Path:
        if self.artifact_root is not None:
            return Path(self.artifact_root).expanduser().resolve()
        return (Path.cwd() / "l2kc-artifacts").resolve()

    def require_real_jobs_allowed(self, provider: str) -> None:
        """Raise when a real call is not explicitly enabled for ``provider``."""
        if self.mode == "real" and self.allow_real_jobs:
            return
        raise RealModeDisabledError(
            "Real calls are disabled. Set mode to 'real' and allow_real_jobs to true.",
            provider=str(provider),
            details={"mode": self.mode, "allow_real_jobs": self.allow_real_jobs},
        )


def _expanded_path(value: str | Path) -> str:
    """Absolute location of a credential path. The text is not for error messages."""
    return str(Path(value).expanduser().resolve())


def _credential_location(
    account: AccountConfig,
    settings: Settings,
) -> tuple[str, ...] | None:
    """Identity of where a credential lives. ``None`` means no shared location.

    Path sources compare expanded paths. Env sources compare the variable name.
    Keyring sources compare service and username. A name that is not declared
    under ``credentials`` uses the convention directory for that provider, so
    it collides with a declared path that expands to the same place.
    """
    ref_name = account.credential_ref
    if ref_name is None:
        return None
    declared = settings.credentials.get(ref_name)
    if declared is None:
        if account.provider is ProviderName.fake:
            return None
        try:
            root = safe_join(settings.state_dir, "credentials", ref_name)
        except PathSecurityError as exc:
            raise ConfigError(
                "credential reference name is not a safe path",
                details={"field": "credential_ref"},
            ) from exc
        if account.provider is ProviderName.kaggle:
            return ("path", _expanded_path(root))
        if account.provider is ProviderName.colab and settings.providers.colab.auth == "oauth2":
            return ("path", _expanded_path(root))
        if account.provider is ProviderName.colab:
            return ("path", _expanded_path(root / "application_default_credentials.json"))
        return None
    if isinstance(declared, KaggleConfigDirCredential | ColabHomeCredential | AdcFileCredential):
        return ("path", _expanded_path(declared.path))
    if isinstance(declared, EnvCredential):
        return ("env", declared.env)
    if isinstance(declared, KeyringCredential):
        return ("keyring", declared.service, declared.username)
    return None


def _reject_shared_credential_locations(settings: Settings) -> None:
    """Reject two accounts whose different ref names resolve to one location.

    The error names the field only. It does not include the path, env var, or
    keyring username.
    """
    accounts = settings.accounts
    if not accounts:
        return
    seen: set[tuple[str, ...]] = set()
    for account in accounts:
        location = _credential_location(account, settings)
        if location is None:
            continue
        if location in seen:
            raise ConfigError(
                "accounts must not share credentials",
                details={"field": "accounts.credential_ref"},
            )
        seen.add(location)


def _reject_secret_keys(
    value: object,
    location: str,
    *,
    credential_entry: bool = False,
) -> None:
    """Reject keys that look like secret material.

    ``credential_ref`` and ``api_key_env`` are references, so they are allowed.
    The top-level ``credentials`` map is allowed. Inside one entry, ``source``,
    ``env``, ``path``, ``service``, and ``username`` are allowed. ``token``,
    ``api_key``, ``password``, and ``secret`` are rejected everywhere, including
    under ``accounts``, ``credentials``, and ``labels``.
    """
    if isinstance(value, dict):
        parent_is_credentials_map = location == "credentials"
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError("config keys must be strings")
            lowered = key.lower()
            here = f"{location}.{key}" if location else key
            allowed = lowered in _ALLOWED_SECRETISH_KEYS
            if location == "" and lowered == "credentials":
                allowed = True
            if credential_entry and lowered in _CREDENTIAL_ENTRY_KEYS:
                allowed = True
            if not allowed and any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
                raise ConfigError(
                    f"Refusing config key '{here}' because it looks like a secret. "
                    "Use credential refs or environment variable names "
                    "(credential_ref, api_key_env), not secret values.",
                    details={"key": here},
                )
            _reject_secret_keys(
                item,
                here,
                credential_entry=parent_is_credentials_map,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_keys(
                item,
                f"{location}[{index}]",
                credential_entry=credential_entry,
            )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"could not read config file: {path}") from exc
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(redact(f"could not parse config: {exc}")) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    return loaded


def _existing_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_file():
        raise ConfigError(f"{label} does not point to a file: {candidate}")
    return candidate


def _search_config_path(explicit: Path | str | None) -> Path | None:
    """Resolve config path.

    Order: explicit argument, ``L2KC_CONFIG``, ``./l2kc.yaml``,
    ``~/.config/l2kc/config.yaml``. An explicit path or ``L2KC_CONFIG`` that
    does not exist is an error. When neither is set, a missing default file
    means "use built-in defaults".
    """
    if explicit is not None:
        return _existing_file(Path(explicit), label="config path")
    env_path = os.environ.get("L2KC_CONFIG")
    if env_path:
        return _existing_file(Path(env_path), label="L2KC_CONFIG")
    candidates = (
        Path.cwd() / "l2kc.yaml",
        Path.home() / ".config" / "l2kc" / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | str | None = None) -> Settings:
    """Load settings from YAML, or defaults when no config file is found."""
    chosen = _search_config_path(path)
    raw: dict[str, Any] = {}
    if chosen is not None:
        raw = _read_yaml(chosen)
    _reject_secret_keys(raw, "")
    state_env = os.environ.get("L2KC_STATE_DIR")
    if state_env:
        raw["state_dir"] = state_env
    # Opt-in undocumented ccu-info: env overrides YAML when truthy.
    ccu_env = os.environ.get("L2KC_COLAB_EXPERIMENTAL_CCU_INFO", "").strip().lower()
    if ccu_env in {"1", "true", "yes", "on"}:
        providers = raw.setdefault("providers", {})
        if not isinstance(providers, dict):
            raise ConfigError("providers must be a mapping")
        colab = providers.setdefault("colab", {})
        if not isinstance(colab, dict):
            raise ConfigError("providers.colab must be a mapping")
        colab["experimental_ccu_info"] = True
    try:
        return Settings.model_validate(raw)
    except ConfigError:
        raise
    except ValidationError as exc:
        fields = [".".join(str(part) for part in error["loc"]) for error in exc.errors()]
        summary = ", ".join(fields) if fields else "unknown field"
        raise ConfigError(
            f"invalid configuration: {summary}",
            details={"fields": fields},
        ) from exc
