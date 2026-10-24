"""Stdio MCP server. Tools call :class:`Client` and return JSON-ready dicts.

The installed MCP SDK is 2.x. ``FastMCP`` was renamed to
``mcp.server.mcpserver.MCPServer``; this module uses that class.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError

from link2kaggle_colab.accounts.credentials import _describe
from link2kaggle_colab.client import Client
from link2kaggle_colab.core.config import Settings, load_config
from link2kaggle_colab.core.errors import InvalidInputError, L2KCError
from link2kaggle_colab.core.models import JobSpec
from link2kaggle_colab.core.paths import resolve_artifact_dest
from link2kaggle_colab.core.redact import redact

_FIXTURE = "Mock data are TEST FIXTURES, not a live account."
_FIXTURE_NOTICE = (
    "TEST FIXTURE: mock quota, balance, and membership are not a live account"
)

TOOL_NAMES = (
    "gpu_list_providers",
    "gpu_list_accounts",
    "gpu_auth_status",
    "gpu_get_quota",
    "gpu_list_resources",
    "gpu_submit_job",
    "gpu_get_job_status",
    "gpu_get_job_logs",
    "gpu_download_artifacts",
    "gpu_cancel_job",
    "gpu_list_jobs",
    "gpu_reconcile_jobs",
    "gpu_recommend_backend",
    "gpu_doctor",
    "gpu_refresh_quota",
)

# Same callables as TOOL_NAMES, registered under the l2kc_* prefix.
ALIAS_NAMES = tuple(f"l2kc_{name.removeprefix('gpu_')}" for name in TOOL_NAMES)

_REMOTE_TRANSPORTS = frozenset({"sse", "http", "streamable-http"})


class _Holder:
    def __init__(self, client: Client | None) -> None:
        self.client = client

    def get(self) -> Client:
        if self.client is None:
            self.client = Client()
        return self.client


def _mode_fields(client: Client) -> dict[str, Any]:
    mode = client.settings.mode
    fields: dict[str, Any] = {"mode": mode, "is_mock": mode == "mock"}
    if mode == "mock":
        fields["fixture_notice"] = _FIXTURE_NOTICE
    return fields


def _failure(client: Client | None, exc: L2KCError) -> dict[str, Any]:
    if client is None:
        body: dict[str, Any] = {"mode": "mock", "is_mock": True, "fixture_notice": _FIXTURE_NOTICE}
    else:
        body = _mode_fields(client)
    body["ok"] = False
    body["error"] = exc.to_dict()
    return body


def _success(client: Client, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.update(_mode_fields(client))
    body["ok"] = True
    return body


def _validation_error(exc: ValidationError) -> InvalidInputError:
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error["loc"]) or "input"
        parts.append(f"{loc}: {error['msg']}")
    summary = "; ".join(parts) if parts else "invalid input"
    return InvalidInputError(redact(summary))


def _invoke(client: Client, func: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        payload = func()
    except L2KCError as exc:
        return _failure(client, exc)
    except ValidationError as exc:
        return _failure(client, _validation_error(exc))
    except Exception as exc:
        text = str(exc).strip() or type(exc).__name__
        return _failure(
            client,
            L2KCError(
                redact(text),
                code="internal_error",
                details={"exception_type": type(exc).__name__},
            ),
        )
    return _success(client, payload)


def gpu_list_providers(client: Client) -> dict[str, Any]:
    """List providers. Mock data are TEST FIXTURES, not a live account."""
    return _invoke(client, lambda: {"providers": client.list_providers()})


def gpu_list_accounts(client: Client) -> dict[str, Any]:
    """List accounts. Mock data are TEST FIXTURES, not a live account.

    Credential values are not included. Quota exhaustion does not rotate to
    another account of the same provider.
    """

    def _load() -> dict[str, Any]:
        return {"accounts": client.list_accounts()}

    return _invoke(client, _load)


def gpu_auth_status(
    client: Client,
    provider: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Auth snapshots. Mock data are TEST FIXTURES, not a live account.

    ``account`` reads that account only. There is no automatic rotation.
    """

    def _load() -> dict[str, Any]:
        rows = [
            item.model_dump(mode="json") for item in client.auth_status(provider, account=account)
        ]
        return {"statuses": rows}

    return _invoke(client, _load)


def gpu_get_quota(
    client: Client,
    provider: str | None = None,
    account: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Quota snapshots. Mock data are TEST FIXTURES, not a live account.

    ``account`` reads that account only. There is no automatic rotation.
    ``refresh`` drops the per-account quota cache entry before reading.
    """

    def _load() -> dict[str, Any]:
        rows = [
            item.model_dump(mode="json")
            for item in client.get_quota(provider, account=account, refresh=refresh)
        ]
        return {"quotas": rows}

    return _invoke(client, _load)


def gpu_refresh_quota(
    client: Client,
    provider: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Refresh quota snapshots. Mock data are TEST FIXTURES, not a live account.

    Equivalent to ``gpu_get_quota(..., refresh=True)``.
    """

    return gpu_get_quota(client, provider=provider, account=account, refresh=True)


def _eligible_flag(gpu_type: str | None, eligible_gpus: list[str] | None) -> bool | None:
    """True/False when the account publishes eligible GPUs; otherwise null."""
    if eligible_gpus is None:
        return None
    if gpu_type is None:
        return None
    return gpu_type in eligible_gpus


def _quota_eligible_map(
    client: Client,
    provider: str | None,
    account: str | None,
) -> dict[tuple[str, str | None], list[str] | None]:
    """Map ``(provider, account_id)`` to ``eligible_gpus`` (refresh=False)."""
    mapping: dict[tuple[str, str | None], list[str] | None] = {}
    for item in client.get_quota(provider, account=account, refresh=False):
        mapping[(str(item.provider), item.account_id)] = item.eligible_gpus
    return mapping


def gpu_list_resources(
    client: Client,
    provider: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Accelerators. Memory is per GPU. Mock data are TEST FIXTURES, not a live account.

    ``account`` lists that account only. There is no automatic rotation.
    When quota publishes ``eligible_gpus``, each row includes ``eligible``.
    """

    def _load() -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        if account is not None:
            items = client.list_resources(provider, account=account)
            owner = str(client.get_account(account)["provider"])
            for item in items:
                dumped = item.model_dump(mode="json")
                dumped["provider"] = owner
                dumped["account_id"] = account
                rows.append(dumped)
        else:
            if provider is None:
                names = [str(row["name"]) for row in client.list_providers() if row["enabled"]]
            else:
                names = [provider]
            defaults = {
                str(row["provider"]): str(row["id"])
                for row in client.list_accounts()
                if row.get("is_default")
            }
            for name in names:
                for item in client.list_resources(name):
                    dumped = item.model_dump(mode="json")
                    dumped["provider"] = name
                    dumped["account_id"] = defaults.get(name)
                    rows.append(dumped)
        eligible_map = _quota_eligible_map(client, provider, account)
        for dumped in rows:
            key = (str(dumped.get("provider")), dumped.get("account_id"))
            dumped["eligible"] = _eligible_flag(
                dumped.get("gpu_type") if isinstance(dumped.get("gpu_type"), str) else None,
                eligible_map.get(key),
            )
        return {"resources": rows}

    return _invoke(client, _load)


def gpu_submit_job(
    client: Client,
    script_path: str,
    provider: str = "auto",
    gpu_type: str | None = None,
    gpu_count: int = 1,
    timeout_seconds: int | None = None,
    name: str | None = None,
    allow_premium: bool = False,
    high_mem: bool = False,
    min_gpu_memory_gb_per_gpu: float | None = None,
    idempotency_key: str | None = None,
    account: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a script. Mock data are TEST FIXTURES, not a live account.

    ``provider`` ``auto`` asks the scheduler to choose. A premium GPU needs
    ``scheduler.allow_premium_gpu`` plus ``allow_premium``. ``high_mem`` maps
    to Colab ``--high-mem`` (CPU/T4/A100/H100/G4; not L4/TPU) and does not
    bypass premium guards. Colab needs
    ``timeout_seconds`` so the CCU budget can be checked. Real jobs also
    require ``mode: real`` and ``allow_real_jobs``. ``idempotency_key`` returns
    the existing job instead of submitting again when the spec matches. A
    different spec is an invalid-input error. ``deduplicated`` is true when
    the existing job is returned. An ambiguous submit is not resubmitted.
    ``account`` pins that account. Quota exhaustion does not rotate to
    another account. ``dry_run`` recommends a backend and does not submit.
    """

    def _load() -> dict[str, Any]:
        if dry_run:
            recommendation = client.recommend_backend(
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                min_gpu_memory_gb_per_gpu=min_gpu_memory_gb_per_gpu,
                allow_premium=allow_premium,
                timeout_seconds=timeout_seconds,
                provider=provider,
                high_mem=high_mem,
                account=account,
            )
            if account is not None:
                recommendation["account_id"] = account
            return {
                "dry_run": True,
                "recommendation": recommendation,
                "estimated_ccu": recommendation.get("estimated_ccu"),
            }
        chosen = None if str(provider).strip().lower() == "auto" else provider
        payload: dict[str, Any] = {
            "script_path": script_path,
            "provider": chosen,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "timeout_seconds": timeout_seconds,
            "name": name,
            "allow_premium": allow_premium,
            "high_mem": high_mem,
            "min_gpu_memory_gb_per_gpu": min_gpu_memory_gb_per_gpu,
        }
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        spec = JobSpec.model_validate(payload)
        details = client.submit_detailed(spec, account=account)
        job = details.handle.model_dump(mode="json")
        return {
            "job": job,
            "decision": None if details.decision is None else details.decision.to_dict(),
            "state": details.state.value,
            "deduplicated": not details.created,
        }

    return _invoke(client, _load)


def gpu_get_job_status(client: Client, job_id: str) -> dict[str, Any]:
    """Job state. Mock data are TEST FIXTURES, not a live account."""

    def _load() -> dict[str, Any]:
        job = client.status(job_id).model_dump(mode="json")
        return {"job": job}

    return _invoke(client, _load)


def gpu_get_job_logs(client: Client, job_id: str, max_chars: int = 20000) -> dict[str, Any]:
    """Tail of the job log. Mock data are TEST FIXTURES, not a live account."""

    def _load() -> dict[str, Any]:
        if max_chars < 0:
            raise InvalidInputError("max_chars must be >= 0")
        text = client.logs(job_id)
        if max_chars == 0:
            tail = ""
        elif len(text) > max_chars:
            tail = text[-max_chars:]
        else:
            tail = text
        return {
            "job_id": job_id,
            "account_id": _stored_account_id(client, job_id),
            "logs": tail,
            "truncated": len(text) > max_chars,
        }

    return _invoke(client, _load)


def gpu_download_artifacts(
    client: Client,
    job_id: str,
    dest_dir: str | None = None,
) -> dict[str, Any]:
    """Download artifacts under ``artifact_root``. Mock data are TEST FIXTURES."""

    def _load() -> dict[str, Any]:
        requested = Path(dest_dir) if dest_dir else None
        target = resolve_artifact_dest(
            client.settings.effective_artifact_root(),
            requested,
            job_id,
        )
        artifacts = client.download(job_id, requested)
        return {
            "job_id": job_id,
            "account_id": _stored_account_id(client, job_id),
            "dest_dir": str(target),
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
        }

    return _invoke(client, _load)


def gpu_cancel_job(client: Client, job_id: str) -> dict[str, Any]:
    """Cancel a job. Mock data are TEST FIXTURES, not a live account.

    Kaggle's official CLI cannot cancel a running kernel.
    """

    def _load() -> dict[str, Any]:
        job = client.cancel(job_id).model_dump(mode="json")
        return {"job": job}

    return _invoke(client, _load)


def gpu_list_jobs(
    client: Client,
    state: list[str] | None = None,
    provider: str | None = None,
    limit: int | None = 50,
) -> dict[str, Any]:
    """List stored jobs. Mock data are TEST FIXTURES, not a live account.

    This does not call a provider and does not submit. ``limit`` defaults to
    50. Pass null to return every match.
    """

    def _load() -> dict[str, Any]:
        return {"jobs": client.list_jobs(states=state, provider=provider, limit=limit)}

    return _invoke(client, _load)


def gpu_reconcile_jobs(
    client: Client,
    job_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Poll job status. Mock data are TEST FIXTURES, not a live account.

    Omit ``job_ids`` to poll every active job, including ``UNKNOWN``.
    Reconcile never submits and never resubmits an ambiguous job.
    """

    def _load() -> dict[str, Any]:
        return {"reports": client.reconcile_jobs(job_ids)}

    return _invoke(client, _load)


def gpu_recommend_backend(
    client: Client,
    gpu_type: str | None = None,
    gpu_count: int = 1,
    min_gpu_memory_gb_per_gpu: float | None = None,
    allow_premium: bool = False,
    timeout_seconds: int | None = None,
    provider: str = "auto",
    account: str | None = None,
    high_mem: bool = False,
) -> dict[str, Any]:
    """Recommend a backend without submitting. Mock data are TEST FIXTURES, not a live account.

    A premium GPU needs ``scheduler.allow_premium_gpu`` plus ``allow_premium``.
    Colab needs ``timeout_seconds`` so the CCU budget can be checked.
    ``provider`` ``auto`` lets the scheduler choose. ``account`` pins that
    account and does not rotate on quota. Per-GPU memory is never summed
    across devices.
    """

    def _load() -> dict[str, Any]:
        decision = client.recommend_backend(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            min_gpu_memory_gb_per_gpu=min_gpu_memory_gb_per_gpu,
            allow_premium=allow_premium,
            timeout_seconds=timeout_seconds,
            provider=provider,
            high_mem=high_mem,
            account=account,
        )
        if account is not None:
            decision["account_id"] = account
        return {"recommendation": decision}

    return _invoke(client, _load)


def gpu_doctor(client: Client) -> dict[str, Any]:
    """Local doctor report. Mock data are TEST FIXTURES, not a live account.

    Does not execute ``kaggle`` or ``colab``. CLI presence is ``shutil.which``.
    """

    def _load() -> dict[str, Any]:
        return {"report": client.doctor()}

    return _invoke(client, _load)


def build_server(client: Client | None = None) -> MCPServer:
    """Build an MCP server that shares ``client``. A missing client is created on first use.

    Registered callables keep explicit parameters so the MCP schema matches the
    module-level tools. Those module-level functions are what tests call.
    """
    holder = _Holder(client)
    server = MCPServer(
        "link2kaggle-colab",
        instructions=(
            "Local bridge to Kaggle and Google Colab. Default mode is mock. "
            "Real GPU jobs require mode real and allow_real_jobs. "
            "A premium GPU needs scheduler.allow_premium_gpu plus allow_premium. "
            "Colab needs timeout_seconds so the CCU budget can be checked. "
            "Pass account to pin one account. Quota exhaustion does not rotate. "
            f"{_FIXTURE}"
        ),
    )

    def _call(func: Callable[..., dict[str, Any]], /, *args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            current = holder.get()
        except L2KCError as exc:
            return _failure(None, exc)
        except Exception as exc:
            text = str(exc).strip() or type(exc).__name__
            return _failure(None, L2KCError(redact(text), code="internal_error"))
        return func(current, *args, **kwargs)

    def list_providers_tool() -> dict[str, Any]:
        return _call(gpu_list_providers)

    def list_accounts_tool() -> dict[str, Any]:
        return _call(gpu_list_accounts)

    def auth_status_tool(
        provider: str | None = None,
        account: str | None = None,
    ) -> dict[str, Any]:
        return _call(gpu_auth_status, provider, account)

    def quota_tool(
        provider: str | None = None,
        account: str | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return _call(gpu_get_quota, provider, account, refresh)

    def refresh_quota_tool(
        provider: str | None = None,
        account: str | None = None,
    ) -> dict[str, Any]:
        return _call(gpu_refresh_quota, provider, account)

    def resources_tool(provider: str | None = None, account: str | None = None) -> dict[str, Any]:
        return _call(gpu_list_resources, provider, account)

    def submit_tool(
        script_path: str,
        provider: str = "auto",
        gpu_type: str | None = None,
        gpu_count: int = 1,
        timeout_seconds: int | None = None,
        name: str | None = None,
        allow_premium: bool = False,
        high_mem: bool = False,
        min_gpu_memory_gb_per_gpu: float | None = None,
        idempotency_key: str | None = None,
        account: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return _call(
            gpu_submit_job,
            script_path,
            provider,
            gpu_type,
            gpu_count,
            timeout_seconds,
            name,
            allow_premium,
            high_mem,
            min_gpu_memory_gb_per_gpu,
            idempotency_key,
            account,
            dry_run,
        )

    def status_tool(job_id: str) -> dict[str, Any]:
        return _call(gpu_get_job_status, job_id)

    def logs_tool(job_id: str, max_chars: int = 20000) -> dict[str, Any]:
        return _call(gpu_get_job_logs, job_id, max_chars)

    def download_tool(job_id: str, dest_dir: str | None = None) -> dict[str, Any]:
        return _call(gpu_download_artifacts, job_id, dest_dir)

    def cancel_tool(job_id: str) -> dict[str, Any]:
        return _call(gpu_cancel_job, job_id)

    def list_jobs_tool(
        state: list[str] | None = None,
        provider: str | None = None,
        limit: int | None = 50,
    ) -> dict[str, Any]:
        return _call(gpu_list_jobs, state, provider, limit)

    def reconcile_jobs_tool(job_ids: list[str] | None = None) -> dict[str, Any]:
        return _call(gpu_reconcile_jobs, job_ids)

    def recommend_tool(
        gpu_type: str | None = None,
        gpu_count: int = 1,
        min_gpu_memory_gb_per_gpu: float | None = None,
        allow_premium: bool = False,
        timeout_seconds: int | None = None,
        provider: str = "auto",
        account: str | None = None,
        high_mem: bool = False,
    ) -> dict[str, Any]:
        return _call(
            gpu_recommend_backend,
            gpu_type,
            gpu_count,
            min_gpu_memory_gb_per_gpu,
            allow_premium,
            timeout_seconds,
            provider,
            account,
            high_mem,
        )

    def doctor_tool() -> dict[str, Any]:
        return _call(gpu_doctor)

    registrations = (
        ("gpu_list_providers", list_providers_tool, gpu_list_providers),
        ("gpu_list_accounts", list_accounts_tool, gpu_list_accounts),
        ("gpu_auth_status", auth_status_tool, gpu_auth_status),
        ("gpu_get_quota", quota_tool, gpu_get_quota),
        ("gpu_list_resources", resources_tool, gpu_list_resources),
        ("gpu_submit_job", submit_tool, gpu_submit_job),
        ("gpu_get_job_status", status_tool, gpu_get_job_status),
        ("gpu_get_job_logs", logs_tool, gpu_get_job_logs),
        ("gpu_download_artifacts", download_tool, gpu_download_artifacts),
        ("gpu_cancel_job", cancel_tool, gpu_cancel_job),
        ("gpu_list_jobs", list_jobs_tool, gpu_list_jobs),
        ("gpu_reconcile_jobs", reconcile_jobs_tool, gpu_reconcile_jobs),
        ("gpu_recommend_backend", recommend_tool, gpu_recommend_backend),
        ("gpu_doctor", doctor_tool, gpu_doctor),
        ("gpu_refresh_quota", refresh_quota_tool, gpu_refresh_quota),
    )
    for name, func, documented in registrations:
        _add(server, name, func, documented)
        alias = f"l2kc_{name.removeprefix('gpu_')}"
        _add(server, alias, func, documented)
    return server


def _add(
    server: MCPServer,
    name: str,
    func: Callable[..., dict[str, Any]],
    documented: Callable[..., dict[str, Any]],
) -> None:
    description = (documented.__doc__ or "").strip() or _FIXTURE
    if "TEST FIXTURE" not in description:
        description = f"{description} {_FIXTURE}"
    try:
        server.add_tool(func, name=name, description=description, structured_output=True)
    except Exception:
        # Plain dict returns need structured_output=False on some SDK builds.
        server.add_tool(func, name=name, description=description, structured_output=False)


def _stored_account_id(client: Client, job_id: str) -> str | None:
    """Account stored on ``job_id`` when the tool does not return a job model.

    Submit, status, and cancel read ``account_id`` from the handle or result.
    Logs and download still use this lookup. It does not call a provider.
    """
    for row in client.list_jobs():
        if row.get("job_id") == job_id:
            value = row.get("account_id")
            return value if isinstance(value, str) else None
    return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _mcp_settings() -> Settings:
    """Load config and disable fake local exec unless explicitly allowed."""
    settings = load_config()
    allow_exec = _env_flag("L2KC_ALLOW_FAKE_EXEC")
    fake = settings.providers.fake.model_copy(update={"allow_exec": allow_exec})
    providers = settings.providers.model_copy(update={"fake": fake})
    return settings.model_copy(update={"providers": providers})


def _credential_home_label(client: Client, account_id: str) -> str:
    """Return a non-secret credential home/path label for startup logs."""
    try:
        account = client._accounts.get(account_id)
        ref_name = account.credential_ref
        ref_config = client.settings.credentials.get(ref_name) if ref_name else None
        view = _describe(ref_name, ref_config, account.provider, client.settings)
        if view.home is not None:
            return str(view.home)
        if view.path is not None:
            return str(view.path)
        if view.kind == "env" and view.env_name:
            return f"env:{view.env_name}"
        if view.kind == "keyring" and view.service:
            return f"keyring:{view.service}"
        return view.kind
    except Exception:
        return "?"


def _log_mcp_startup(client: Client, settings: Settings) -> None:
    """One stderr line so stdio MCP stays clean. Never prints secrets."""
    account_rows = client.list_accounts()
    account_ids = ",".join(account["id"] for account in account_rows) or "-"
    homes: list[str] = []
    for account in account_rows:
        account_id = str(account["id"])
        homes.append(f"{account_id}->{_credential_home_label(client, account_id)}")
    credential_homes = ",".join(homes) or "-"
    config_env = os.environ.get("L2KC_CONFIG") or "-"
    print(
        "l2kc-mcp:"
        f" mode={settings.mode}"
        f" allow_real_jobs={settings.allow_real_jobs}"
        f" L2KC_CONFIG={config_env}"
        f" state_dir={settings.state_dir}"
        f" artifact_root={settings.effective_artifact_root()}"
        f" fake.allow_exec={settings.providers.fake.allow_exec}"
        f" accounts={account_ids}"
        f" credential_homes={credential_homes}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the link2kaggle-colab MCP server on stdio with a pinned Client."""
    parser = argparse.ArgumentParser(prog="link2kaggle-colab-mcp")
    parser.add_argument(
        "--transport",
        default="stdio",
        help="MCP transport. Only stdio is enabled.",
    )
    args = parser.parse_args(argv)
    transport = str(args.transport).strip().lower()
    if transport in _REMOTE_TRANSPORTS:
        print(
            "error: remote MCP transport "
            f"{transport!r} is disabled until auth+sandbox are in place; "
            "use --transport stdio",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if transport != "stdio":
        print(
            f"error: unsupported MCP transport {transport!r}; only stdio is enabled",
            file=sys.stderr,
        )
        raise SystemExit(2)
    settings = _mcp_settings()
    client = Client(settings)
    _log_mcp_startup(client, settings)
    build_server(client).run(transport="stdio")
