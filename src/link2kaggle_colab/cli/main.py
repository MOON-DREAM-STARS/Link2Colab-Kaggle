"""Command-line interface. Each command constructs a Client and prints its result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ValidationError

from link2kaggle_colab import __version__
from link2kaggle_colab.accounts.credentials import credential_health
from link2kaggle_colab.client import Client
from link2kaggle_colab.core.errors import InvalidInputError, L2KCError, SubmitOutcomeUnknownError
from link2kaggle_colab.core.models import GpuType, JobSpec, ProviderName
from link2kaggle_colab.core.paths import resolve_artifact_dest
from link2kaggle_colab.core.redact import redact

app = typer.Typer(no_args_is_help=True, add_completion=False)
auth_app = typer.Typer(no_args_is_help=True, add_completion=False)
jobs_app = typer.Typer(no_args_is_help=True, add_completion=False)
accounts_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(auth_app, name="auth")
app.add_typer(jobs_app, name="jobs")
app.add_typer(accounts_app, name="accounts")

MOCK_BANNER = (
    "MOCK MODE: quota, balance, and membership values are TEST FIXTURES, not a live account."
)

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help="YAML config file. Overrides L2KC_CONFIG."),
]
JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Print machine-readable JSON on stdout."),
]
AccountOpt = Annotated[
    str | None,
    typer.Option(
        "--account",
        help="Account id. Pins that account. Quota exhaustion does not rotate.",
    ),
]
_FIXTURE_ACCOUNT_NOTE = (
    "TEST FIXTURE: mock auth, quota, and membership are not a live account."
)


@app.callback()
def cli(
    ctx: typer.Context,
    config: ConfigOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Link2Kaggle&Colab command-line interface.

    Default mode is mock. Real GPU jobs need mode real and allow_real_jobs.
    Global options can also be placed before the command: ``l2kc --json doctor``.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["json"] = json_output


@auth_app.callback()
def auth_group() -> None:
    """Authentication snapshots. Mock results are TEST FIXTURES."""


@app.command()
def version(ctx: typer.Context) -> None:
    """Print the package version."""
    _execute(ctx, lambda _client: {"version": __version__}, _print_version)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Show mode, state directory, CLI presence, and auth. Does not run CLIs."""
    _execute(ctx, lambda client: client.doctor(), _print_doctor)


@auth_app.command("status")
def auth_status(
    ctx: typer.Context,
    provider: Annotated[
        ProviderName | None,
        typer.Option("--provider", help="kaggle, colab, or fake. Default: all enabled."),
    ] = None,
    account: AccountOpt = None,
) -> None:
    """Show authentication state for the selected account."""
    _execute(
        ctx,
        lambda client: _dump(client.auth_status(provider, account=account)),
        _print_auth,
    )


@app.command("providers")
def providers_command(ctx: typer.Context) -> None:
    """List providers, whether they are enabled, and what they support."""
    _execute(ctx, lambda client: client.list_providers(), _print_providers)


@app.command()
def quota(
    ctx: typer.Context,
    provider: Annotated[
        ProviderName | None,
        typer.Option("--provider", help="kaggle, colab, or fake. Default: all enabled."),
    ] = None,
    account: AccountOpt = None,
) -> None:
    """Show quota for the selected account. Mock numbers are TEST FIXTURES."""
    _execute(
        ctx,
        lambda client: _dump(client.get_quota(provider, account=account)),
        _print_quota,
    )


@app.command()
def resources(
    ctx: typer.Context,
    provider: Annotated[
        ProviderName | None,
        typer.Option("--provider", help="kaggle, colab, or fake. Default: all enabled."),
    ] = None,
    account: AccountOpt = None,
) -> None:
    """List accelerators for the selected account. Memory is per GPU."""

    def _action(client: Client) -> list[dict[str, Any]]:
        if account is not None:
            items = client.list_resources(provider, account=account)
            owner = str(client.get_account(account)["provider"])
            rows: list[dict[str, Any]] = []
            for resource in items:
                dumped = resource.model_dump(mode="json")
                dumped["provider"] = owner
                dumped["account_id"] = account
                rows.append(dumped)
            return rows
        if provider is None:
            names = [
                ProviderName(row["name"])
                for row in client.list_providers()
                if row["enabled"]
            ]
        else:
            names = [provider]
        defaults = _default_account_ids(client)
        rows = []
        for name in names:
            for resource in client.list_resources(name):
                dumped = resource.model_dump(mode="json")
                dumped["provider"] = name.value
                dumped["account_id"] = defaults.get(name.value)
                rows.append(dumped)
        return rows

    _execute(ctx, _action, _print_resources)


@app.command(context_settings={"allow_extra_args": True})
def submit(
    ctx: typer.Context,
    file: Annotated[Path, typer.Argument(help="Local .py or .ipynb script.")],
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            help="auto, kaggle, colab, or fake. auto uses the scheduler.",
        ),
    ] = "auto",
    gpu: Annotated[
        GpuType | None,
        typer.Option("--gpu", help="Accelerator (T4, L4, A100, TPU_V5E1, TPU_V6E1, …). NONE=CPU."),
    ] = None,
    gpu_count: Annotated[int, typer.Option("--gpu-count", help="GPU count. 0 requests CPU.")] = 1,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Timeout in seconds. Required for a Colab CCU budget."),
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Job slug.")] = None,
    allow_premium: Annotated[
        bool,
        typer.Option(
            "--allow-premium",
            "--confirm-premium",
            help=(
                "per-job confirmation for a premium GPU/TPU; "
                "also requires scheduler.allow_premium_gpu: true"
            ),
        ),
    ] = False,
    high_mem: Annotated[
        bool,
        typer.Option(
            "--high-mem",
            help=(
                "Colab high-RAM shape (--high-mem). "
                "Supported for CPU/T4/A100/H100/G4; not L4/TPU. "
                "Does not bypass premium guards."
            ),
        ),
    ] = False,
    min_gpu_memory_gb: Annotated[
        float | None,
        typer.Option(
            "--min-gpu-memory-gb",
            help="Minimum GB on each GPU. Memory is not summed across GPUs.",
        ),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        typer.Option(
            "--idempotency-key",
            help="Return the existing job when this key and spec were already submitted.",
        ),
    ] = None,
    account: AccountOpt = None,
) -> None:
    """Submit a script. Arguments after -- are passed to the script."""

    def _action(client: Client) -> dict[str, Any]:
        script_args = _script_args(ctx.args)
        try:
            spec = JobSpec(
                script_path=file,
                args=script_args,
                provider=_provider_choice(provider, allow_fake=True),
                gpu_type=gpu,
                gpu_count=gpu_count,
                timeout_seconds=timeout,
                name=name,
                allow_premium=allow_premium,
                high_mem=high_mem,
                min_gpu_memory_gb_per_gpu=min_gpu_memory_gb,
                idempotency_key=idempotency_key,
            )
        except ValidationError as exc:
            raise _validation_error(exc) from exc
        details = client.submit_detailed(spec, account=account)
        job = details.handle.model_dump(mode="json")
        return {
            "job": job,
            "decision": None if details.decision is None else details.decision.to_dict(),
            "state": details.state.value,
            "deduplicated": not details.created,
        }

    _execute(ctx, _action, _print_submit)


@app.command()
def status(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Job id returned by submit.")],
) -> None:
    """Show the state of a job. Uses the account stored on the job."""

    def _action(client: Client) -> dict[str, Any]:
        return client.status(job_id).model_dump(mode="json")

    _execute(ctx, _action, _print_job)


@app.command()
def logs(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Job id returned by submit.")],
) -> None:
    """Print logs for a job. Uses the account stored on the job."""

    def _action(client: Client) -> dict[str, Any]:
        text = client.logs(job_id)
        return {
            "job_id": job_id,
            "account_id": _stored_account_id(client, job_id),
            "logs": text,
        }

    def _human(payload: dict[str, Any]) -> None:
        account_id = payload.get("account_id")
        if account_id:
            typer.echo(f"account_id: {account_id}", err=True)
        text = str(payload["logs"])
        typer.echo(text, nl=not text.endswith("\n"))

    _execute(ctx, _action, _human)


@app.command()
def download(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Job id returned by submit.")],
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Directory for artifacts. Default: ./l2kc-artifacts/<job_id>."),
    ] = None,
) -> None:
    """Download job artifacts. Uses the account stored on the job."""

    def _action(client: Client) -> dict[str, Any]:
        artifacts = client.download(job_id, dest)
        target = resolve_artifact_dest(
            client.settings.effective_artifact_root(),
            dest,
            job_id,
        )
        return {
            "job_id": job_id,
            "account_id": _stored_account_id(client, job_id),
            "dest_dir": str(target),
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
        }

    _execute(ctx, _action, _print_artifacts)


@app.command()
def cancel(
    ctx: typer.Context,
    job_id: Annotated[str, typer.Argument(help="Job id returned by submit.")],
) -> None:
    """Cancel a job. Uses the account stored on the job."""

    def _action(client: Client) -> dict[str, Any]:
        return client.cancel(job_id).model_dump(mode="json")

    _execute(ctx, _action, _print_job)


@jobs_app.callback()
def jobs_group() -> None:
    """Stored jobs. Reconcile polls status and does not submit."""


@jobs_app.command("list")
def jobs_list(
    ctx: typer.Context,
    state: Annotated[
        list[str] | None,
        typer.Option("--state", help="State to include. Repeat for more than one."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="kaggle, colab, or fake."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Maximum rows. Omit to return every match."),
    ] = None,
    json_output: JsonOpt = False,
) -> None:
    """List stored jobs. This does not call a provider."""
    _use_json(ctx, json_output)

    def _action(client: Client) -> list[dict[str, Any]]:
        return client.list_jobs(states=state, provider=provider, limit=limit)

    _execute(ctx, _action, _print_job_rows)


@jobs_app.command("reconcile")
def jobs_reconcile(
    ctx: typer.Context,
    job_ids: Annotated[
        list[str] | None,
        typer.Argument(help="Job ids. Omit to poll every active job."),
    ] = None,
    json_output: JsonOpt = False,
) -> None:
    """Poll job status. Active jobs, including UNKNOWN, are not resubmitted."""
    _use_json(ctx, json_output)

    def _action(client: Client) -> list[dict[str, Any]]:
        return client.reconcile_jobs(job_ids or None)

    _execute(ctx, _action, _print_reconcile, failure_code=_reconcile_exit)


@app.command()
def recommend(
    ctx: typer.Context,
    gpu: Annotated[
        GpuType | None,
        typer.Option("--gpu", help="Accelerator type. NONE requests CPU."),
    ] = None,
    gpu_count: Annotated[int, typer.Option("--gpu-count", min=0)] = 1,
    min_gpu_memory_gb: Annotated[
        float | None,
        typer.Option(
            "--min-gpu-memory-gb",
            help="Minimum GB on each GPU. Memory is not summed across GPUs.",
        ),
    ] = None,
    allow_premium: Annotated[bool, typer.Option("--allow-premium")] = False,
    high_mem: Annotated[
        bool,
        typer.Option(
            "--high-mem",
            help="Colab high-RAM shape. Not L4/TPU. Does not bypass premium guards.",
        ),
    ] = False,
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Timeout in seconds. Required for a Colab CCU budget."),
    ] = None,
    provider: Annotated[
        str,
        typer.Option("--provider", help="auto, kaggle, or colab. auto uses the scheduler."),
    ] = "auto",
    account: AccountOpt = None,
) -> None:
    """Recommend a backend. This does not submit a job or rotate accounts."""

    def _action(client: Client) -> dict[str, Any]:
        payload = client.recommend_backend(
            gpu_type=gpu,
            gpu_count=gpu_count,
            min_gpu_memory_gb_per_gpu=min_gpu_memory_gb,
            allow_premium=allow_premium,
            timeout_seconds=timeout,
            provider=_provider_choice(provider, allow_fake=False),
            high_mem=high_mem,
            account=account,
        )
        chosen = payload.get("provider")
        if account is not None:
            payload["account_id"] = account
        elif isinstance(chosen, str):
            payload["account_id"] = _default_account_ids(client).get(chosen)
        return payload

    _execute(ctx, _action, _print_recommendation)


@accounts_app.callback()
def accounts_group() -> None:
    """Accounts. Selection does not rotate when quota or auth fails."""


@accounts_app.command("list")
def accounts_list(ctx: typer.Context, json_output: JsonOpt = False) -> None:
    """List accounts. Credential columns are source kind and presence, never values."""
    _use_json(ctx, json_output)
    _execute(ctx, _accounts_list_payload, _print_accounts)


@accounts_app.command("show")
def accounts_show(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Argument(help="Account id.")],
    json_output: JsonOpt = False,
) -> None:
    """Show one account: health, labels, and credential presence."""
    _use_json(ctx, json_output)

    def _action(client: Client) -> dict[str, Any]:
        return _account_show_payload(client, account_id)

    _execute(ctx, _action, _print_account_show)


@accounts_app.command("enable")
def accounts_enable(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Argument(help="Account id.")],
    json_output: JsonOpt = False,
) -> None:
    """Enable an account and persist the override.

    Exit status is 2 when ``providers.<provider>.enabled`` is false, so the
    account stays disabled after the override is saved.
    """
    _use_json(ctx, json_output)
    _execute(
        ctx,
        lambda client: _account_change(client, account_id, enabled=True),
        _print_account_change,
        failure_code=_enable_exit,
    )


@accounts_app.command("disable")
def accounts_disable(
    ctx: typer.Context,
    account_id: Annotated[str, typer.Argument(help="Account id.")],
    json_output: JsonOpt = False,
) -> None:
    """Disable an account and persist the override."""
    _use_json(ctx, json_output)
    _execute(
        ctx,
        lambda client: _account_change(client, account_id, enabled=False),
        _print_account_change,
    )


def _flags(ctx: typer.Context) -> tuple[Path | None, bool]:
    current: typer.Context | None = ctx
    data: dict[str, Any] = {}
    while current is not None:
        if isinstance(current.obj, dict):
            data = current.obj
            break
        current = current.parent
    config = data.get("config")
    as_json = bool(data.get("json"))
    return config if isinstance(config, Path) else None, as_json


def _use_json(ctx: typer.Context, enabled: bool) -> None:
    """Turn on JSON output when a subcommand passes ``--json``."""
    if not enabled:
        return
    ctx.ensure_object(dict)
    ctx.obj["json"] = True


def _execute(
    ctx: typer.Context,
    action: Any,
    human: Any,
    *,
    failure_code: Any = None,
) -> None:
    config_path, as_json = _flags(ctx)
    try:
        client = Client(config_path=config_path)
        if client.settings.mode == "mock":
            typer.echo(MOCK_BANNER, err=True)
        payload = action(client)
    except SubmitOutcomeUnknownError as exc:
        _print_unknown_submit(exc)
        raise typer.Exit(code=2) from exc
    except L2KCError as exc:
        typer.echo(f"error[{exc.code}]: {exc.message}", err=True)
        if as_json:
            typer.echo(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        typer.echo(f"error[internal_error]: {redact(str(exc))}", err=True)
        raise typer.Exit(code=1) from exc
    ready = _dump(payload)
    if as_json:
        typer.echo(json.dumps(ready, indent=2, sort_keys=True))
    else:
        human(ready)
    if failure_code is not None:
        code = failure_code(ready)
        if code:
            raise typer.Exit(code=code)


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _provider_choice(value: str, *, allow_fake: bool) -> ProviderName | None:
    """Map a CLI provider flag to a model value. ``auto`` is ``None``."""
    text = value.strip().lower()
    if text == "auto":
        return None
    allowed = "auto, kaggle, colab, or fake" if allow_fake else "auto, kaggle, or colab"
    if text == "fake" and not allow_fake:
        raise InvalidInputError(
            f"provider must be {allowed}",
            details={"provider": value},
        )
    try:
        return ProviderName(text)
    except ValueError as exc:
        raise InvalidInputError(
            f"provider must be {allowed}",
            details={"provider": value},
        ) from exc


def _validation_error(exc: ValidationError) -> InvalidInputError:
    parts: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(item) for item in error["loc"]) or "input"
        parts.append(f"{loc}: {error['msg']}")
    summary = "; ".join(parts) if parts else "invalid input"
    return InvalidInputError(redact(summary))


def _script_args(raw: list[str]) -> list[str]:
    args = list(raw)
    if args and args[0] == "--":
        return args[1:]
    return args


def _print_version(payload: dict[str, Any]) -> None:
    typer.echo(payload["version"])


def _print_doctor(payload: dict[str, Any]) -> None:
    typer.echo(f"mode: {payload['mode']}")
    typer.echo(f"allow_real_jobs: {str(payload['allow_real_jobs']).lower()}")
    typer.echo(f"state_dir: {payload['state_dir']}")
    typer.echo(f"state_dir_writable: {str(payload['state_dir_writable']).lower()}")
    typer.echo(f"python_version: {payload['python_version']}")
    if payload.get("fixture_notice"):
        typer.echo(payload["fixture_notice"])
    for row in payload["providers"]:
        cli = row["cli"]
        found = cli.get("found")
        auth = row.get("auth") or {}
        if "error" in auth:
            auth_text = f"error {auth['error'].get('code')}"
        elif auth:
            auth_text = f"{auth.get('state')} {auth.get('message', '')}".strip()
        else:
            auth_text = "disabled"
        typer.echo(
            f"provider {row['name']} enabled={str(row['enabled']).lower()} "
            f"cli={cli.get('cli_path')} found={found} executed={str(cli.get('executed')).lower()} "
            f"auth={auth_text}"
        )


def _print_auth(payload: list[dict[str, Any]]) -> None:
    if not payload:
        typer.echo("(no providers)")
        return
    for row in payload:
        account_id = row.get("account_id") or "-"
        typer.echo(
            f"{row['provider']}  account={account_id}  {row['state']}  method={row['method']}  "
            f"mock={str(row['is_mock']).lower()}  {row.get('account_label') or '-'}"
        )
        if row.get("message"):
            typer.echo(f"  {row['message']}")


def _print_providers(payload: list[dict[str, Any]]) -> None:
    for row in payload:
        caps = row.get("capabilities") or {}
        supported = ",".join(
            label
            for key, label in (
                ("supports_submit", "submit"),
                ("supports_status", "status"),
                ("supports_logs", "logs"),
                ("supports_cancel", "cancel"),
                ("supports_download", "download"),
                ("supports_quota", "quota"),
            )
            if caps.get(key)
        )
        typer.echo(
            f"{row['name']}  enabled={str(row['enabled']).lower()}  "
            f"is_mock={str(row['is_mock']).lower()}  mode={row['mode']}  "
            f"capabilities={supported or '-'}"
        )


def _print_quota(payload: list[dict[str, Any]]) -> None:
    if not payload:
        typer.echo("(no providers)")
        return
    for row in payload:
        account_id = row.get("account_id") or "-"
        typer.echo(
            f"{row['provider']}  account={account_id}  unit={row['unit']}  "
            f"total={row.get('total')}  used={row.get('used')}  remaining={row.get('remaining')}  "
            f"fixture={str(row.get('is_fixture')).lower()}"
        )
        if row.get("message"):
            typer.echo(f"  {row['message']}")


def _print_resources(payload: list[dict[str, Any]]) -> None:
    if not payload:
        typer.echo("(no resources)")
        return
    for row in payload:
        memory = row.get("gpu_memory_gb_per_gpu")
        memory_text = "unknown" if memory is None else str(memory)
        label = row.get("label") or row.get("provider_resource_id") or ""
        owner = row.get("provider") or "-"
        account_id = row.get("account_id") or "-"
        typer.echo(
            f"{owner}  account={account_id}  {row['gpu_type']}  count={row['gpu_count']}  "
            f"gb_per_gpu={memory_text}  premium={str(row['premium']).lower()}  {label}"
        )


def _print_job(payload: dict[str, Any]) -> None:
    typer.echo(f"job_id: {payload['job_id']}")
    typer.echo(f"provider: {payload['provider']}")
    if payload.get("account_id"):
        typer.echo(f"account_id: {payload['account_id']}")
    typer.echo(f"state: {payload['state']}")
    if payload.get("message"):
        typer.echo(f"message: {payload['message']}")
    note = _desync_note(payload)
    if note:
        typer.echo(f"desync: {note}")
    typer.echo(f"is_mock: {str(payload.get('is_mock')).lower()}")


def _desync_note(payload: dict[str, Any]) -> str | None:
    details = payload.get("details")
    if not isinstance(details, dict):
        return None
    desync = details.get("desync")
    if not isinstance(desync, dict):
        return None
    reported = desync.get("reported")
    if not reported:
        return None
    return f"provider reported {reported}"


def _print_unknown_submit(exc: SubmitOutcomeUnknownError) -> None:
    typer.echo(f"error[{exc.code}]: {exc.message}", err=True)
    job_id = exc.details.get("job_id")
    if isinstance(job_id, str) and job_id:
        typer.echo(f"job_id: {job_id}", err=True)
    typer.echo("This job will not be resubmitted. Run `l2kc jobs reconcile`.", err=True)


def _print_job_rows(payload: list[dict[str, Any]]) -> None:
    if not payload:
        typer.echo("(no jobs)")
        return
    typer.echo(f"{'job_id':<32}  {'account_id':<18}  {'provider':<8}  {'state':<12}  message")
    for row in payload:
        message = str(row.get("message") or "").replace("\n", " ")
        typer.echo(
            f"{row['job_id']:<32}  {str(row.get('account_id') or '-'):<18}  "
            f"{str(row.get('provider') or '-'):<8}  "
            f"{str(row.get('state') or '-'):<12}  {message}"
        )


def _print_reconcile(payload: list[dict[str, Any]]) -> None:
    if not payload:
        typer.echo("(no jobs)")
        return
    typer.echo(f"{'job_id':<32}  {'provider':<8}  {'before':<12}  {'after':<12}  ok  note")
    for row in payload:
        typer.echo(
            f"{row['job_id']:<32}  {str(row.get('provider') or '-'):<8}  "
            f"{str(row.get('before') or '-'):<12}  {str(row.get('after') or '-'):<12}  "
            f"{str(bool(row.get('ok'))).lower()}  {_report_note(row)}"
        )


def _report_note(row: dict[str, Any]) -> str:
    if row.get("skipped") and row.get("reason"):
        return f"skipped {row['reason']}"
    note = str(row.get("note") or "")
    error = row.get("error")
    if isinstance(error, dict) and error.get("code") and not row.get("ok"):
        code = str(error["code"])
        return f"{code} {note}".strip()
    return note


def _reconcile_exit(payload: Any) -> int:
    rows = payload if isinstance(payload, list) else []
    if any(not row.get("ok") for row in rows):
        return 1
    return 0


def _print_artifacts(payload: dict[str, Any]) -> None:
    if payload.get("account_id"):
        typer.echo(f"account_id: {payload['account_id']}")
    typer.echo(f"dest_dir: {payload['dest_dir']}")
    artifacts = payload.get("artifacts") or []
    if not artifacts:
        typer.echo("(no artifacts)")
        return
    for item in artifacts:
        typer.echo(f"{item['name']}  {item['local_path']}  {item['size_bytes']}")


def _print_submit(payload: dict[str, Any]) -> None:
    job = payload["job"]
    typer.echo(job["job_id"])
    typer.echo(f"state: {payload.get('state')}")
    if job.get("account_id"):
        typer.echo(f"account_id: {job['account_id']}")
    if payload.get("deduplicated"):
        typer.echo("deduplicated: true")
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return
    provider = decision.get("provider") or "none"
    candidate = decision.get("candidate_id") or "none"
    estimated = decision.get("estimated_ccu")
    ccu = "n/a" if estimated is None else estimated
    fallback = str(bool(decision.get("fallback_used"))).lower()
    typer.echo(
        f"decision: provider={provider} candidate={candidate} "
        f"estimated_ccu={ccu} fallback_used={fallback}",
        err=True,
    )


def _print_recommendation(payload: dict[str, Any]) -> None:
    provider = payload.get("provider") or "none"
    typer.echo(f"provider: {provider}")
    if payload.get("account_id"):
        typer.echo(f"account_id: {payload['account_id']}")
    candidate = payload.get("candidate_id") or "none"
    typer.echo(f"candidate: {candidate}")
    estimated = payload.get("estimated_ccu")
    typer.echo(f"estimated_ccu: {'n/a' if estimated is None else estimated}")
    typer.echo(f"fallback_used: {str(bool(payload.get('fallback_used'))).lower()}")
    typer.echo("reasons:")
    for reason in payload.get("reasons") or []:
        typer.echo(f"- {reason}")
    warnings = payload.get("warnings") or []
    if warnings:
        typer.echo("warnings:")
        for warning in warnings:
            typer.echo(f"- {warning}")


def _stored_account_id(client: Client, job_id: str) -> str | None:
    """Account stored on ``job_id`` when the command does not return a job model.

    Submit, status, and cancel read ``account_id`` from the handle or result.
    Logs and download still use this lookup. It does not call a provider.
    """
    for row in client.list_jobs():
        if row.get("job_id") == job_id:
            value = row.get("account_id")
            return value if isinstance(value, str) else None
    return None


def _default_account_ids(client: Client) -> dict[str, str]:
    ids: dict[str, str] = {}
    for row in client.list_accounts():
        if row.get("is_default") and isinstance(row.get("id"), str):
            ids[str(row["provider"])] = row["id"]
    return ids


def _credential_summary(client: Client, row: dict[str, Any]) -> dict[str, str]:
    """Source kind and presence. The secret value is not read into the result."""
    ref = row.get("credential_ref")
    ref_name = ref if isinstance(ref, str) else None
    ref_config = client.settings.credentials.get(ref_name) if ref_name else None
    health = credential_health(
        ref_name,
        ref_config,
        ProviderName(str(row["provider"])),
        client.settings,
    )
    source = health["source"]
    if source.startswith("env:"):
        kind = "env"
    elif source.startswith("file:"):
        kind = "file"
    elif source.startswith("keyring:"):
        kind = "keyring"
    else:
        kind = source
    return {"kind": kind, "state": health["state"], "source": source}


def _accounts_list_payload(client: Client) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in client.list_accounts():
        rows.append(
            {
                "id": row["id"],
                "provider": row["provider"],
                "enabled": row["enabled"],
                "default": row["is_default"],
                "credential": _credential_summary(client, row),
            }
        )
    return rows


def _account_show_payload(client: Client, account_id: str) -> dict[str, Any]:
    row = client.get_account(account_id)
    body: dict[str, Any] = {
        "id": row["id"],
        "provider": row["provider"],
        "enabled": row["enabled"],
        "default": row["is_default"],
        "labels": row["labels"],
        "credential_ref": row["credential_ref"],
        "mock_scenario": row["mock_scenario"],
        "implicit": row["implicit"],
        "credential": _credential_summary(client, row),
    }
    if client.settings.mode == "mock":
        body["fixture_notice"] = _FIXTURE_ACCOUNT_NOTE
        if row["enabled"]:
            status = client.auth_status(account=account_id)[0]
            body["auth_status"] = status.model_dump(mode="json")
    return body


def _account_change(client: Client, account_id: str, *, enabled: bool) -> dict[str, Any]:
    row = client.enable_account(account_id) if enabled else client.disable_account(account_id)
    payload: dict[str, Any] = {
        "id": row["id"],
        "provider": row["provider"],
        "enabled": row["enabled"],
        "default": row["is_default"],
    }
    if enabled and row["enabled"] is False:
        payload["reason"] = "provider_disabled"
        payload["warning"] = (
            "account remains disabled because "
            f"providers.{row['provider']}.enabled is false"
        )
    return payload


def _enable_exit(payload: Any) -> int:
    """2 when an enable request left the account disabled."""
    if isinstance(payload, dict) and payload.get("enabled") is False:
        return 2
    return 0


def _print_accounts(payload: list[dict[str, Any]]) -> None:
    header = ("id", "provider", "enabled", "default", "credential")
    rendered: list[tuple[str, str, str, str, str]] = []
    for row in payload:
        credential = row.get("credential") if isinstance(row.get("credential"), dict) else {}
        kind = str(credential.get("kind") or "-")
        state = str(credential.get("state") or "-")
        rendered.append(
            (
                str(row.get("id") or "-"),
                str(row.get("provider") or "-"),
                str(bool(row.get("enabled"))).lower(),
                str(bool(row.get("default"))).lower(),
                f"{kind} {state}",
            )
        )
    widths = [len(name) for name in header]
    for line in rendered:
        for index, cell in enumerate(line):
            widths[index] = max(widths[index], len(cell))
    typer.echo("  ".join(name.ljust(widths[index]) for index, name in enumerate(header)))
    for line in rendered:
        typer.echo("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(line)))


def _print_account_show(payload: dict[str, Any]) -> None:
    typer.echo(f"id: {payload['id']}")
    typer.echo(f"provider: {payload['provider']}")
    typer.echo(f"enabled: {str(bool(payload['enabled'])).lower()}")
    typer.echo(f"default: {str(bool(payload['default'])).lower()}")
    labels = payload.get("labels") or {}
    if isinstance(labels, dict) and labels:
        rendered = " ".join(f"{key}={value}" for key, value in labels.items())
    else:
        rendered = "-"
    typer.echo(f"labels: {rendered}")
    credential = payload.get("credential") if isinstance(payload.get("credential"), dict) else {}
    typer.echo(f"credential_kind: {credential.get('kind')}")
    typer.echo(f"credential_state: {credential.get('state')}")
    typer.echo(f"credential_source: {credential.get('source')}")
    scenario = payload.get("mock_scenario")
    if scenario:
        typer.echo(f"mock_scenario: {scenario}")
    notice = payload.get("fixture_notice")
    if notice:
        typer.echo(f"fixture_notice: {notice}")
    auth = payload.get("auth_status")
    if isinstance(auth, dict):
        typer.echo(
            f"auth_state: {auth.get('state')}  mock={str(auth.get('is_mock')).lower()}"
        )
        if auth.get("message"):
            typer.echo(f"auth_message: {auth['message']}")


def _print_account_change(payload: dict[str, Any]) -> None:
    typer.echo(f"id: {payload['id']}")
    typer.echo(f"provider: {payload['provider']}")
    typer.echo(f"enabled: {str(bool(payload['enabled'])).lower()}")
    typer.echo(f"default: {str(bool(payload['default'])).lower()}")
    if payload.get("reason"):
        typer.echo(f"reason: {payload['reason']}")
    if payload.get("warning"):
        typer.echo(f"warning: {payload['warning']}")


if __name__ == "__main__":
    app()
