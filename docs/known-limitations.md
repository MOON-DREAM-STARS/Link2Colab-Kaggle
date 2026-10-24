# Known limitations

## Safety limits that stay in force

- No real GPU and no real Colab CCU from the default test suite or from mock mode.
- Live tests, when they exist, must be marked `live` and stay excluded (`pytest -m "not live"`).
- Real mode needs `mode: real`; GPU/CCU submit also needs `allow_real_jobs: true`.
- Default Colab CCU rates in config are **unverified**. CCU and GPU-hours are never converted.

## MCP trust boundary

The stdio MCP server runs with the user's local privileges. Downloads are confined to `artifact_root` (default `./l2kc-artifacts`); paths outside that root raise `PathSecurityError`. Fake local exec is **off** in the MCP process unless `L2KC_ALLOW_FAKE_EXEC=1` (or `true`/`yes`). The server pins config at startup (`L2KC_CONFIG` / search path) and logs mode, roots, account ids, and non-secret credential home labels on stderr. Only connect trusted MCP clients.

Remote transports (`sse`, `http`, `streamable-http`) are refused at process start (`--transport` exit 2) until auth and sandbox exist. Only `stdio` runs.

Canonical tools are the `gpu_*` names. Each is also registered as `l2kc_*` (same callable). New tools: `gpu_doctor` / `l2kc_doctor`, `gpu_refresh_quota` / `l2kc_refresh_quota`. `gpu_get_quota` accepts `refresh`; `gpu_submit_job` accepts `dry_run` (recommend only, no submit). `gpu_list_resources` may attach `eligible` when quota publishes `eligible_gpus`. Structured tool output is preferred when the MCP SDK accepts the return annotation; otherwise unstructured.

Kaggle cancel stays `not_supported` (official CLI has no cancel). No fake cancel path.

## Colab exec output matching

Exit code 0 is success and is not scanned for "disconnect". A non-zero `exec` is a runtime disconnect only when stderr or the combined log matches a runtime marker such as "the runtime disconnected"; other non-zero exits stay a normal failure with a redacted excerpt. The substring false positive (P2-1) is fixed.

## Kaggle premium accelerators

Kaggle A100/H100 are also gated in the adapter and may still be rejected server-side. Needs a real-account check.

## V0.2 scheduler (open P3)

- CCU estimate counts `timeout_seconds` only (setup/transfer omitted).
- Premium-vs-unavailable error chooses by reason substrings.
- `auto` ignores P100/H100 (not in default cost priority); pin the provider to use them.
- Per-process quota cache in a long-lived MCP server.
- A100 variant pricing (40 GB vs 80 GB) unverified on a live account.

## V0.3 Job Manager (open P3)

P3-1 (idempotency key + different spec) is **fixed** (`03ea307`). P3-3 (PENDING crash window) is **fixed**: a new job is inserted as `SUBMITTING` with `submit_started_at` and `submit_pid`. Still open: P3-2, P3-4..P3-8 (scheduler-before-dedupe on resubmit, stale_submitting vs job timeout, DB file mode, spec-hash dedupe surprise, unlocked provider build, WARNING last-resort handler) plus CP8 re-review notes. See `docs/reviews/v0.3-review.md`.

## Colab quota sources and CCU=0 eligible GPUs

**Default:** Colab quota comes from official ``colab usage`` text parse (balance + optional
``Eligible GPUs:`` line). The undocumented HTTP endpoint
``https://colab.research.google.com/tun/m/ccu-info`` is **experimental and opt-in only**
(``providers.colab.experimental_ccu_info: true`` and/or env ``L2KC_COLAB_EXPERIMENTAL_CCU_INFO=1``).
When enabled, a successful structured parse can populate ``eligible_gpus``; on parse/HTTP
failure the adapter degrades to ``colab usage``. Do not treat ccu-info as a stable API.

When Colab compute-unit balance is 0, the scheduler may still attempt accelerators listed in
live ``eligible_gpus`` (from opt-in ccu-info or from a usage line that publishes them; for
example free Pro accounts that only list T4). Without that list, ``eligible_gpus`` may be
``None`` and the existing unknown+0 CCU T4 warning path remains. Availability is still not
guaranteed. ``membership_tier`` in ``state_dir/accounts/<id>/identity.json`` is informational
only (``free_pro`` ≠ paid Pro) and is **not** a scheduler input.

## V0.4 accounts

- `JobHandle` / `JobResult` carry `account_id` from the stored row — P3-1 **fixed**. CLI and MCP read it from the handle or result.
- `accounts enable` exits 2 with `reason: provider_disabled` when the provider mask keeps the account off — P3-2 **fixed**.
- `doctor` is one auth row per provider (default account), not a full list — P3-3.
- Reconcile skipped rows use `ok: true` with `skipped: true` — P3-4.
- `colab_home` sets child `HOME`; whether the real `colab` CLI honors it for `~/.config/colab-cli/token.json` is unverified — P3-5.
- `keyring` credential source needs the optional `keyring` package; without it health is `unsupported` — P3-6.
- Plugin providers are interface-only; `ProviderName` does not accept a plugin name in config yet.

## Unwired on purpose

`kaggle auth login`, `kaggle auth print-access-token`, `kernels init`, `kernels push --no-run`, `kernels delete`, `kernels output -o/--file-pattern`, Kaggle cancel, `colab run` / `status` / `log` / `rm`, remote MCP bridges, and the Colab REST API.

### Won't-fix-for-now (documented)

- **Kaggle cancel.** Official `kaggle` CLI cannot cancel a running kernel. MCP/CLI surface `not_supported`. No synthetic cancel.
- **Remote MCP transports.** Refused until auth + sandbox; stdio only.

### Shipped: Colab `--tpu` / `--high-mem`

JobSpec carries `high_mem: bool` (default false). `GpuType` includes `TPU_V5E1` / `TPU_V6E1` (Colab-only; maps to `colab new --tpu v5e1|v6e1`, mutually exclusive with `--gpu`). Resource catalog, scheduler candidates (`colab:TPU-v5e1` / `colab:TPU-v6e1`), CCU rate keys (fixture estimates), MCP (`high_mem`), and CLI (`--high-mem`, `--gpu TPU_V5E1`) are wired. `high_mem` does **not** bypass premium guards. L4 and TPU reject `high_mem` (single machine shape). TPU/H100 stay outside default `cost_priority`; request them explicitly (and pin `provider=colab` when needed). Live TPU/high-mem CCU rates are unverified fixtures — confirm with `colab usage`.
