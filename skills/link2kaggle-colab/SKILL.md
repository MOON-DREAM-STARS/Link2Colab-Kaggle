---
name: link2kaggle-colab
description: Use the link2kaggle-colab MCP tools or the l2kc CLI to inspect providers and submit local scripts to Kaggle, Colab, or the offline fake runner. Mock mode is the default. Real GPU jobs and premium accelerators stay off unless the user explicitly enables them. MCP disables fake local exec unless `L2KC_ALLOW_FAKE_EXEC=1`.
---

# Link2Kaggle&Colab

Use the stdio MCP server `link2kaggle-colab` (`link2kaggle-colab-mcp` or `python -m link2kaggle_colab.mcp`). The `l2kc` CLI calls the same `Client`. Do not invent quota, membership, or job state.

Mock mode is the default. Every tool result includes `mode` and `is_mock`. When `mode` is `mock`, quota, balance, membership, and auth fields are **TEST FIXTURES**, not a live account. Say that when you repeat those numbers to the user.

## Tools

Canonical names are `gpu_*`. Each tool is also registered as `l2kc_*` (same callable), e.g. `gpu_list_providers` ↔ `l2kc_list_providers`, `gpu_doctor` ↔ `l2kc_doctor`, `gpu_refresh_quota` ↔ `l2kc_refresh_quota`. Prefer `gpu_*` in docs and new clients.

Call the tools in this order when the user wants a GPU job:

1. `gpu_list_providers` — see which backends are enabled.
2. `gpu_list_accounts` — account ids, which are enabled, and which is the default. No secret values.
3. `gpu_auth_status` — optional `provider` of `kaggle`, `colab`, or `fake`, and optional `account`.
4. `gpu_get_quota` — same optional `provider` and `account`. Optional `refresh` (default false) drops the cache first. Treat `is_fixture: true` as a fixture. Or call `gpu_refresh_quota` / `l2kc_refresh_quota`.
5. `gpu_list_resources` — optional `provider` and `account`. Memory is **per GPU**. Never add memory across `gpu_count`. A Kaggle T4 is two 16 GB GPUs, not one 32 GB GPU. When quota publishes `eligible_gpus`, each row includes `eligible` (bool or null).
6. `gpu_recommend_backend` — does not submit. Pass `gpu_type`, `gpu_count`, `min_gpu_memory_gb_per_gpu`, `allow_premium`, `high_mem` (default false; Colab `--high-mem`), `timeout_seconds` (required when Colab may be chosen, so the CCU budget can be checked), `provider` (`auto`, `kaggle`, or `colab`), and optional `account`. Colab TPU: `gpu_type` `TPU_V5E1` or `TPU_V6E1` (premium; prefer `provider=colab`).
7. `gpu_doctor` / `l2kc_doctor` — local doctor report (`mode`, writability, CLI presence, auth). Does not execute `kaggle` or `colab`.
8. `gpu_submit_job` — `script_path`. `provider` defaults to `auto` (the scheduler). Optional `account`, `gpu_type` (including Colab `TPU_V5E1` / `TPU_V6E1`), `gpu_count` (default 1), `timeout_seconds` (required when Colab may be chosen), `name`, `allow_premium` (default false), `high_mem` (default false; Colab high-RAM; not L4/TPU; does not bypass premium), `min_gpu_memory_gb_per_gpu`, `idempotency_key`, `dry_run` (default false). When `dry_run` is true, returns `{dry_run, recommendation, estimated_ccu}` and does **not** submit. Otherwise the result includes `job` (with `account_id`), `decision` (`decision` is null for CPU and `fake`), and `deduplicated`. The same key returns the existing job and does not submit again. `job.job_id` is ours, not the provider id.
9. `gpu_get_job_status` — repeat until the state is terminal (`SUCCEEDED`, `FAILED`, `CANCELLED`). Uses the stored `account_id`. `UNKNOWN` means the submit outcome is ambiguous.
10. `gpu_get_job_logs` — optional `max_chars` (default 20000) returns the tail. Uses the stored account.
11. `gpu_download_artifacts` — optional `dest_dir` under `artifact_root` (default `./l2kc-artifacts`). Paths outside the root are rejected. Uses the stored account.
12. `gpu_cancel_job` — uses the stored account. Colab and the mock can cancel. Kaggle's official CLI cannot; a `not_supported` error is expected.
13. `gpu_list_jobs` — optional `state` (list), `provider`, and `limit` (default 50). Stored rows include `account_id`. Does not call a provider.
14. `gpu_reconcile_jobs` — optional `job_ids`. Polls status for those ids, or every active job when omitted. Never submits. A disabled or unknown account is skipped and the row is left unchanged. Use this for `UNKNOWN`. It will not resubmit the job.

Do not switch to another account when quota, a rate limit, or auth fails. `account_disabled` is not a cue to submit on a different account. See `docs/accounts.md`.

MCP process notes: fake local exec stays off unless `L2KC_ALLOW_FAKE_EXEC=1` (or `true`/`yes`). Only `--transport stdio` is enabled; remote transports exit 2.

A successful payload is `{"ok": true, "mode": "mock"|"real", "is_mock": ..., ...}`. A known failure is `{"ok": false, "error": {"code", "message", "provider", "details"}}`. Do not retry a `not_supported`, `premium_not_allowed`, `real_mode_disabled`, `quota_exceeded`, or `submit_outcome_unknown` error by submitting again. Call `gpu_reconcile_jobs` for `submit_outcome_unknown`. That poll does not resubmit.

## Safety

- Real mode is off by default. Do not set `mode: real` or `allow_real_jobs: true` unless the user explicitly asks for a real job. Either flag alone does not submit real GPU work.
- Do not pass `allow_premium: true` unless the user explicitly allows a premium accelerator (Colab A100, H100, G4, TPU_V5E1, TPU_V6E1). Premium also needs `scheduler.allow_premium_gpu: true`. The per-job flag does not override that setting. H100 has no default CCU rate and is refused until a rate is configured. `high_mem` does not bypass premium guards. L4 and TPU reject `high_mem`.
- Do not read or paste tokens, `kaggle.json`, OAuth caches, or Application Default Credentials. Config stores credential references only.
- Do not claim a mock quota, balance, or membership is the user's live account.
- `gpu_recommend_backend` must not be followed by a submit the user did not ask for.
