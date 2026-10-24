# Authentication

Link2Kaggle&Colab never stores secret values in config. Config holds **references** only
(`credential_ref`, `credential_source`, `api_key_env` as an environment variable **name**).

## Modes

| Mode | Behaviour |
| --- | --- |
| `mode: mock` (default) | No network. Auth snapshots are TEST FIXTURES. `doctor` does not execute `kaggle` or `colab`. |
| `mode: real` | Presence checks and (when allowed) official CLI calls. Success is never hard-coded. |

Real GPU / CCU submit additionally requires `allow_real_jobs: true`. Either flag alone is not enough.

## Rules

- Never commit tokens, `.env` files, `kaggle.json`, OAuth caches, PEM/key files, or Application Default Credentials.
- Never log tokens, authorization headers, refresh tokens, or credential file contents. Redact them in errors and before SQLite writes.
- `l2kc doctor` / real-mode auth presence use the **default account's** credential overlay (not a bare ambient `HOME/.kaggle` when an account is configured). Presence only; CLIs are not executed for doctor.
- Multi-account credential sources: `docs/accounts.md`. Setup pointers: `docs/real-auth-setup.md`. Official CLI surfaces: `docs/api-verification.md`.

## Real auth status

V0.1–V0.4 development and CI used mock mode only. **Real Kaggle/Colab login and paid GPU jobs were not verified end-to-end in this repo's test history.**
