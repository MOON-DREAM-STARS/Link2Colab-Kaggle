# Real auth setup

Real mode is opt-in. Mock remains the default.

## Switches

1. Set `mode: real` in your local config (copy from `config.example.yaml`; do not commit secrets).
2. For any job that uses GPU or Colab compute units, also set `allow_real_jobs: true`.

Either flag alone does not submit real GPU work. `mode: real` without `allow_real_jobs` still blocks GPU/CCU submits. `allow_real_jobs` without `mode: real` stays in mock.

## Credentials (references only)

Point config at a reference, never at the secret value itself:

- Kaggle: `KAGGLE_API_TOKEN` / `KAGGLE_CONFIG_DIR`, or a multi-account `credentials:` entry (`kaggle_config_dir`, `env`, …).
- Colab: `adc` (Application Default Credentials path) or `oauth2` / `colab_home`, matching `providers.colab.auth`.

Do not commit tokens, `.env` files, `kaggle.json`, OAuth caches, or ADC JSON. See `docs/accounts.md` for multi-account shapes and `docs/authentication.md` for redaction rules.

## Disclaimer

**Real authentication, real GPU jobs, and live Colab CCU rates were not tested in V0.1–V0.4.**
Default `scheduler.ccu_rates` are unverified estimates. Confirm balances with the provider UI or `colab usage` / `kaggle` quota commands on your own account before spending money.

Provider login command details and verified CLI surfaces: `docs/api-verification.md`.
