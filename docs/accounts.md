# Accounts

Default is one account per provider. Config can name more. Selection is deterministic. Link2Kaggle&Colab does not switch to another account of the same provider when quota is exhausted, a rate limit is hit, authentication fails, or the provider returns an error. Doing that would evade the provider's limit.

Mock quotas, balances, and membership values are **TEST FIXTURES**. They are not a live account.

## Config

`accounts` is optional. When it is absent, the registry synthesizes `kaggle-default`, `colab-default`, and `fake-default`. Those legacy ids use ambient credentials (the process environment and home) and are enabled only when that provider is enabled. Their adapter state stays on the historical paths (`state_dir/kaggle`, `state_dir/colab`, `state_dir/mock/<provider>`, `state_dir/fake`).

```yaml
accounts:
  - id: kaggle-primary          # ^[a-z0-9][a-z0-9-]{0,62}$, unique
    provider: kaggle            # kaggle | colab | fake
    credential_ref: kaggle-main # name of a credential reference, never a secret
    enabled: true               # default true
    default: false              # at most one default per provider
    labels: {team: a}           # strings, not secrets
    mock_scenario: quota_45h    # TEST FIXTURE selector, mock mode only
credentials:
  kaggle-main: {source: kaggle_config_dir, path: ~/.l2kc/credentials/kaggle-main}
  kaggle-alt: {source: env, env: KAGGLE_ALT_API_TOKEN}
  colab-main: {source: adc_file, path: ~/creds/colab-main-adc.json}
  colab-alt: {source: colab_home, path: ~/.l2kc/credentials/colab-alt}
  vault-x: {source: keyring, service: l2kc, username: kaggle-x}
```

`examples/config.multi-account.yaml` is a mock-mode copy of this shape. It stores references only.

A `credential_ref` that is not listed under `credentials` uses `state_dir/credentials/<ref>/`:

- Kaggle: `kaggle_config_dir` at that directory.
- Colab with `providers.colab.auth: oauth2`: `colab_home` at that directory.
- Colab with `auth: adc`: an ADC file at `<dir>/application_default_credentials.json`.
- `fake`: no credentials.

`credential_ref: null` means ambient credentials, the same inheritance as a single-account setup. At most one ambient account per provider. Two accounts cannot share a `credential_ref` or the same resolved location (path, env var name, or keyring service and username).

Validation rejects duplicate ids, more than one default per provider, and any string that looks like a token, key, PEM block, or `kaggle.json` body. The error names the field. It does not echo the value.

## Credential sources

| `source` | What is stored | Child process |
| --- | --- | --- |
| `kaggle_config_dir` | Directory path | `KAGGLE_CONFIG_DIR` set; `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY` unset |
| `env` | Environment variable **name** | Value of that variable, read at use time, passed as `KAGGLE_API_TOKEN`. Not logged. |
| `adc_file` | ADC JSON path | `GOOGLE_APPLICATION_CREDENTIALS` set |
| `colab_home` | Directory used as child `HOME` | `HOME` set; `GOOGLE_APPLICATION_CREDENTIALS` unset. Token path `<path>/.config/colab-cli/token.json` is presence-only |
| `keyring` | Service and username | Optional `keyring` package. Without it, health is `unsupported` |

Health reports `present`, `missing`, or `unsupported`, plus a source string such as `env:NAME`, `file:<path>`, `keyring:<service>`, `ambient`, or `none`. It does not read file contents, return secret values, or call the network. `l2kc accounts list` prints the source kind and that presence flag. `l2kc accounts show` adds labels and, in mock mode, a TEST FIXTURE note plus the mock auth snapshot.

`colab_home` isolation against the real `colab` CLI is unverified. See `docs/known-limitations.md`.

## Isolation

Each account gets its own child environment. The runner copies the process environment, applies unsets, then applies sets. It does not mutate `os.environ`, and it does not store that overlay on command results or logs.

Non-implicit adapter and mock state lives under `state_dir/accounts/<account_id>/`. The quota cache key is `(provider, account_id)`. `account.mock_scenario` overrides `mock.kaggle_scenario` or `mock.colab_scenario` for that account only. One account's error does not change another account's job rows.

## Selection

`select_account` returns the explicit id, or the account with `default: true`, or the first listed account for that provider. It does not read quota.

- The explicit id must exist, match the provider, and be enabled.
- A disabled default raises `account_disabled`. The next account of that provider is not used.
- `--account` on `submit`, `recommend`, `quota`, `auth status`, and `resources` pins that account. A conflicting `--provider` is `config_error`.
- The scheduler asks for one account per provider. If that account is disabled or missing, the provider is ineligible. It does not try a second account.
- `status`, `logs`, `download`, and `cancel` do not take `--account`. They use the `account_id` stored on the job. Job output includes that id.

## Disabled accounts

`l2kc accounts disable ID` and `enable ID` persist overrides in `state_dir/accounts_state.json`. Config is not rewritten.

A disabled account is refused before any provider call on submit, status, logs, download, and cancel (`account_disabled`, exit 2). `jobs reconcile` skips that row and reports `skipped: true` with `reason: account_disabled`. An unknown stored id is `reason: account_not_found`. The row is not updated, and the provider is not called. Another account's rows are unchanged.

## Legacy ids and migration

Jobs written before account support are backfilled to `<provider>-default`. If that id is not in the configured list, `resolve_for_record` follows the provider's current default. Any other stored id must match a configured account. JSON index import uses the same `<provider>-default` id.

## Plugins

`register_provider_factory(name, factory)` and `load_entry_point_factories(group="link2kaggle_colab.providers")` are the extension points. `ProviderFactory` is `(settings, account, runner) -> Provider`. Entry points load only when that function is called, not at import. Built-in names win and cannot be replaced. V0.4 ships no extra backends. `ProviderName` stays `kaggle`, `colab`, and `fake`, so a plugin name is not accepted in config yet.

## CLI errors

Known errors print `error[<code>]: <message>` on stderr and exit 2. Unexpected errors exit 1. `--json` writes the payload to stdout only. The mock-mode banner stays on stderr and is not part of the JSON document.

## Identity cache

On successful Colab or Kaggle quota refresh, the adapter writes a non-secret
``state_dir/accounts/<account_id>/identity.json`` (email or username, quota
remaining, ``eligible_gpus`` for Colab when known, and a display-only
``membership_tier``). ``free_pro`` means a free Pro-shaped Colab account, not a
paid subscription. The scheduler never reads ``membership_tier``; GPU exemptions
at CCU=0 use ``eligible_gpus`` only.

Colab ``eligible_gpus`` normally comes from ``colab usage`` when that CLI prints
an Eligible GPUs line. The undocumented ``ccu-info`` endpoint is experimental /
opt-in (``providers.colab.experimental_ccu_info`` or
``L2KC_COLAB_EXPERIMENTAL_CCU_INFO``); see ``docs/known-limitations.md``.

