# Linux and WSL2 (Ubuntu) quick start

Verified path for local development: Ubuntu on WSL2 or native Linux, Python >= 3.11.
Mock mode is the default. This page does not spend GPU time or Colab CCU.

## Install

```bash
# From the repo root
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e '.[dev]'
```

Optional provider CLIs (needed only for `mode: real`; not required for mock smoke):

- Kaggle CLI: install the official `kaggle` package / CLI so `kaggle` is on `PATH`.
- Colab CLI: package `google-colab-cli`; executable name is `colab`. Auth values are `oauth2` or `adc` (default `adc`).

Exact install paths for those CLIs on every distro are not re-verified here. See `docs/api-verification.md`.

## Mock-mode smoke

```bash
.venv/bin/l2kc doctor
.venv/bin/l2kc submit examples/train_example.py --provider fake
.venv/bin/l2kc jobs list
```

Expect a `MOCK MODE` banner on stderr from `l2kc`. Quota, balance, and membership text are **TEST FIXTURES**. `fake` runs locally and never calls Kaggle or Colab.

Offline gate (no network, GPU, or CCU):

```bash
.venv/bin/python -m compileall -q src
.venv/bin/ruff check .
.venv/bin/pytest -m "not live" -q --cov=link2kaggle_colab
```

## MCP entry (stdio)

Point a trusted MCP client at the installed console script, or at the venv module:

```json
{
  "mcpServers": {
    "link2kaggle-colab": {
      "command": ".venv/bin/python",
      "args": ["-m", "link2kaggle_colab.mcp"]
    }
  }
}
```

Equivalent: `"command": "link2kaggle-colab-mcp"` when that script is on `PATH`.

## Real mode (opt-in, not tested on GPU here)

Real mode needs **both**:

1. `mode: real` in config (never implied by a missing key).
2. `allow_real_jobs: true` for any submit that uses GPU or Colab CCU.

Either flag alone is not enough. Credential setup: `docs/real-auth-setup.md` and `docs/authentication.md`.
**Real auth, real GPU jobs, and live Colab CCU rates were not tested in V0.1–V0.4.** Default Colab CCU rates in config are unverified estimates.

## Notes

- State directory defaults to `~/.l2kc` (`L2KC_STATE_DIR` overrides). Job store is `state_dir/l2kc.db`.
- Do not commit `.env`, `kaggle.json`, ADC JSON, OAuth caches, or tokens.
- Local CI: run `make ci` (or `./scripts/ci.sh`) before pushing. GitHub Actions is optional and parked under `ci/` — see `ci/README.md`.
