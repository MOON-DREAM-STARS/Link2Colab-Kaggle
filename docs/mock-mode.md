# Mock mode

## Default

`mode: mock` is the default. `real` must be set explicitly. Real GPU jobs additionally require `allow_real_jobs: true`.

Mock mode must not reach the network and must not spend GPU time or Colab CCU. `Client.doctor` detects the `kaggle` and `colab` binaries with `shutil.which` and does not execute them.

## What you see

- `l2kc` prints `MOCK MODE: quota, balance, and membership values are TEST FIXTURES, not a live account.` on stderr. `--json` stdout does not include that banner.
- MCP tools return `mode: mock`, `is_mock: true`, and a `fixture_notice`. Their descriptions say mock data are test fixtures.
- `recommend` adds the same fixture warning. It does not submit a job.

## Test fixtures

Quotas, balances, and membership values produced by mocks are **TEST FIXTURES**. Label them as test fixtures in code and in user-visible output. They are not the state of a real account.

See docs/api-verification.md before copying any live quota or billing field into the mock.

## Quick smoke

See `docs/linux-wsl-setup.md` for `l2kc doctor`, `l2kc submit --provider fake`, and MCP entry on Linux/WSL2.
