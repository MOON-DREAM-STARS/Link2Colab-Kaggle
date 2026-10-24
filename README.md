# Link2Kaggle-Colab

Local MCP server and CLI to submit jobs to **Kaggle** and **Google Colab** from your machine.

[![CI](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-brightgreen)](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/releases/tag/v1.0.0)

English | [简体中文](README.zh-CN.md)

## Status

**v1.0.0** — initial public release on `main` ([changelog](./CHANGELOG.md)).

- Default mode is **mock**: no real account, network, GPU, or Colab compute units (CCU).
- Real GPU/CCU submit needs both `mode: real` and `allow_real_jobs: true`.
- Mock quota / balance / membership text is a **TEST FIXTURE**, not a live account.

## Features

- **Providers:** Kaggle, Google Colab, and offline `fake` runner
- **Surfaces:** SDK (`link2kaggle_colab.Client`), CLI (`l2kc`), stdio MCP server
- **Scheduler:** recommend / auto placement with quota-aware guards
- **Jobs:** SQLite job manager, list / reconcile, idempotency keys
- **Multi-account:** credential references in config (paths / env / keyring), no secrets in YAML

| Piece | Name |
| --- | --- |
| Distribution | `link2kaggle-colab` |
| Import package | `link2kaggle_colab` |
| SDK | `link2kaggle_colab.Client` |
| CLI | `l2kc` |
| MCP (stdio) | `link2kaggle-colab-mcp` or `python -m link2kaggle_colab.mcp` |
| Skill | `skills/link2kaggle-colab/SKILL.md` |

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/l2kc doctor
.venv/bin/l2kc quota
.venv/bin/l2kc submit examples/train_example.py --provider fake
```

Local CI gate:

```bash
make ci
# or: ./scripts/ci.sh
```

`l2kc doctor` prints a `MOCK MODE` line on stderr (not part of `--json` stdout). Linux/WSL notes: [`docs/linux-wsl-setup.md`](docs/linux-wsl-setup.md).

## Safety

Connect only trusted MCP clients. Downloads are confined to `artifact_root` (default `./l2kc-artifacts`); paths outside that root raise `PathSecurityError`. In the MCP process, local `fake` exec is **off** unless `L2KC_ALLOW_FAKE_EXEC=1`. Real billed work is double-gated (`mode: real` + `allow_real_jobs: true`).

See [`SECURITY.md`](SECURITY.md) and [`docs/known-limitations.md`](docs/known-limitations.md).

## MCP setup (Cursor / stdio)

```json
{
  "mcpServers": {
    "link2kaggle-colab": {
      "command": "link2kaggle-colab-mcp"
    }
  }
}
```

From a virtualenv:

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

Canonical tools use the `gpu_*` prefix (also registered as `l2kc_*`). Agent guidance: [`skills/link2kaggle-colab/SKILL.md`](skills/link2kaggle-colab/SKILL.md).

## Real mode (spend warning)

Real Kaggle/Colab auth and GPU/CCU jobs can cost money or quota. Read [`docs/real-auth-setup.md`](docs/real-auth-setup.md) before enabling:

1. `mode: real` in config
2. `allow_real_jobs: true` for any GPU/CCU submit

Either flag alone does not submit billed work. Default CI never burns GPU or Colab CCU.

## Docs index

| Doc | Topic |
| --- | --- |
| [`SECURITY.md`](SECURITY.md) | Trust boundary, reporting |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev setup, `make ci`, mock-first |
| [`docs/known-limitations.md`](docs/known-limitations.md) | Limitations and Won't-Fix |
| [`docs/mock-mode.md`](docs/mock-mode.md) | Mock defaults |
| [`docs/real-auth-setup.md`](docs/real-auth-setup.md) | Real credentials |
| [`docs/accounts.md`](docs/accounts.md) | Multi-account |
| [`docs/architecture.md`](docs/architecture.md) | Design overview |
| [`docs/linux-wsl-setup.md`](docs/linux-wsl-setup.md) | Linux/WSL |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |
| [`config.example.yaml`](config.example.yaml) | Config template |

## CLI (cheat sheet)

Global options go before the subcommand: `l2kc --config ./l2kc.yaml doctor`.

| Command | Purpose |
| --- | --- |
| `l2kc doctor` | Mode, state dir, CLI presence, auth (does not run `kaggle`/`colab`) |
| `l2kc providers` / `quota` / `resources` | Providers and fixtures (mock) or live reads (real) |
| `l2kc recommend …` | Pick a backend; does not submit |
| `l2kc submit FILE [--provider auto\|kaggle\|colab\|fake] …` | Submit `.py` / `.ipynb` |
| `l2kc status` / `logs` / `download` / `cancel` | Job lifecycle |
| `l2kc accounts …` / `l2kc jobs …` | Accounts and local job store |

Known errors print `error[<code>]: <message>` on stderr and exit 2. Full flag lists: run `l2kc --help` / `l2kc submit --help`.

## Limitations (short)

- Mock quotas are fixtures; default Colab CCU rates are **unverified**.
- Kaggle cancel is `not_supported` (official CLI has no cancel).
- Remote MCP transports are refused; **stdio only**.
- Some premium accelerators need a real-account check.

Full list: [`docs/known-limitations.md`](docs/known-limitations.md).

## License

[Apache License 2.0](./LICENSE)
