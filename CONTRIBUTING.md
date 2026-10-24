# Contributing

Thanks for helping with Link2Kaggle&Colab.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Python >= 3.11.

## Local CI (required before push)

```bash
make ci
# or: ./scripts/ci.sh
```

That runs `compileall`, `ruff check .`, offline `pytest` (`-m "not live"`), and `python -m build`.

Faster loop:

```bash
make lint
make test
```

## Mock-first

- Default mode is `mock`. Do not submit real Kaggle/Colab jobs in development or CI.
- Real GPU/CCU work needs both `mode: real` and `allow_real_jobs: true` — only when the user explicitly wants spend.
- Never commit secrets (tokens, `.env`, `kaggle.json`, credential JSON, keys). See [`SECURITY.md`](SECURITY.md).

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `chore:`, etc.

## Docs

Keep `README.md` and `README.zh-CN.md` in sync for user-facing changes.
