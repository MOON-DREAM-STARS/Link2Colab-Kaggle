# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-24

Initial public release of **Link2Kaggle&Colab**: a local MCP server and CLI that submits jobs to Kaggle and Google Colab from your machine.

### Added

- **Providers:** Kaggle, Google Colab, and offline `fake` runner (mock-first by default).
- **Surfaces:** SDK (`link2kaggle_colab.Client`), CLI (`l2kc`), stdio MCP server (`link2kaggle-colab-mcp`).
- **Scheduler:** recommend / auto placement with quota-aware guards; CCU and GPU-hours are never converted.
- **Jobs:** SQLite job manager, list / reconcile, idempotency keys, restart recovery.
- **Multi-account:** credential references in config (paths / env / keyring); no secrets in YAML.
- **Safety:** artifact path sandbox, MCP stdio-only, double gate for real GPU/CCU (`mode: real` + `allow_real_jobs: true`).
- Offline CI: ruff, pytest (`-m "not live"`), and package build on Python 3.11 / 3.12.

### Notes

- Default mode is **mock**: no real account, network, GPU, or Colab compute units (CCU).
- Mock quota / balance / membership text is a **TEST FIXTURE**, not a live account.
- License: Apache-2.0.
