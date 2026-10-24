# Security Policy

## Trust boundary

Link2Kaggle&Colab is a **local** CLI and MCP server. It runs with your user privileges.

| Surface | Boundary |
| --- | --- |
| MCP transport | **stdio only.** Remote transports (`sse`, `http`, `streamable-http`) are refused at startup. |
| Artifact downloads | Confined to `artifact_root` (default `./l2kc-artifacts`). Paths outside that root raise `PathSecurityError`. |
| Fake / local exec | In the MCP process, local script execution via provider `fake` is **off** unless `L2KC_ALLOW_FAKE_EXEC=1` (or `true` / `yes`). |
| Real GPU / Colab CCU | Double-gated: `mode: real` **and** `allow_real_jobs: true`. Either flag alone does not submit billed work. Mock is the default. |

Connect only trusted MCP clients. Do not treat mock quota, balance, or membership text as a live account.

More detail: [`docs/known-limitations.md`](docs/known-limitations.md).

## Reporting a vulnerability

Please report security issues privately via [GitHub Security Advisories](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/security/advisories/new) (preferred). If Private Vulnerability Reporting is enabled for the repository, use that flow from the Security tab. Do **not** open a public GitHub Issue for a vulnerability — public Issues are not private. Do not include secrets, tokens, or credential files in the report.

## What not to commit

Never commit tokens, `.env` files, OAuth caches, `kaggle.json`, credential JSON, PEM/key material, or Application Default Credentials. Config should store credential **references** only.

---

## 安全说明（简要）

- MCP 仅 **stdio**；产物下载限制在 `artifact_root`，越界抛 `PathSecurityError`。
- MCP 下 `fake` 本地执行默认关闭，需 `L2KC_ALLOW_FAKE_EXEC=1`。
- 真实 GPU/CCU 需同时 `mode: real` 与 `allow_real_jobs: true`。
- 漏洞请通过 GitHub Security Advisories / Private Vulnerability Reporting 私下报告；勿使用公开 Issue，勿附带密钥。
