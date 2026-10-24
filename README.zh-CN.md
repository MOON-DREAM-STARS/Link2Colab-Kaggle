# Link2Kaggle-Colab

本机 MCP 服务器与 CLI：把本地脚本提交到 **Kaggle** 与 **Google Colab**。

[![CI](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-brightgreen)](https://github.com/MOON-DREAM-STARS/Link2Colab-Kaggle/releases/tag/v1.0.0)

[English](README.md) | 简体中文

## 状态

**v1.0.0** — 首次公开发布，分支 `main`（[变更日志](./CHANGELOG.md)）。

- 默认 **mock**：不碰真实账号、网络、GPU 或 Colab 算力单位（CCU）。
- 真实 GPU/CCU 提交需同时设置 `mode: real` 与 `allow_real_jobs: true`。
- Mock 里的额度 / 余额 / 会员文案是 **测试夹具（TEST FIXTURE）**，不是线上账号。

## 功能

- **Provider：** Kaggle、Google Colab、离线 `fake`
- **入口：** SDK（`link2kaggle_colab.Client`）、CLI（`l2kc`）、stdio MCP
- **调度：** recommend / auto，带额度相关护栏
- **任务：** SQLite 任务管理、list / reconcile、幂等键
- **多账号：** 配置里只存凭据引用（路径 / 环境变量 / keyring），不写密钥

| 组件 | 名称 |
| --- | --- |
| 发行包名 | `link2kaggle-colab` |
| 导入包 | `link2kaggle_colab` |
| SDK | `link2kaggle_colab.Client` |
| CLI | `l2kc` |
| MCP（stdio） | `link2kaggle-colab-mcp` 或 `python -m link2kaggle_colab.mcp` |
| Skill | `skills/link2kaggle-colab/SKILL.md` |

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/l2kc doctor
.venv/bin/l2kc quota
.venv/bin/l2kc submit examples/train_example.py --provider fake
```

本地 CI 门槛：

```bash
make ci
# 或: ./scripts/ci.sh
```

`l2kc doctor` 会在 stderr 打 `MOCK MODE`（不属于 `--json` stdout）。Linux/WSL：[`docs/linux-wsl-setup.md`](docs/linux-wsl-setup.md)。

## 安全

只连接可信 MCP 客户端。产物下载限制在 `artifact_root`（默认 `./l2kc-artifacts`），越界抛 `PathSecurityError`。MCP 进程内 `fake` 本地执行默认关闭，除非设置 `L2KC_ALLOW_FAKE_EXEC=1`。真实计费任务双闸门：`mode: real` + `allow_real_jobs: true`。

详见 [`SECURITY.md`](SECURITY.md) 与 [`docs/known-limitations.md`](docs/known-limitations.md)。

## MCP 配置（Cursor / stdio）

```json
{
  "mcpServers": {
    "link2kaggle-colab": {
      "command": "link2kaggle-colab-mcp"
    }
  }
}
```

虚拟环境：

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

规范工具名为 `gpu_*`（同时注册 `l2kc_*`）。Agent 用法见 [`skills/link2kaggle-colab/SKILL.md`](skills/link2kaggle-colab/SKILL.md)。

## Real 模式（会花钱 / 耗额度）

真实鉴权与 GPU/CCU 任务可能产生费用或消耗额度。启用前请阅读 [`docs/real-auth-setup.md`](docs/real-auth-setup.md)：

1. 配置 `mode: real`
2. 任何 GPU/CCU 提交再设 `allow_real_jobs: true`

单独开任一开关都不会提交计费任务。默认 CI 不烧真实 GPU/CCU。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [`SECURITY.md`](SECURITY.md) | 信任边界、漏洞报告 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 开发环境、`make ci`、mock-first |
| [`docs/known-limitations.md`](docs/known-limitations.md) | 限制与 Won't-Fix |
| [`docs/mock-mode.md`](docs/mock-mode.md) | Mock 默认行为 |
| [`docs/real-auth-setup.md`](docs/real-auth-setup.md) | 真实凭据 |
| [`docs/accounts.md`](docs/accounts.md) | 多账号 |
| [`docs/architecture.md`](docs/architecture.md) | 架构概览 |
| [`docs/linux-wsl-setup.md`](docs/linux-wsl-setup.md) | Linux/WSL |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |
| [`config.example.yaml`](config.example.yaml) | 配置模板 |

## CLI 速查

全局选项写在子命令前：`l2kc --config ./l2kc.yaml doctor`。

| 命令 | 作用 |
| --- | --- |
| `l2kc doctor` | 模式、状态目录、CLI 是否存在、鉴权概况（不执行 `kaggle`/`colab`） |
| `l2kc providers` / `quota` / `resources` | Provider 与额度/资源 |
| `l2kc recommend …` | 只推荐后端，不提交 |
| `l2kc submit FILE [--provider auto\|kaggle\|colab\|fake] …` | 提交 `.py` / `.ipynb` |
| `l2kc status` / `logs` / `download` / `cancel` | 任务生命周期 |
| `l2kc accounts …` / `l2kc jobs …` | 账号与本地任务库 |

已知错误在 stderr 打印 `error[<code>]: <message>` 并以退出码 2 结束。完整参数见 `l2kc --help`。

## 限制（简述）

- Mock 额度是夹具；默认 Colab CCU 费率 **未经验证**。
- Kaggle 取消为 `not_supported`（官方 CLI 无法取消运行中的 kernel）。
- 远程 MCP 传输被拒绝；**仅 stdio**。
- 部分高端加速器需真实账号再确认。

完整列表：[`docs/known-limitations.md`](docs/known-limitations.md)。

## 许可证

[Apache License 2.0](./LICENSE)
