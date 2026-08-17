[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent 是一个轻量级的 Tornado 后端和渐进式 Web 应用（PWA），用于以最少配置在本地浏览和预览类器官数据集。它按文件类型提供预览渲染，涵盖表格、显微镜图像（包括 TIFF）、压缩包、gzip 文本以及 AnnData `.h5ad` 分析对象。

## 🎯 一览

| 目标 | 仓库提供的内容 |
|---|---|
| 本地优先的数据集探索 | 在本地 `datasets/` 工作区中进行数据集发现、元数据查看和文件浏览 |
| 丰富预览 | 表格、图像（含 TIFF）、压缩包、`.gz` 和 `.h5ad` 的预览路径 |
| 离线友好前端 | 内置可安装的 PWA 外壳，并配套 service worker 与 manifest |
| 实用操作能力 | 提供归档解压与按类别过滤的索引路径 |

## Overview 🔭

核心应用面向交互式数据集探索设计，且部署门槛极低：

- 后端 API 与预览引擎在 `app.py` 中
- PWA 前端在 `web/`
- 下载辅助工具在 `scripts/`
- 本地数据工作区为 `datasets/`（由 git 忽略）

本仓库还包含相关的研究与工具工作区（`BioAgent`、`BioAgentUtils`、`references`、`results`、`vendor`、`papers` 子模块）。本 README 主要描述顶层 `OrganoidAgent` 应用的运行方式。

## Features ✨

- 本地数据集索引并提供大小与文件数量汇总
- 递归列出数据集文件，并推断文件类型
- CSV/TSV/XLS/XLSX 表格预览支持
- TIFF/JPG/PNG 图像预览支持
- `.h5ad` 摘要支持，支持 embedding/PCA 的散点图预览生成
- ZIP/TAR/TGZ 压缩包内容列表示例 + 首张图像预览尝试
- `.gz` 文本前几行预览支持
- 为大型打包数据集提供归档解压端点
- 从 Markdown 渲染数据集级元数据卡片
- 带有 service worker 与 manifest 的 PWA 前端
- 基础路径清洗（`safe_dataset_path`）确保文件访问限制在 `datasets/` 下

### 一眼看懂

| 范围 | 提供能力 |
|---|---|
| 数据集发现 | 按目录层级展示数据集列表，并给出文件数与大小汇总 |
| 文件探索 | 递归列出并推断文件类型（`image`、`table`、`analysis`、`archive` 等） |
| 丰富预览 | 表格、TIFF/图片、gzip 文本片段、压缩包内容、AnnData 摘要 |
| 分析可视化 | 基于 `obsm` embeddings 或 PCA 回退的 `.h5ad` 散点预览 |
| 打包支持 | 压缩包列表展示 + 大型压缩数据的解压端点 |
| Web 体验 | 可安装的 PWA，配套离线友好的 service worker 资源 |

## Project Structure 🗂️

```text
OrganoidAgent/
├─ app.py
├─ web/
│  ├─ index.html
│  ├─ app.js
│  ├─ styles.css
│  ├─ sw.js
│  ├─ manifest.json
│  └─ icons/
├─ scripts/
│  ├─ download_organoid_datasets.py
│  ├─ download_drug_screening_datasets.py
│  └─ overlay_segmentations.py
├─ datasets/                      # downloaded data and preview cache (git-ignored)
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # submodule: prompt-is-all-you-need
├─ i18n/                          # currently present for multilingual README files
├─ BioAgent/                      # related but separate app
├─ BioAgentUtils/                 # related training/data utilities
├─ references/
├─ results/
└─ vendor/                        # external submodules (copilot-sdk, paper-agent, codex)
```

## Prerequisites ✅

- Python `3.10+`
- 推荐的环境管理工具：`conda` 或 `venv`

从源码推断的必需/可选 Python 包：

| 包 | 作用 |
|---|---|
| `tornado` | 启动服务所需 |
| `pandas` | 可选：表格预览支持 |
| `anndata`, `numpy` | 可选：`.h5ad` 预览与分析绘图 |
| `Pillow` | 可选：图像渲染及生成预览 |
| `tifffile` | 可选：TIFF 预览支持 |
| `requests` | 可选：数据集下载脚本 |
| `kaggle` | 可选：药物筛选脚本中的 Kaggle 下载 |

说明：目前顶层应用还没有根目录级别的 `requirements.txt`、`pyproject.toml` 或 `environment.yml`。

## Installation ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# 方案 A：conda（示例）
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# 方案 B：仅最小运行环境
pip install tornado
```

## Usage 🚀

### Quick Start

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # 如果已安装依赖可省略
python app.py --port 8080
```

打开 `http://localhost:8080`。

### API Smoke Test

```bash
curl http://localhost:8080/api/datasets
```

### Download Data (Optional)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

下载的数据保存在 `datasets/`（git-ignored）。

## API Endpoints 🌐

| 方法 | 端点 | 用途 |
|---|---|---|
| `GET` | `/api/datasets` | 列出包含汇总统计的数据集 |
| `GET` | `/api/datasets/{name}` | 列出某个数据集的全部文件 |
| `GET` | `/api/datasets/{name}/metadata` | 返回 Markdown 元数据卡片 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | 按类别列出文件 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | 按文件类型返回预览负载 |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 将压缩包解压到同级 `_extracted` 目录 |
| `GET` | `/files/<path>` | 原始数据集文件服务 |
| `GET` | `/previews/<path>` | 已生成预览资源服务 |

示例预览调用：

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Configuration 🧩

当前运行配置有意保持简洁：

- 服务器端口：`app.py` 中的 `--port` 参数（默认 `8080`）
- 数据目录：仓库根路径下固定为 `datasets/`
- 预览缓存：`datasets/.cache/previews`
- 元数据映射：`app.py` 中的 `DATASET_METADATA` 字典
- 下载器的 GitHub API token（可选）：`GITHUB_TOKEN` 环境变量或 `--github-token`

说明：如果你需要可配置的数据集根目录或生产环境服务器设置，目前这些尚未在顶层配置文件中暴露。

## Examples 🧪

### Browse category-specific files

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Extract an archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Run selective download modes

```bash
# Organoid datasets: skip GEO, keep Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening datasets: only Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Development Notes 🛠️

- 后端从 `web/` 提供前端静态资源。
- Service worker 与 manifest 位于 `web/sw.js` 和 `web/manifest.json`。
- 文件类型路由与预览逻辑实现在 `app.py`。
- 手动验证（当前项目说明）：PWA 在 `http://localhost:8080` 加载成功
- 手动验证（当前项目说明）：`/api/datasets` 返回 JSON
- 手动验证（当前项目说明）：CSV/XLSX/图像/压缩包可正常预览渲染

## Troubleshooting 🩺

- 预览库报 `ModuleNotFoundError`：安装缺失包（`pandas`、`anndata`、`numpy`、`Pillow`、`tifffile`）。
- 数据集列表为空：确认 `datasets/` 下有数据，且目录未使用点号作为前缀。
- `.h5ad` 预览缺少散点图：检查 `anndata`、`numpy`、`Pillow` 是否已安装。
- 大型压缩包预览/解压异常：使用解压端点并直接检查解压后的文件。
- GitHub 下载器出现速率限制：通过环境变量或 CLI 参数提供 `GITHUB_TOKEN`。
- Kaggle 下载无法运行：安装 `kaggle` 并在 `~/.kaggle/kaggle.json` 配置凭据。

## Roadmap 🧭

该顶层应用的下一步改进方向（目前尚未完全实现）：

- 增加根目录依赖清单（`requirements.txt` 或 `pyproject.toml`）
- 为 API 处理器与预览函数补充自动化测试
- 增加可配置的数据集根目录和缓存参数
- 增加明确的生产环境运行方案（非 debug 模式、反向代理说明）
- 扩展 `i18n/` 下的多语言文档

## Contributing 🤝

欢迎参与贡献。建议的实用流程：

1. Fork 并创建聚焦分支。
2. 保持变更仅覆盖一个逻辑范围。
3. 手动验证应用启动与关键端点。
4. 提交 PR，包含摘要、执行命令与 UI 变更截图。

本仓库本地风格规范：

- Python：4 空格缩进，函数/文件名使用 snake_case，类名使用 CapWords
- 本应用前端逻辑保留在 `web/app.js` 中（避免不必要的框架重写）
- 仅在逻辑不明显时写简明注释

## Project Layout (Canonical Summary) 📌

- `app.py`：Tornado 服务器与 API 路由。
- `web/`：PWA 资源。
- `scripts/`：数据集下载辅助脚本。
- `datasets/`：本地数据存储。
- `papers/`：包含参考资料的子模块。

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

目前仓库根目录尚未包含顶层项目 `LICENSE` 文件。

说明：在补充根级许可证前，顶层 OrganoidAgent 代码库的复用和再分发条款暂未明确。
