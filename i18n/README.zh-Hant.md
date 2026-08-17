[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent 是一個輕量級的 Tornado 後端與漸進式網頁應用（PWA），可用最少設定在本機瀏覽並預覽類器官資料集。它支援依檔案型別進行預覽轉譯，涵蓋表格、顯微影像（含 TIFF）、封存檔、gzip 文字，以及 AnnData `.h5ad` 分析物件。

## 🎯 一覽

| 目標 | 本專案提供的內容 |
|---|---|
| 本機優先的資料集探索 | 於本機 `datasets/` 工作目錄進行資料集發現、詮釋資料與檔案瀏覽 |
| 豐富預覽 | 表格、影像（含 TIFF）、封存檔、`.gz` 與 `.h5ad` 的預覽流程 |
| 離線友善前端 | 內建可安裝的 PWA 外殼，並搭配 service worker 與 manifest |
| 實用操作能力 | 提供封存解壓與依類別篩選的索引路徑 |

## 概覽 🔭

核心應用以低門檻的互動式資料集探索為設計目標：

- `app.py` 中提供後端 API 與預覽引擎
- `web/` 中提供 PWA 前端
- `scripts/` 中提供下載輔助腳本
- 本機資料集工作區為 `datasets/`（由 git 忽略）

本儲存庫亦包含其他研究與工具工作區（`BioAgent`、`BioAgentUtils`、`references`、`results`、`vendor`、`papers` 子模組）。本 README 主要說明頂層 `OrganoidAgent` 應用的實作。

## 功能 ✨

- 本機資料集索引，並提供大小與檔案數摘要
- 遞迴式資料集檔案列舉，並推斷檔案類型
- 支援 CSV/TSV/XLS/XLSX 表格預覽
- 支援 TIFF/JPG/PNG 圖片預覽
- 支援 `.h5ad` 摘要，並可產生 embedding/PCA 散點圖預覽
- 支援 ZIP/TAR/TGZ 封存檔列舉與首張圖片預覽嘗試
- 支援 `.gz` 文字前段預覽
- 為大型封裝資料集提供封存解壓端點
- 由 Markdown 渲染的資料集層級元資料卡片
- 具備 service worker 與 manifest 的 PWA 前端
- 基本路徑淨化（`safe_dataset_path`）將存取限制在 `datasets/` 下

### 一眼看懂

| 領域 | 提供內容 |
|---|---|
| 資料集發現 | 按目錄層級列出資料集，附帶檔案數與大小摘要 |
| 檔案探索 | 遞迴列舉並推斷檔案類別（`image`、`table`、`analysis`、`archive` 等） |
| 豐富預覽 | 表格、TIFF/圖片、gzip 文字片段、封存內容、AnnData 摘要 |
| 分析視覺化 | 由 `obsm` embeddings 或 PCA 回退產生的 `.h5ad` 散點預覽 |
| 封裝支援 | 封存檔內容列表 + 大型壓縮套件的解壓端點 |
| 網頁體驗 | 可安裝 PWA，並使用有利離線瀏覽的 service worker 資源 |

## 專案結構 🗂️

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

## 前置需求 ✅

- Python `3.10+`
- 建議的環境管理工具：`conda` 或 `venv`

從原始碼推斷的必備／可選 Python 套件：

| 套件 | 作用 |
|---|---|
| `tornado` | 伺服器啟動必需 |
| `pandas` | 可選：表格預覽支援 |
| `anndata`, `numpy` | 可選：`.h5ad` 預覽與分析繪圖 |
| `Pillow` | 可選：影像轉譯與產生預覽 |
| `tifffile` | 可選：TIFF 預覽支援 |
| `requests` | 可選：資料集下載腳本 |
| `kaggle` | 可選：藥物篩選腳本中的 Kaggle 下載 |

備註：目前頂層應用尚未提供根目錄層級的 `requirements.txt`、`pyproject.toml` 或 `environment.yml`。

## 安裝 ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# 方案 A：conda（示例）
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# 方案 B：僅最小執行環境
pip install tornado
```

## 使用方式 🚀

### 快速開始

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # 如已安裝相依套件可略過
python app.py --port 8080
```

開啟 `http://localhost:8080`。

### API 冒煙測試

```bash
curl http://localhost:8080/api/datasets
```

### 下載資料（可選）

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

下載後的資料會放在 `datasets/`（git-ignored）。

## API 端點 🌐

| 方法 | 端點 | 用途 |
|---|---|---|
| `GET` | `/api/datasets` | 列出含摘要統計的資料集 |
| `GET` | `/api/datasets/{name}` | 列出單一資料集的檔案 |
| `GET` | `/api/datasets/{name}/metadata` | 回傳 markdown 元資料卡片 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | 依類別列舉檔案 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | 依檔案型別回傳預覽資料 |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 將封存檔解壓到同層 `_extracted` 資料夾 |
| `GET` | `/files/<path>` | 提供原始資料集檔案服務 |
| `GET` | `/previews/<path>` | 提供已產生的預覽資源 |

預覽呼叫範例：

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 設定 🧩

目前執行設定刻意保持精簡：

- 伺服器埠：`app.py` 中的 `--port` 引數（預設 `8080`）
- 資料目錄：固定為相對倉庫根目錄的 `datasets/`
- 預覽快取：`datasets/.cache/previews`
- 元資料對映：`app.py` 中的 `DATASET_METADATA` 字典
- 下載工具所用 GitHub API token（可選）：`GITHUB_TOKEN` 環境變數或 `--github-token`

備註：若你需要可設定的資料集根目錄或正式環境伺服器參數，目前未在頂層設定檔中提供。

## 範例 🧪

### 瀏覽指定類別檔案

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### 解壓封存檔

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 執行選擇性下載模式

```bash
# 類器官資料集：略過 GEO，保留 Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# 藥物篩選資料集：僅下載 Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 開發備註 🛠️

- 後端從 `web/` 提供前端靜態資源。
- Service worker 與 manifest 位於 `web/sw.js` 與 `web/manifest.json`。
- 檔案型別路由與預覽邏輯實作在 `app.py`。
- 手動驗證（目前專案指引）：PWA 可於 `http://localhost:8080` 載入
- 手動驗證（目前專案指引）：`/api/datasets` 回傳 JSON
- 手動驗證（目前專案指引）：CSV/XLSX/圖片/封存檔可正常渲染預覽

## 疑難排解 🩺

- 預覽函式庫發生 `ModuleNotFoundError`：安裝缺少的套件（`pandas`、`anndata`、`numpy`、`Pillow`、`tifffile`）。
- 資料集清單為空：確認 `datasets/` 下有資料，且目錄未以 `.` 為前綴。
- `.h5ad` 預覽缺少散點圖：請確認已安裝 `anndata`、`numpy` 與 `Pillow`。
- 大型封存檔預覽或解壓問題：使用解壓端點，並直接檢視已解壓的檔案。
- GitHub 下載器遭遇速率限制：透過環境變數或 CLI 參數提供 `GITHUB_TOKEN`。
- Kaggle 下載無法運作：安裝 `kaggle` 並在 `~/.kaggle/kaggle.json` 設定認證。

## 路線圖 🧭

本頂層應用的後續規劃（尚未完全實作）：

- 增加根目錄相依套件清單（`requirements.txt` 或 `pyproject.toml`）
- 為 API handler 與預覽函式補上自動化測試
- 增加可設定的資料集根目錄與快取參數
- 新增明確的正式環境執行設定（非偵錯模式、反向代理指引）
- 擴充 `i18n/` 下的多語言文件

## 貢獻方式 🤝

歡迎參與貢獻。建議的實作流程：

1. Fork 並建立專注分支。
2. 保持變更只聚焦於單一邏輯區塊。
3. 手動驗證應用啟動與關鍵端點。
4. 提交 PR，需包含摘要、執行指令與 UI 變更截圖。

本儲存庫的本地風格規範：

- Python：4 空格縮排，函式與檔名使用 snake_case，類別使用 CapWords
- 本應用前端邏輯保留在 `web/app.js`（避免不必要的框架重寫）
- 僅在邏輯不明顯時，撰寫簡潔註解

## 專案結構（標準摘要） 📌

- `app.py`：Tornado 伺服器與 API 路由。
- `web/`：PWA 資源。
- `scripts/`：資料集下載輔助腳本。
- `datasets/`：本機資料儲存。
- `papers/`：含參考資料的子模組。



## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
