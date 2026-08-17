[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Format-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent は、最小限のセットアップでオルガノイドデータセットをローカルで閲覧・プレビューするための軽量な Tornado バックエンドと Progressive Web App（PWA）です。テーブル、顕微鏡画像（TIFF を含む）、アーカイブ、gzip テキスト、AnnData の `.h5ad` 解析オブジェクトに対し、ファイル種別を意識したプレビュー表示を提供します。

## 🎯 一目でわかること

| 目的 | このリポジトリで得られること |
|---|---|
| ローカル優先のデータセット探索 | `datasets/` ワークスペース配下のローカルデータセットを、メタデータ付きで探索し、ファイルを参照できます |
| 高機能プレビュー | テーブル、画像（TIFF を含む）、アーカイブ、`.gz`、`.h5ad` のプレビュー処理 |
| オフラインを意識したフロントエンド | サービスワーカーとマニフェストを持つ、インストール可能な PWA シェル |
| 実用的な運用 | アーカイブ解凍とカテゴリ絞り込みインデックス |

## 概要 🔭

このコアアプリは、最小限のセットアップで対話的なデータセット探索を行えるよう設計されています。

- `app.py` にバックエンド API とプレビューエンジン
- `web/` に PWA フロントエンド
- `scripts/` にダウンロードヘルパー
- `datasets/` にローカルデータセットワークスペース（git 忽略）

本リポジトリには関連する研究・ユーティリティ領域（`BioAgent`、`BioAgentUtils`、`references`、`results`、`vendor`、`papers` サブモジュール）も含まれます。README で主に説明する実行対象はトップレベルの `OrganoidAgent` アプリです。

## 機能 ✨

- サイズとファイル数の要約付きローカルデータセット索引
- 推定ファイル種別に基づく再帰的なファイル一覧
- CSV/TSV/XLS/XLSX テーブルのプレビュー
- TIFF/JPG/PNG 画像のプレビュー
- `.h5ad` のサマリー表示（embedding/PCA 散布図プレビューの生成付き）
- ZIP/TAR/TGZ アーカイブの一覧表示と先頭画像プレビュー試行
- `.gz` テキストの先頭行プレビュー
- 大規模パッケージデータセット向けアーカイブ展開エンドポイント
- Markdown からレンダリングするデータセット単位のメタデータカード
- サービスワーカーとマニフェストを備えた PWA フロントエンド
- ファイルアクセスを `datasets/` 配下に制限する基本的なパスサニタイズ（`safe_dataset_path`）

### 一目で分かる内容

| 分野 | 提供内容 |
|---|---|
| データセット探索 | ディレクトリ単位でデータセット一覧を表示し、ファイル数とサイズ要約を付与 |
| ファイル探索 | 再帰的な一覧と種別推定（`image`、`table`、`analysis`、`archive` など） |
| 豊富なプレビュー | テーブル、TIFF/画像、gzip テキスト断片、アーカイブ内容、AnnData サマリー |
| 解析ビジュアル | `obsm` 埋め込み、または PCA フォールバックによる `.h5ad` 散布図プレビュー |
| パッケージ対応 | 大きな圧縮バンドル向けのアーカイブ一覧と展開エンドポイント |
| Web UX | オフライン重視のサービスワーカー資産を使うインストール可能な PWA |

## プロジェクト構成 🗂️

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
├─ datasets/                      # ダウンロード済みデータとプレビューキャッシュ（git 無視）
├─ metadata/
│  └─ zenodo_10643410.md
├─ papers/                        # サブモジュール: prompt-is-all-you-need
├─ i18n/                          # 現在は多言語 README ファイルを保持
├─ BioAgent/                      # 関連だが別アプリ
├─ BioAgentUtils/                 # 関連するトレーニング/データユーティリティ
├─ references/
├─ results/
└─ vendor/                        # 外部サブモジュール（copilot-sdk、paper-agent、codex）
```

## 前提条件 ✅

- Python `3.10+`
- 推奨環境管理: `conda` または `venv`

ソースから推定される必須/任意の Python パッケージ:

| パッケージ | 役割 |
|---|---|
| `tornado` | サーバ起動に必要 |
| `pandas` | 任意: テーブルプレビュー対応 |
| `anndata`, `numpy` | 任意: `.h5ad` プレビューと解析プロット |
| `Pillow` | 任意: 画像レンダリングと生成プレビュー |
| `tifffile` | 任意: TIFF プレビュー対応 |
| `requests` | 任意: データセットダウンロードスクリプト |
| `kaggle` | 任意: 薬剤スクリーニングスクリプトの Kaggle ダウンロード |

前提メモ: 現在、トップレベルアプリにはルートの `requirements.txt`、`pyproject.toml`、`environment.yml` はありません。

## インストール ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda（例）
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: 最小構成のみ
pip install tornado
```

## 使い方 🚀

### クイックスタート

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # 依存関係がすでにある場合は任意
python app.py --port 8080
```

`http://localhost:8080` を開いてください。

### API スモークテスト

```bash
curl http://localhost:8080/api/datasets
```

### データのダウンロード（任意）

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

ダウンロード済みデータは `datasets/` に保存されます（git 無視）。

## API エンドポイント 🌐

| メソッド | エンドポイント | 目的 |
|---|---|---|
| `GET` | `/api/datasets` | サマリ統計付きでデータセット一覧を取得 |
| `GET` | `/api/datasets/{name}` | 1 つのデータセットのファイル一覧を取得 |
| `GET` | `/api/datasets/{name}/metadata` | Markdown メタデータカードを返却 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | カテゴリ単位のファイル一覧 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | ファイル種別依存のプレビュー応答を返却 |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 同階層の `_extracted` フォルダへアーカイブ展開 |
| `GET` | `/files/<path>` | 生のデータセットファイル配信 |
| `GET` | `/previews/<path>` | 生成済みプレビュー資産を配信 |

プレビュー呼び出し例:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 設定 🧩

現在の実行設定は意図的に最小限です。

- サーバーポート: `app.py` の `--port` 引数（デフォルト `8080`）
- データディレクトリ: リポジトリルート相対で固定の `datasets/`
- プレビューキャッシュ: `datasets/.cache/previews`
- メタデータマッピング: `app.py` の `DATASET_METADATA` 辞書
- ダウンローダー用 GitHub API トークン（任意）: 環境変数 `GITHUB_TOKEN` または `--github-token`

前提メモ: データセットルートや本番サーバー設定を変更可能にする設定は、現時点ではトップレベルの設定ファイルとして公開されていません。

## 例 🧪

### カテゴリ別ファイルを参照

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### アーカイブを展開

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 選択的ダウンロードモードを実行

```bash
# Organoid データセット: GEO をスキップし、Zenodo のみ残す
python scripts/download_organoid_datasets.py --skip-geo

# Drug-screening データセット: Zenodo のみ使用
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 開発ノート 🛠️

- バックエンドが `web/` からフロントエンド静的アセットを配信します。
- サービスワーカーとマニフェストは `web/sw.js` と `web/manifest.json` にあります。
- ファイル種別ルーティングとプレビュー実装は `app.py` にあります。
- 現行ガイドラインによる手動確認: PWA が `http://localhost:8080` で読み込まれること
- 現行ガイドラインによる手動確認: `/api/datasets` が JSON を返すこと
- 現行ガイドラインによる手動確認: CSV/XLSX/画像/アーカイブがレンダリングされること

## トラブルシューティング 🩺

- プレビューライブラリで `ModuleNotFoundError` が出る: 不足しているパッケージ（`pandas`、`anndata`、`numpy`、`Pillow`、`tifffile`）をインストール
- データセット一覧が空: `datasets/` にデータがあり、ディレクトリ名がドット始まりでないことを確認
- `.h5ad` プレビューで散布図画像が表示されない: `anndata`、`numpy`、`Pillow` がインストール済みか確認
- 大きなアーカイブのプレビュー／展開で問題がある: 展開エンドポイントを使い、展開後のファイルを直接確認
- GitHub ダウンローダーのレート制限エラー: 環境変数または CLI フラグで `GITHUB_TOKEN` を指定
- Kaggle のダウンロードが動作しない: `kaggle` をインストールし、`~/.kaggle/kaggle.json` の認証情報を設定

## ロードマップ 🧭

このルートアプリでまだ完全実装されていない、考えられる次の改善点:

- ルート依存関係マニフェスト（`requirements.txt` または `pyproject.toml`）の追加
- API ハンドラとプレビュー関数に対する自動テストの追加
- データセットルートとキャッシュ設定の可変化
- 明示的な本番運用プロファイルの追加（非デバッグ、リバースプロキシガイド）
- `i18n/` 下で多言語ドキュメントを拡張

## コントリビューション 🤝

コントリビューションは歓迎します。実務的な流れ:

1. フォークして、焦点を絞ったブランチを作成。
2. 変更は1つの論理領域に限定。
3. アプリ起動と主要エンドポイントを手動で検証。
4. 変更内容要約、実行コマンド、UI 変更時はスクリーンショット付きで PR を作成。

このリポジトリのローカルコーディング規約:

- Python: 4 スペースインデント、関数・ファイル名は snake_case、クラスは CapWords
- 本アプリでは `web/app.js` にフロントエンドロジックを置く（不要なフレームワーク再設計を避ける）
- コメントは簡潔にし、ロジックが自明でない箇所のみに記述

## プロジェクト構成（標準要約） 📌

- `app.py`: Tornado サーバーと API ルート
- `web/`: PWA アセット
- `scripts/`: データセットダウンロードヘルパー
- `datasets/`: ローカルデータ保存
- `papers/`: 参考資料を含むサブモジュール

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

現在のリポジトリルートにはトップレベルの `LICENSE` ファイルはありません。

前提メモ: ルートライセンスが追加されるまで、トップレベル `OrganoidAgent` コードベースの再利用・再配布条件は未指定として扱ってください。
