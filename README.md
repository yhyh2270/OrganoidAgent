# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Analysis](https://img.shields.io/badge/Analysis-Segmentation%20%7C%20Viability-7b2cbf)
[![CI](https://github.com/yhyh2270/OrganoidAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/yhyh2270/OrganoidAgent/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

OrganoidAgent 是一个面向类器官显微图像和实验数据的本地优先分析平台。它把数据集浏览、TIFF/表格预览、类器官分割、形态学定量、荧光活性预测和 Agent 辅助工作流整合在一个 Tornado + PWA 应用中。

> OrganoidAgent is a local-first platform for organoid dataset exploration, image segmentation, morphology analysis, fluorescence-based viability prediction, and agent-assisted workflows.

## 主要功能

| 模块 | 功能 |
| --- | --- |
| 数据浏览 | 索引 `datasets/`，递归浏览文件并预览 CSV、Excel、TIFF、常规图像、压缩包和 AnnData 文件 |
| 形态学分析 | 运行多尺度 Cellpose 工作流，计算面积、尺寸、边缘、曲率及中心性等指标 |
| 活性预测 | 结合 YOLO、SAM 与 ConvNeXt-Tiny，生成类器官掩膜、活性评分和图像证据 |
| 批量报告 | 输出 JSON、CSV、Markdown 报告、裁剪图、掩膜和可视化叠加图 |
| Agent Studio | 支持会话、工作流解析、Codex 作业、状态查询与结果管理 |
| PWA | 无大型前端框架，可从浏览器安装，并缓存静态资源 |

本项目用于科研分析，不应在未经实验验证的情况下将预测结果直接解释为临床结论。

## 处理流程

```text
显微图像
   ├─ YOLO：定位类器官
   ├─ SAM：生成精细实例掩膜
   ├─ 形态特征与质量控制
   └─ ConvNeXt-Tiny：预测活性分数
                   └─ JSON / CSV / Markdown / 掩膜 / 叠加图
```

默认报告将预测结果分为高活性（`>= 0.8`）、中活性（`0.6–0.8`）和低活性（`< 0.6`）；这些阈值只用于结果汇总和排序。

## 项目结构

```text
OrganoidAgent/
├── app.py                       # Tornado API 与 PWA 服务
├── agent_studio.py              # Agent 会话和 Codex 作业管理
├── web/                         # HTML、JavaScript、CSS、Service Worker 和图标
├── fluorescence_prediction/     # 分割、活性推理、特征和报告
│   ├── SAM/                     # Segment Anything 实现
│   └── weights/                 # 本地模型权重（Git 忽略）
├── analysis-tools/              # 多尺度分割与实验分析脚本
├── api-tests/                   # API/分割复现脚本及说明
├── scripts/                     # 环境检查、数据下载和命令行入口
├── config/workflows.json        # 分析环境和工作流配置
├── datasets/                    # 演示数据；完整实验数据与缓存由 Git 忽略
├── analysis-outputs/            # 运行结果（Git 忽略）
├── requirements.txt             # Web 和数据预览依赖
├── requirements-analysis.txt    # 完整分析依赖
└── environment.yml              # Conda 环境定义
```

## 系统要求

- Python 3.10+
- Windows、Linux 或 macOS
- Web 与数据预览可使用 CPU
- 分割和活性预测推荐 NVIDIA GPU；示例 Conda 环境使用 CUDA 12.1
- 完整环境及模型权重需要数 GB 磁盘空间

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yhyh2270/OrganoidAgent.git
cd OrganoidAgent
```

### 2A. 安装完整分析环境（推荐）

```bash
conda env create -f environment.yml
conda activate organoid
python scripts/check_environment.py --profile analysis
```

`environment.yml` 默认安装 CUDA 12.1 版本的 PyTorch。没有 NVIDIA GPU 时，请先将其中的 `pytorch-cuda=12.1` 替换为适合本机的 CPU 配置。

### 2B. 只安装 Web 与数据预览功能

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --profile core
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --profile core
```

在 venv 中增加完整分析功能：

```bash
python -m pip install -r requirements-analysis.txt
```

如需特定 CUDA 版本，先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 `torch` 和 `torchvision`。

## 配置模型权重

模型文件体积较大，不包含在普通 Git 历史中，而是作为 [`models-v1.0.0`](https://github.com/yhyh2270/OrganoidAgent/releases/tag/models-v1.0.0) Release 资源发布。使用完整活性预测前，请将权重放入：

```text
fluorescence_prediction/weights/
├── viability_best.pth
├── yolo_organoid_best.pt
└── sam_vit_b_01ec64.pth
```

### 自动下载并校验（推荐）

推荐在仓库根目录执行内置下载器；它会跳过校验通过的现有文件，并对新下载文件执行大小和 SHA-256 双重校验：

```bash
python scripts/download_model_weights.py
```

如需重新下载全部文件：

```bash
python scripts/download_model_weights.py --force
```

### 手动下载并校验（Windows PowerShell）

也可以使用以下 PowerShell 命令完成同样的下载和校验：

```powershell
$release = "https://github.com/yhyh2270/OrganoidAgent/releases/download/models-v1.0.0"
$weightDir = "fluorescence_prediction\weights"
New-Item -ItemType Directory -Force -Path $weightDir | Out-Null

$weights = @{
  "sam_vit_b_01ec64.pth" = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
  "viability_best.pth"   = "cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b"
  "yolo_organoid_best.pt" = "2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf"
}

foreach ($file in $weights.Keys) {
  $target = Join-Path $weightDir $file
  Invoke-WebRequest "$release/$file" -OutFile $target
  $actual = (Get-FileHash -Algorithm SHA256 $target).Hash.ToLower()
  if ($actual -ne $weights[$file]) { throw "SHA-256 mismatch: $file" }
  Write-Host "Verified $file"
}
```

### 手动下载并校验（Linux/macOS）

在仓库根目录执行：

```bash
set -euo pipefail
release="https://github.com/yhyh2270/OrganoidAgent/releases/download/models-v1.0.0"
weight_dir="fluorescence_prediction/weights"
mkdir -p "$weight_dir"

curl -fL "$release/sam_vit_b_01ec64.pth" -o "$weight_dir/sam_vit_b_01ec64.pth"
curl -fL "$release/viability_best.pth" -o "$weight_dir/viability_best.pth"
curl -fL "$release/yolo_organoid_best.pt" -o "$weight_dir/yolo_organoid_best.pt"

cd "$weight_dir"
printf '%s  %s\n' \
  'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912' 'sam_vit_b_01ec64.pth' \
  'cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b' 'viability_best.pth' \
  '2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf' 'yolo_organoid_best.pt' \
  | sha256sum --check --strict
```

macOS 如果没有 `sha256sum`，可先安装 GNU coreutils（`brew install coreutils`），再将最后一行的 `sha256sum` 改为 `gsha256sum`。

### 文件校验值

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `sam_vit_b_01ec64.pth` | 375,042,383 bytes | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| `viability_best.pth` | 335,956,115 bytes | `cb328bae14ac8d1b77556636e4925e8c7549db83903c15430999c8b8ced9759b` |
| `yolo_organoid_best.pt` | 6,258,019 bytes | `2236790c8e9f027310f63306686215ff6928120b0d59938304e68f965e21b0cf` |

运行以下命令检查依赖、解释器和权重是否就绪：

```bash
python scripts/check_environment.py --profile analysis
```

如果推理或 Cellpose 使用独立环境，可设置：

```powershell
$env:ORGANOID_FLUORESCENCE_PYTHON = "C:\path\to\python.exe"
$env:ORGANOID_CELLPOSE_PYTHON = "C:\path\to\python.exe"
```

Linux/macOS 使用同名环境变量和绝对解释器路径。

未设置环境变量时，应用使用启动它的当前 Python 解释器。`config/workflows.json` 中也可以填写项目部署环境的解释器路径；仓库默认不包含任何特定计算机的绝对路径。

## 启动与使用

```bash
python app.py --port 8080
```

打开 <http://localhost:8080>。首次使用可将 TIFF、CSV、XLSX 或其他支持的文件放到 `datasets/`，刷新页面后即可浏览和预览。

快速验证后端：

```bash
curl http://localhost:8080/api/datasets
curl http://localhost:8080/api/fluorescence/status
```

常用 API：

| 路径 | 用途 |
| --- | --- |
| `GET /api/datasets` | 列出本地数据集 |
| `GET /api/preview?path=...` | 生成文件预览 |
| `GET /api/fluorescence/status` | 检查推理环境与权重 |
| `POST /api/fluorescence/run` | 创建荧光活性预测任务 |
| `POST /api/morphology/run` | 创建形态学分析任务 |
| `GET /api/morphology/status?id=...` | 查询形态学任务状态 |
| `POST /api/morphology/cancel` | 取消形态学任务 |
| `POST /api/agent/chat` | 使用 Agent Studio 会话接口 |

## 命令行活性预测

单张图像：

```bash
python scripts/run_fluorescence_prediction.py \
  --input datasets/example/image.tif \
  --output-dir analysis-outputs/fluorescence_prediction/single
```

批量目录（Windows PowerShell）：

```powershell
python scripts/run_fluorescence_prediction.py `
  --input datasets\example_batch `
  --output-dir analysis-outputs\fluorescence_prediction\batch `
  --order desc
```

主要输出包括 `results.json`、`results.csv`、`report.md`，以及每个样本的裁剪图、掩膜、叠加图和结构化结果。

## 下载示例数据

仓库包含两个可直接运行的精简演示数据集：

- `datasets/01_Density_demo_10x/`：低、中、高密度的 10× 明场 TIFF 示例及清单。
- `datasets/02_Fluorescence_demo/`：用于荧光活性预测的 TIFF 示例。

完整实验数据不会提交到 Git。需要更多公开数据时可按需运行下载脚本，文件将保存到 `datasets/`：

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

部分来源可能要求网络访问、Kaggle 凭据或接受数据提供方的使用条款。

## 开发与验证

修改后至少进行以下检查：

```bash
python -m compileall app.py agent_studio.py fluorescence_prediction scripts
python scripts/check_environment.py --profile core
python app.py --port 8080
```

然后确认：

1. PWA 可在浏览器打开。
2. `/api/datasets` 返回 JSON。
3. CSV、XLSX、TIFF、普通图片和压缩包预览正常。
4. 如已配置分析环境，权重状态显示为 ready，并用少量样本验证推理输出。

## 常见问题

- **页面可打开但没有数据**：确认文件位于仓库的 `datasets/` 中，并刷新页面。
- **推理状态不是 ready**：运行 analysis 环境检查，核对三个权重文件及 Python 环境变量。
- **CUDA/torch 安装失败**：安装与驱动兼容的 PyTorch 构建，或改用 CPU 环境。
- **TIFF/Excel 无法预览**：确认已安装 `imagecodecs`、`tifffile`、`openpyxl` 和 `xlrd`。
- **端口被占用**：改用 `python app.py --port 8081`。

## 数据、隐私与安全

- `datasets/` 中仅跟踪两个演示目录；完整实验数据、`analysis-outputs/`、模型权重、缓存和本地环境默认不会进入 Git。
- 不要把 API 密钥、患者身份信息或受限制的原始数据提交到仓库。
- Agent Studio 可启动本地 Codex 作业；仅在可信环境中运行，并检查其文件访问范围。
- 使用外部数据集和模型时，请遵守各自许可证和数据治理要求。

## 许可证

本项目依据 [Apache License 2.0](LICENSE) 发布。版权归属信息见 [NOTICE](NOTICE)。
