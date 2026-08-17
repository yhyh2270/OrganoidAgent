[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent는 최소한의 설정으로 로컬에서 오가노이드 데이터셋을 탐색하고 미리보기할 수 있는 경량 Tornado 백엔드 및 PWA(Progressive Web App)입니다. 표 형식, 현미경 이미지(TIFF 포함), 압축 아카이브, gzip 텍스트, AnnData `.h5ad` 분석 객체를 파일 형식별 미리보기 렌더링으로 지원합니다.

## 🎯 한눈에 보기

| 목표 | 이 저장소가 제공하는 내용 |
|---|---|
| 로컬 우선 데이터셋 탐색 | 로컬 `datasets/` 작업공간의 데이터셋 탐색, 메타데이터, 파일 탐색 |
| 풍부한 미리보기 | 테이블, 이미지(TIFF 포함), 아카이브, `.gz`, `.h5ad` 미리보기 경로 |
| 오프라인 친화형 프런트엔드 | 서비스 워커와 매니페스트가 포함된 설치형 PWA 셸 |
| 실용적인 동작 | 아카이브 추출 + 카테고리 필터링 인덱싱 경로 |

## 개요 🔭

이 핵심 앱은 최소한의 설정으로 대화형 데이터셋 탐색을 위해 설계되었습니다.

- 백엔드 API 및 미리보기 엔진: `app.py`
- PWA 프런트엔드: `web/`
- 다운로드 보조 스크립트: `scripts/`
- 로컬 데이터셋 작업공간: `datasets/` (git-ignored)

이 저장소에는 인접한 연구/유틸리티 작업공간도 포함되어 있습니다(`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, `papers` 서브모듈). 이 README에서 주로 다루는 런타임은 최상위 `OrganoidAgent` 앱입니다.

## 기능 ✨

- 크기와 파일 개수 요약이 있는 로컬 데이터셋 인덱싱
- 추론된 파일 유형과 함께 재귀적 데이터셋 파일 목록 표시
- CSV/TSV/XLS/XLSX 테이블 미리보기 지원
- TIFF/JPG/PNG 이미지 미리보기 지원
- 임베딩/PCA 산점도 생성이 가능한 `.h5ad` 요약 미리보기 지원
- ZIP/TAR/TGZ 아카이브 목록 + 첫 번째 이미지 미리보기 시도 지원
- `.gz` 텍스트 선두 라인 미리보기 지원
- 대형 패키지 데이터셋을 위한 아카이브 추출 엔드포인트
- 마크다운으로 렌더링되는 데이터셋 수준 메타데이터 카드
- 서비스 워커와 매니페스트가 포함된 PWA 프런트엔드
- 파일 접근을 `datasets/` 하위로 제한하는 기본 경로 정리(`safe_dataset_path`)

### 한눈에 보기

| 영역 | 제공 내용 |
|---|---|
| 데이터셋 탐색 | 파일 수와 크기 요약이 포함된 디렉터리 단위 목록 |
| 파일 탐색 | 재귀적 목록 및 유형 추론 (`image`, `table`, `analysis`, `archive` 등) |
| 풍부한 미리보기 | 테이블, TIFF/이미지, gzip 텍스트 조각, 아카이브 내용, AnnData 요약 |
| 분석 시각화 | `obsm` 임베딩 또는 PCA 대체값 기반 `.h5ad` 산점도 미리보기 |
| 패키징 지원 | 대용량 압축 번들에 대한 아카이브 목록 + 추출 엔드포인트 |
| 웹 UX | 오프라인 친화형 자산을 제공하는 설치형 PWA |

## 프로젝트 구조 🗂️

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

## 선행 조건 ✅

- Python `3.10+`
- 권장 환경 관리자: `conda` 또는 `venv`

소스에서 추론되는 필수/선택 패키지:

| 패키지 | 역할 |
|---|---|
| `tornado` | 서버 실행 필수 |
| `pandas` | 선택: 테이블 미리보기 지원 |
| `anndata`, `numpy` | 선택: `.h5ad` 미리보기 및 분석 플로팅 |
| `Pillow` | 선택: 이미지 렌더링 및 생성 미리보기 |
| `tifffile` | 선택: TIFF 미리보기 지원 |
| `requests` | 선택: 데이터셋 다운로드 스크립트 |
| `kaggle` | 선택: drug-screening 스크립트의 Kaggle 다운로드 |

참고: 현재 최상위 앱에는 `requirements.txt`, `pyproject.toml`, `environment.yml`가 없습니다.

## 설치 ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Option A: conda (예시)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Option B: 최소 런타임만 실행
pip install tornado
```

## 사용법 🚀

### 빠른 시작

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # 이미 의존성이 설치되어 있으면 생략 가능
python app.py --port 8080
```

`http://localhost:8080`을 열어주세요.

### API 스모크 테스트

```bash
curl http://localhost:8080/api/datasets
```

### 데이터 다운로드 (선택)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

다운로드된 데이터는 `datasets/`에 저장됩니다( git-ignored ).

## API 엔드포인트 🌐

| 메서드 | 엔드포인트 | 목적 |
|---|---|---|
| `GET` | `/api/datasets` | 요약 통계를 포함한 데이터셋 목록 |
| `GET` | `/api/datasets/{name}` | 특정 데이터셋의 파일 목록 |
| `GET` | `/api/datasets/{name}/metadata` | 마크다운 메타데이터 카드 반환 |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | 카테고리 기반 파일 목록 |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | 파일 유형 기반 미리보기 페이로드 |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | 형제 경로 `_extracted` 폴더로 아카이브 추출 |
| `GET` | `/files/<path>` | 원시 데이터셋 파일 제공 |
| `GET` | `/previews/<path>` | 생성된 미리보기 자산 제공 |

미리보기 호출 예시:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## 설정 🧩

현재 런타임 설정은 의도적으로 간단합니다.

- 서버 포트: `app.py`의 `--port` 인자 (기본값 `8080`)
- 데이터 디렉터리: 저장소 루트 기준 상대 경로 `datasets/`
- 미리보기 캐시: `datasets/.cache/previews`
- 메타데이터 매핑: `app.py`의 `DATASET_METADATA` 딕셔너리
- 다운로드어용 GitHub API 토큰(선택): 환경 변수 `GITHUB_TOKEN` 또는 `--github-token`

참고: 데이터셋 루트나 운영용 서버 설정을 자유롭게 바꾸려면, 최상위 설정 파일로 아직 노출되지 않았습니다.

## 사용 예시 🧪

### 카테고리별 파일 탐색

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### 아카이브 추출 실행

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### 선택적 다운로드 모드 실행

```bash
# 오가노이드 데이터셋: GEO는 건너뛰고 Zenodo만 유지
python scripts/download_organoid_datasets.py --skip-geo

# drug-screening 데이터셋: Zenodo만 사용
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## 개발 노트 🛠️

- 백엔드는 `web/`에서 프런트엔드 정적 자산을 제공합니다.
- 서비스 워커와 매니페스트는 `web/sw.js`와 `web/manifest.json`에 있습니다.
- 파일 유형 라우팅과 미리보기 로직은 `app.py`에 구현되어 있습니다.
- 수동 검증(현재 프로젝트 지침): PWA가 `http://localhost:8080`에서 로드되는지 확인
- 수동 검증(현재 프로젝트 지침): `/api/datasets`가 JSON을 반환하는지 확인
- 수동 검증(현재 프로젝트 지침): CSV/XLSX/이미지/아카이브 미리보기가 렌더링되는지 확인

## 문제 해결 🩺

- 미리보기 라이브러리의 `ModuleNotFoundError`: 누락 패키지(`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`) 설치
- 데이터셋 목록이 비어 있음: `datasets/` 아래에 데이터가 있고 디렉터리가 dot(`.`)로 시작하지 않았는지 확인
- `.h5ad` 미리보기가 산점도 이미지를 표시하지 않음: `anndata`, `numpy`, `Pillow` 설치 여부 확인
- 대형 아카이브 미리보기/추출 문제: 추출 엔드포인트를 사용하고 추출된 파일을 직접 확인
- GitHub 다운로드 속도 제한 오류: 환경 변수 또는 CLI 플래그로 `GITHUB_TOKEN` 제공
- Kaggle 다운로드 작동 안 함: `kaggle` 설치 및 `~/.kaggle/kaggle.json` 자격증명 설정

## 로드맵 🧭

현재 루트 앱에 아직 완전히 반영되지 않은 개선 사항:

- 루트 종속성 매니페스트 추가 (`requirements.txt` 또는 `pyproject.toml`)
- API 핸들러와 미리보기 함수 자동 테스트 추가
- 데이터셋 루트 및 캐시 설정을 구성 가능하게 설정
- 운영 환경 실행 프로파일(비-디버그, 역방향 프록시 가이드) 추가
- `i18n/` 하위 다국어 문서 확장

## 기여하기 🤝

기여를 환영합니다. 실무적인 작업 흐름:

1. 포크 후 집중 브랜치 생성
2. 변경 사항을 하나의 논리적 영역으로 제한
3. 앱 시작 및 핵심 엔드포인트 수동 검증
4. UI 변경 사항은 요약/실행 명령/스크린샷과 함께 PR 제출

저장소의 로컬 스타일 규칙:

- Python: 4칸 들여쓰기, 함수/파일은 snake_case, 클래스는 CapWords
- 본 앱에서 프런트엔드 로직은 `web/app.js`에 유지(불필요한 프레임워크 리라이트 지양)
- 주석은 핵심 로직만 간결하게 유지

## 프로젝트 레이아웃 (표준 요약) 📌

- `app.py`: Tornado 서버와 API 라우트.
- `web/`: PWA 자산.
- `scripts/`: 데이터셋 다운로드 보조 스크립트.
- `datasets/`: 로컬 데이터 저장.
- `papers/`: 참고 자료가 들어 있는 서브모듈.

## 라이선스 📄

최상위 프로젝트에 현재 루트 `LICENSE` 파일이 없습니다.

참고: 루트 라이선스가 추가될 때까지 최상위 OrganoidAgent 코드베이스의 재사용/재배포 조건은 미정의입니다.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |
