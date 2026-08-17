[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# OrganoidAgent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Tornado-2c7fb8)
![Frontend](https://img.shields.io/badge/Frontend-PWA-0a9396)
![Data](https://img.shields.io/badge/Data-Local%20First-4c956c)
![Format](https://img.shields.io/badge/Preview-Multi--format-f4a261)
![Status](https://img.shields.io/badge/Status-Active-success)

OrganoidAgent là một backend Tornado nhẹ và Progressive Web App (PWA) để duyệt và xem trước các bộ dữ liệu organoid cục bộ với mức thiết lập tối thiểu. Dự án này cung cấp rendering preview có nhận biết loại file cho bảng, ảnh kính hiển vi (kể cả TIFF), kho lưu trữ, văn bản gzip và các đối tượng phân tích AnnData `.h5ad`.

## 🎯 Tóm tắt nhanh

| Mục tiêu | Repo này cung cấp gì |
|---|---|
| Khám phá dữ liệu theo hướng local-first | Khám phá, xem metadata và duyệt tệp từ không gian làm việc local `datasets/` |
| Preview đa dạng | Các đường preview cho bảng, ảnh (kể cả TIFF), kho lưu trữ, `.gz` và `.h5ad` |
| Frontend thân thiện offline | PWA có thể cài đặt với service worker và manifest |
| Tác vụ thực tế | Trích xuất kho lưu trữ + chỉ số phân loại theo danh mục |

## Tổng quan 🔭

Ứng dụng lõi được thiết kế cho việc khám phá dữ liệu tương tác với thiết lập tối thiểu:

- API backend và engine preview trong `app.py`
- Frontend PWA trong `web/`
- Tiện ích tải dữ liệu trong `scripts/`
- Không gian làm việc tập dữ liệu local trong `datasets/` (đã thêm vào .gitignore)

Repo này cũng chứa các workspace nghiên cứu và tiện ích liên quan (`BioAgent`, `BioAgentUtils`, `references`, `results`, `vendor`, submodule `papers`). Runtime chính được mô tả trong README này là ứng dụng `OrganoidAgent` cấp cao nhất.

## Tính năng ✨

- Lập chỉ mục dữ liệu local với tổng hợp kích thước và số lượng tệp
- Liệt kê file dataset đệ quy kèm suy luận loại file
- Hỗ trợ preview cho bảng CSV/TSV/XLS/XLSX
- Hỗ trợ preview cho ảnh TIFF/JPG/PNG
- Hỗ trợ preview cho bản tóm tắt `.h5ad` với tạo ảnh phân tán embedding/PCA
- Hỗ trợ preview cho danh sách nội dung ZIP/TAR/TGZ + thử preview ảnh đầu tiên
- Hỗ trợ preview dòng đầu của văn bản `.gz`
- Endpoint trích xuất kho lưu trữ cho các dataset đóng gói lớn
- Thẻ metadata cấp dataset render từ Markdown
- Frontend PWA kèm service worker và manifest
- Tẩy làm sạch đường dẫn cơ bản (`safe_dataset_path`) để khóa truy cập file trong phạm vi `datasets/`

### Tóm tắt nhanh

| Khu vực | Góp phần gì |
|---|---|
| Khám phá dataset | Danh sách dataset theo thư mục kèm số file và tổng kích thước |
| Khám phá tệp | Liệt kê đệ quy và suy luận loại (`image`, `table`, `analysis`, `archive`,...) |
| Preview đa dạng | Bảng, TIFF/ảnh, đoạn đầu file gzip, nội dung archive, tóm tắt AnnData |
| Trực quan phân tích | Mini-ảnh scatter `.h5ad` từ embedding trong `obsm` hoặc fallback PCA |
| Hỗ trợ đóng gói | Liệt kê archive + endpoint trích xuất cho các gói nén lớn |
| Trải nghiệm web | PWA có thể cài đặt, tài nguyên service worker thân thiện offline |

## Cấu trúc dự án 🗂️

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

## Yêu cầu ✅

- Python `3.10+`
- Công cụ quản lý môi trường đề xuất: `conda` hoặc `venv`

Các gói Python bắt buộc/tùy chọn suy ra từ source:

| Gói | Vai trò |
|---|---|
| `tornado` | Bắt buộc để khởi chạy server |
| `pandas` | Tùy chọn: hỗ trợ preview bảng |
| `anndata`, `numpy` | Tùy chọn: preview `.h5ad` và vẽ biểu đồ phân tích |
| `Pillow` | Tùy chọn: render ảnh và tạo preview |
| `tifffile` | Tùy chọn: hỗ trợ preview TIFF |
| `requests` | Tùy chọn: script tải dataset |
| `kaggle` | Tùy chọn: tải Kaggle trong script sàng lọc thuốc |

Lưu ý giả định: hiện chưa có `requirements.txt`, `pyproject.toml` hoặc `environment.yml` ở root cho app cấp cao nhất.

## Cài đặt ⚙️

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent

# Tuỳ chọn A: conda (ví dụ)
conda create -n organoid python=3.10 -y
conda activate organoid
pip install tornado pandas anndata numpy pillow tifffile requests

# Tuỳ chọn B: runtime tối thiểu
pip install tornado
```

## Sử dụng 🚀

### Bắt đầu nhanh

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
conda activate organoid  # tùy chọn nếu bạn đã cài đủ deps
python app.py --port 8080
```

Mở `http://localhost:8080`.

### Kiểm tra smoke API

```bash
curl http://localhost:8080/api/datasets
```

### Tải dữ liệu (Tùy chọn)

```bash
python scripts/download_organoid_datasets.py
python scripts/download_drug_screening_datasets.py
```

Dữ liệu đã tải sẽ nằm ở `datasets/` (đã được git-ignore).

## API Endpoints 🌐

| Phương thức | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/api/datasets` | Liệt kê datasets với thống kê tóm tắt |
| `GET` | `/api/datasets/{name}` | Liệt kê tệp của một dataset |
| `GET` | `/api/datasets/{name}/metadata` | Trả về thẻ metadata dạng markdown |
| `GET` | `/api/category/{datasets|segmentation|features|analysis}` | Danh sách file theo phân loại |
| `GET` | `/api/preview?path=<relative_path_under_datasets>` | Trả về payload preview theo loại file |
| `POST` | `/api/extract?path=<archive_relative_path_under_datasets>` | Giải nén archive vào thư mục `_extracted` ngang cấp |
| `GET` | `/files/<path>` | Phục vụ file dataset thô |
| `GET` | `/previews/<path>` | Phục vụ tài sản preview đã tạo |

Ví dụ gọi preview:

```bash
curl "http://localhost:8080/api/preview?path=zenodo_10643410/some_file.h5ad"
```

## Cấu hình 🧩

Cấu hình runtime hiện tại được giữ gọn cố ý:

- Cổng server: đối số `--port` trong `app.py` (mặc định `8080`)
- Thư mục dữ liệu: cố định tại `datasets/` so với root repo
- Cache preview: `datasets/.cache/previews`
- Mapping metadata: từ điển `DATASET_METADATA` trong `app.py`
- GitHub API token cho downloader (tùy chọn): biến môi trường `GITHUB_TOKEN` hoặc tùy chọn CLI `--github-token`

Lưu ý giả định: nếu bạn cần cấu hình gốc dataset linh hoạt hơn hoặc cấu hình server production, hiện chưa được expose trong file cấu hình cấp cao.

## Ví dụ 🧪

### Duyệt file theo từng danh mục

```bash
curl http://localhost:8080/api/category/analysis
curl http://localhost:8080/api/category/features
```

### Giải nén một archive

```bash
curl -X POST "http://localhost:8080/api/extract?path=zenodo_8177571/sample_archive.zip"
```

### Chạy chế độ tải chọn lọc

```bash
# Datasets organoid: bỏ qua GEO, giữ Zenodo
python scripts/download_organoid_datasets.py --skip-geo

# Datasets sàng lọc thuốc: chỉ Zenodo
python scripts/download_drug_screening_datasets.py --skip-figshare --skip-github --skip-kaggle
```

## Ghi chú phát triển 🛠️

- Backend phục vụ các tài sản tĩnh frontend từ `web/`.
- Service worker và manifest nằm ở `web/sw.js` và `web/manifest.json`.
- Định tuyến theo loại file và preview được triển khai trong `app.py`.
- Kiểm chứng thủ công (theo guideline hiện tại): PWA load được tại `http://localhost:8080`
- Kiểm chứng thủ công (theo guideline hiện tại): `/api/datasets` trả về JSON
- Kiểm chứng thủ công (theo guideline hiện tại): preview render được cho CSV/XLSX/images/archives

## Khắc phục sự cố 🩺

- `ModuleNotFoundError` cho các thư viện preview: cài thiếu gói (`pandas`, `anndata`, `numpy`, `Pillow`, `tifffile`).
- Danh sách dataset rỗng: xác nhận dữ liệu tồn tại trong `datasets/` và thư mục không bắt đầu bằng dấu chấm.
- Preview `.h5ad` thiếu ảnh scatter: kiểm tra `anndata`, `numpy`, `Pillow` đã được cài chưa.
- Vấn đề preview/trích xuất archive lớn: dùng endpoint trích xuất và kiểm tra trực tiếp file đã giải nén.
- Lỗi giới hạn tần suất của GitHub downloader: truyền `GITHUB_TOKEN` qua biến môi trường hoặc flag CLI.
- Tải Kaggle không hoạt động: cài `kaggle` và cấu hình thông tin xác thực `~/.kaggle/kaggle.json`.

## Lộ trình 🧭

Những cải tiến tiềm năng (chưa được triển khai đầy đủ ở app root):

- Thêm manifest phụ thuộc root (`requirements.txt` hoặc `pyproject.toml`)
- Thêm automated tests cho các API handler và hàm preview
- Thêm cấu hình gốc dataset và cache có thể chỉnh được
- Thêm hồ sơ chạy production rõ ràng hơn (non-debug, hướng dẫn reverse-proxy)
- Mở rộng tài liệu đa ngôn ngữ trong `i18n/`

## Đóng góp 🤝

Đóng góp luôn được chào đón. Quy trình thực tế:

1. Fork và tạo branch tập trung.
2. Giữ thay đổi gói trong một phạm vi logic duy nhất.
3. Kiểm tra thủ công việc khởi động app và các endpoint chính.
4. Mở PR kèm tóm tắt, lệnh đã chạy, và ảnh chụp màn hình cho thay đổi UI.

Quy ước phong cách local của repository:

- Python: thụt lề 4 spaces, hàm/tệp dùng snake_case, lớp dùng CapWords
- Giữ logic frontend trong `web/app.js` cho app này (tránh viết lại bằng framework không cần thiết)
- Giữ comment ngắn gọn và chỉ khi logic không hiển nhiên

## Tổng hợp cấu trúc dự án (bản chuẩn) 📌

- `app.py`: Tornado server và các route API.
- `web/`: tài nguyên PWA.
- `scripts/`: tiện ích tải dataset.
- `datasets/`: nơi lưu trữ dữ liệu local.
- `papers/`: submodule chứa tài liệu tham khảo.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## License 📄

Không có file `LICENSE` cấp cao nhất hiện đang tồn tại trong thư mục gốc repository.

Lưu ý giả định: cho đến khi có thêm giấy phép gốc, coi như điều khoản tái sử dụng/phân phối chưa được chỉ định cho codebase OrganoidAgent cấp cao.
