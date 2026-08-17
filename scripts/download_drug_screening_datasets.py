#!/usr/bin/env python3
import argparse
import os
import re
import time
from pathlib import Path

import requests


FIGSHARE_COLLECTION_ID = "6528956"
GITHUB_REPO = "eitm-org/organoid_drug_response"
GITHUB_DATA_DIR = "data"
ZENODO_RECORDS = {
    "organoidnetdata": "10643410",
}


def format_bytes(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def sanitize_name(name):
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name.strip("_")


def download_file(url, dest, timeout=(5, 60), chunk_size=8 * 1024 * 1024):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    resume_from = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
        if resp.status_code == 404:
            print(f"[skip] {url} (404)")
            return False
        if resp.status_code == 416:
            print(f"[ok] {dest} already complete")
            return True
        if resume_from > 0 and resp.status_code == 200:
            # Server did not honor Range; restart download.
            resume_from = 0
        resp.raise_for_status()

        content_length = resp.headers.get("Content-Length")
        total = int(content_length) + resume_from if content_length else None
        mode = "ab" if resume_from > 0 else "wb"

        downloaded = resume_from
        last_update = time.monotonic()
        with open(dest, mode) as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if total and now - last_update > 1:
                    pct = (downloaded / total) * 100
                    print(
                        f"[dl] {dest.name} {pct:6.2f}% "
                        f"({format_bytes(downloaded)}/{format_bytes(total)})",
                        end="\r",
                        flush=True,
                    )
                    last_update = now

        if total:
            print(
                f"[ok] {dest.name} 100.00% "
                f"({format_bytes(downloaded)}/{format_bytes(total)})"
            )
        else:
            print(f"[ok] {dest.name} ({format_bytes(downloaded)})")
        return True


def download_figshare_collection(collection_id, out_dir):
    out_dir = Path(out_dir) / f"figshare_{collection_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    page = 1
    while True:
        url = (
            f"https://api.figshare.com/v2/collections/{collection_id}/articles"
            f"?page={page}&page_size=100"
        )
        resp = requests.get(url, timeout=(5, 30))
        resp.raise_for_status()
        articles = resp.json()
        if not articles:
            break

        for article in articles:
            article_id = article["id"]
            detail = requests.get(
                f"https://api.figshare.com/v2/articles/{article_id}",
                timeout=(5, 30),
            ).json()
            title = sanitize_name(detail.get("title", str(article_id)))
            article_dir = out_dir / f"{article_id}_{title}"
            article_dir.mkdir(parents=True, exist_ok=True)

            for f in detail.get("files", []):
                name = f["name"]
                url = f["download_url"]
                print(f"[figshare] {article_id} {name}")
                download_file(url, article_dir / name)

        page += 1


def download_github_data(repo, data_dir, out_dir, token=None):
    out_dir = Path(out_dir) / "github_organoid_drug_response"
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"

    def walk(path, target_dir):
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        resp = requests.get(url, headers=headers, timeout=(5, 30))
        resp.raise_for_status()
        items = resp.json()

        for item in items:
            name = item["name"]
            if item["type"] == "dir":
                walk(item["path"], target_dir / name)
            elif item["type"] == "file":
                download_url = item["download_url"]
                print(f"[github] {item['path']}")
                download_file(download_url, target_dir / name)

    walk(data_dir, out_dir)


def download_zenodo_record(record_id, out_dir):
    out_dir = Path(out_dir) / f"zenodo_{record_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=(5, 30))
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if not files:
        print(f"[warn] No files found for Zenodo record {record_id}")
        return

    for f in files:
        name = f["key"]
        url = f["links"]["self"]
        print(f"[zenodo] {record_id} {name}")
        download_file(url, out_dir / name)


def download_kaggle_datasets(dataset_slugs, out_dir):
    try:
        import kaggle  # noqa: F401
    except Exception:
        print("[kaggle] Kaggle API not installed. Run: pip install kaggle")
        return

    try:
        api = kaggle.api
        api.authenticate()
    except Exception:
        print(
            "[kaggle] Missing credentials. Set ~/.kaggle/kaggle.json "
            "from your Kaggle account."
        )
        return

    out_dir = Path(out_dir) / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    for slug in dataset_slugs:
        slug = slug.strip()
        if not slug:
            continue
        dest = out_dir / slug.replace("/", "_")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[kaggle] {slug}")
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)


def main():
    parser = argparse.ArgumentParser(
        description="Download organoid drug-screening datasets."
    )
    default_base = Path(__file__).resolve().parents[1] / "datasets"
    parser.add_argument(
        "--base-dir",
        default=str(default_base),
        help="Directory to store datasets (default: ./datasets)",
    )
    parser.add_argument("--skip-figshare", action="store_true")
    parser.add_argument("--skip-github", action="store_true")
    parser.add_argument("--skip-zenodo", action="store_true")
    parser.add_argument("--skip-kaggle", action="store_true")
    parser.add_argument(
        "--kaggle-datasets",
        nargs="*",
        default=[],
        help="Kaggle dataset slugs like user/dataset (requires Kaggle API token)",
    )
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token (optional, to avoid rate limits)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_figshare:
        download_figshare_collection(FIGSHARE_COLLECTION_ID, base_dir)
    if not args.skip_github:
        download_github_data(GITHUB_REPO, GITHUB_DATA_DIR, base_dir, args.github_token)
    if not args.skip_zenodo:
        for record_id in ZENODO_RECORDS.values():
            download_zenodo_record(record_id, base_dir)
    if not args.skip_kaggle:
        download_kaggle_datasets(args.kaggle_datasets, base_dir)


if __name__ == "__main__":
    main()
