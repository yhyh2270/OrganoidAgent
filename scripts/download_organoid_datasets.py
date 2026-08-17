#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import requests


ZENODO_RECORD_DEFAULT = "8177571"
GEO_DEFAULT_SERIES = ["GSE270064", "GSE296369"]
GEO_EXTRA_FILES = {
    "GSE296369": {
        "suppl": ["GSE296369_RNA_seq_matrix_cilia_KO.xlsx"],
    },
}


def format_bytes(num):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def download_file(url, dest, timeout=(5, 60), chunk_size=8 * 1024 * 1024):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    resume_from = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    try:
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
    except requests.RequestException as exc:
        print(f"[error] {url}: {exc}")
        return False


def download_zenodo_record(record_id, base_dir):
    base_dir = Path(base_dir)
    out_dir = base_dir / f"zenodo_{record_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(url, timeout=(5, 30))
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if not files:
        print(f"[warn] No files found for Zenodo record {record_id}")
        return

    for f in files:
        name = f["key"]
        file_url = f["links"]["self"]
        print(f"[zenodo] {name}")
        download_file(file_url, out_dir / name)


def download_geo_series(series_id, base_dir):
    series_id = series_id.upper()
    base_dir = Path(base_dir) / series_id
    prefix = f"{series_id[:-3]}nnn"
    root = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{series_id}"

    files = {
        "miniml": [f"{series_id}_family.xml.tgz"],
        "soft": [f"{series_id}_family.soft.gz"],
        "matrix": [f"{series_id}_series_matrix.txt.gz"],
    }
    files.update(GEO_EXTRA_FILES.get(series_id, {}))

    for subdir, names in files.items():
        for name in names:
            url = f"{root}/{subdir}/{name}"
            dest = base_dir / subdir / name
            print(f"[geo] {series_id} {subdir}/{name}")
            download_file(url, dest)


def main():
    parser = argparse.ArgumentParser(description="Download organoid datasets.")
    default_base = Path(__file__).resolve().parents[1] / "datasets"
    parser.add_argument(
        "--base-dir",
        default=str(default_base),
        help="Directory to store datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--zenodo-record",
        default=ZENODO_RECORD_DEFAULT,
        help="Zenodo record ID to download",
    )
    parser.add_argument(
        "--geo-series",
        nargs="*",
        default=GEO_DEFAULT_SERIES,
        help="GEO series IDs to download",
    )
    parser.add_argument("--skip-zenodo", action="store_true")
    parser.add_argument("--skip-geo", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_zenodo:
        download_zenodo_record(args.zenodo_record, base_dir)
    if not args.skip_geo:
        for series_id in args.geo_series:
            download_geo_series(series_id, base_dir)


if __name__ == "__main__":
    main()
