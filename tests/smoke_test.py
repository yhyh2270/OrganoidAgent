#!/usr/bin/env python3
"""Start OrganoidAgent and verify its public entry points with demo data."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def available_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return response.read()


def main() -> int:
    port = available_port()
    process = subprocess.Popen(
        [sys.executable, "app.py", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        last_error: Exception | None = None
        for _ in range(40):
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"server exited with {process.returncode}:\n{output}")
            try:
                index = fetch(f"http://127.0.0.1:{port}/")
                dataset_payload = json.loads(fetch(f"http://127.0.0.1:{port}/api/datasets"))
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)
        else:
            raise RuntimeError(f"server did not become ready: {last_error}")

        if b"OrganoidAgent" not in index:
            raise AssertionError("PWA index does not contain the application name")
        datasets = dataset_payload.get("datasets", [])
        names = {item["name"] for item in datasets}
        expected = {
            "01_Density_experiment_10x",
            "02_Sodium_alginate_experiment_10x",
            "03_Y-27632_experiment_10x",
            "05_Fluorescence_demo",
        }
        if not expected.issubset(names):
            raise AssertionError(f"missing demo datasets: {sorted(expected - names)}")
        print(f"Smoke test passed on port {port}; datasets: {sorted(names)}")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
