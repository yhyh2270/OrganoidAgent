from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def select_python() -> Path:
    configured = os.environ.get("ORGANOID_FLUORESCENCE_PYTHON", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(r"D:\app\conda\envs\torch38\python.exe") if os.name == "nt" else None,
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("No fluorescence Python interpreter found; set ORGANOID_FLUORESCENCE_PYTHON")


def main() -> int:
    python = select_python()
    runner = ROOT / "fluorescence_prediction" / "run.py"
    command = [str(python), str(runner), *sys.argv[1:]]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
