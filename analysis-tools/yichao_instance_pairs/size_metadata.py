#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import numpy as np


QUANTILE_POINTS: tuple[tuple[str, float], ...] = (
    ("p00", 0.0),
    ("p01", 0.01),
    ("p025", 0.025),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p975", 0.975),
    ("p99", 0.99),
    ("p100", 1.0),
)


def as_float_array(values: Iterable[object]) -> np.ndarray:
    return np.asarray([float(value) for value in values], dtype=np.float64)


def percentile_ranks(values: Iterable[object]) -> np.ndarray:
    arr = as_float_array(values)
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)
    ordered = np.sort(arr)
    return np.searchsorted(ordered, arr, side="right") / float(ordered.size)


def quantile_level_20(percentile: float) -> int:
    clamped = min(1.0, max(0.0, float(percentile)))
    if clamped <= 0.0:
        return 1
    return min(20, max(1, int(np.ceil(clamped * 20.0))))


def quantile_summary(values: Iterable[object]) -> dict[str, float]:
    arr = as_float_array(values)
    if arr.size == 0:
        return {label: float("nan") for label, _ in QUANTILE_POINTS}
    return {label: float(np.quantile(arr, q)) for label, q in QUANTILE_POINTS}
