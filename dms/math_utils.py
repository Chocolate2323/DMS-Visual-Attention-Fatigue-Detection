from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Iterable

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_mean(values: Iterable[float], default: float | None = None) -> float | None:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return default
    return float(mean(values))


def smooth_value(previous: float | None, current: float | None, alpha: float) -> float | None:
    if current is None or not np.isfinite(current):
        return previous
    if previous is None or not np.isfinite(previous):
        return float(current)
    return float(alpha * current + (1.0 - alpha) * previous)


def prune_time_window(queue: deque, now_seconds: float, window_seconds: float) -> None:
    while queue and now_seconds - queue[0][0] > window_seconds:
        queue.popleft()
