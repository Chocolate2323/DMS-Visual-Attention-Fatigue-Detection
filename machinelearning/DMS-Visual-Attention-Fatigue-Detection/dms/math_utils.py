from __future__ import annotations

from collections import deque
from statistics import mean
from typing import Iterable

import numpy as np


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """二维/三维点之间的欧氏距离。"""
    return float(np.linalg.norm(a - b))


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """把数值限制在指定区间内。"""
    return max(low, min(high, value))


def safe_mean(values: Iterable[float], default: float | None = None) -> float | None:
    """忽略 None 和非有限值后求平均，常用于基线校准。"""
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return default
    return float(mean(values))


def smooth_value(previous: float | None, current: float | None, alpha: float) -> float | None:
    """指数平滑，降低逐帧关键点抖动。"""
    if current is None or not np.isfinite(current):
        return previous
    if previous is None or not np.isfinite(previous):
        return float(current)
    return float(alpha * current + (1.0 - alpha) * previous)


def prune_time_window(queue: deque, now_seconds: float, window_seconds: float) -> None:
    """移除滑动窗口外的旧事件。"""
    while queue and now_seconds - queue[0][0] > window_seconds:
        queue.popleft()
