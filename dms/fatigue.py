from __future__ import annotations

from collections import deque

import numpy as np

from . import landmarks as lm
from .math_utils import clip, euclidean, prune_time_window
from .types import FaceObservation


def eye_aspect_ratio(points: np.ndarray) -> float:
    """计算 EAR。数值越小，眼睛越接近闭合。"""
    return (euclidean(points[1], points[5]) + euclidean(points[2], points[4])) / (
        2.0 * max(euclidean(points[0], points[3]), 1e-6)
    )


def mouth_aspect_ratio(points: np.ndarray) -> float:
    """计算 MAR。数值越大，嘴巴张开程度越高。"""
    width = max(euclidean(points[0], points[1]), 1e-6)
    height = euclidean(points[2], points[3])
    return height / width


class FatigueFeatureExtractor:
    """从关键点中提取疲劳检测需要的 EAR 和 MAR。"""

    def extract(self, observation: FaceObservation) -> tuple[float | None, float | None]:
        if not observation.found or observation.landmarks is None:
            return None, None

        width, height = observation.image_size
        landmarks = observation.landmarks

        left_eye = lm.pixel_points(landmarks, lm.LEFT_EYE_EAR, width, height)
        right_eye = lm.pixel_points(landmarks, lm.RIGHT_EYE_EAR, width, height)
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        mouth = lm.pixel_points(
            landmarks,
            [lm.MOUTH_LEFT, lm.MOUTH_RIGHT, lm.MOUTH_TOP, lm.MOUTH_BOTTOM],
            width,
            height,
        )
        mar = mouth_aspect_ratio(mouth)
        return float(ear), float(mar)


class FatigueState:
    """维护疲劳检测的时间窗口和状态机。"""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.closed_frames: deque[tuple[float, bool]] = deque()
        self.blinks: deque[tuple[float, float]] = deque()
        self.long_closures: deque[tuple[float, float]] = deque()
        self.yawns: deque[tuple[float, float]] = deque()
        self.nods: deque[tuple[float, float]] = deque()

        self._closed_started_at: float | None = None
        self._yawn_started_at: float | None = None
        self._nod_started_at: float | None = None
        self._fatigue_started_at: float | None = None
        self._start_seconds: float | None = None

    def update(
        self,
        now_seconds: float,
        ear: float | None,
        mar: float | None,
        pitch_delta: float | None,
        ear_baseline: float | None,
    ) -> dict:
        if self._start_seconds is None:
            self._start_seconds = now_seconds

        cfg = self.config
        eye_threshold = cfg["default_ear_threshold"]
        if ear_baseline is not None and np.isfinite(ear_baseline):
            # 用个人睁眼基线动态生成闭眼阈值，比固定阈值更适合不同眼型。
            eye_threshold = clip(float(ear_baseline) * cfg["ear_closed_ratio"], 0.14, 0.32)

        is_closed = bool(ear is not None and ear < eye_threshold)
        is_yawning = bool(mar is not None and mar > cfg["mar_yawn_threshold"])
        is_nodding = bool(pitch_delta is not None and pitch_delta < -cfg["nod_pitch_threshold_deg"])

        if ear is not None:
            self._update_eye_state(now_seconds, is_closed)
        self._update_yawn_state(now_seconds, is_yawning)
        self._update_nod_state(now_seconds, is_nodding)

        perclos_window = cfg["perclos_window_seconds"]
        yawn_window = cfg["yawn_window_seconds"]
        nod_window = cfg["nod_window_seconds"]
        prune_time_window(self.closed_frames, now_seconds, perclos_window)
        prune_time_window(self.blinks, now_seconds, 60.0)
        prune_time_window(self.long_closures, now_seconds, perclos_window)
        prune_time_window(self.yawns, now_seconds, yawn_window)
        prune_time_window(self.nods, now_seconds, nod_window)

        valid_frames = len(self.closed_frames)
        closed_count = sum(1 for _, closed in self.closed_frames if closed)
        perclos = closed_count / valid_frames if valid_frames else 0.0

        # blink_rate 按最近一分钟折算；视频刚开始时用实际运行时长避免偏低。
        elapsed = max(now_seconds - (self._start_seconds or now_seconds), 1.0)
        blink_window = min(60.0, elapsed)
        blink_rate = len(self.blinks) * 60.0 / blink_window
        avg_blink_duration = (
            sum(duration for _, duration in self.blinks) / len(self.blinks)
            if self.blinks
            else 0.0
        )
        current_closure = now_seconds - self._closed_started_at if self._closed_started_at else 0.0

        perclos_score = clip(perclos / max(cfg["perclos_fatigue_threshold"], 1e-6))
        long_eye_score = clip(max(current_closure, self._max_recent_duration(self.long_closures)) / 1.5)
        yawn_score = clip(len(self.yawns) / 3.0)
        nod_score = clip(len(self.nods) / 3.0)

        weights = cfg["weights"]
        fatigue_score = clip(
            weights["perclos"] * perclos_score
            + weights["long_eye_closure"] * long_eye_score
            + weights["yawn"] * yawn_score
            + weights["head_nod"] * nod_score
        )

        immediate = current_closure >= cfg["immediate_fatigue_eye_closed_seconds"]
        candidate = fatigue_score >= cfg["fatigue_score_threshold"] or immediate
        fatigue_state = self._debounced_state(
            now_seconds,
            candidate,
            cfg["fatigue_hold_seconds"],
            immediate=immediate,
        )

        return {
            "fatigue_state": fatigue_state,
            "fatigue_score": fatigue_score,
            "eye_threshold": eye_threshold,
            "is_eye_closed": is_closed,
            "current_eye_closure_seconds": current_closure,
            "perclos": perclos,
            "blink_rate_per_min": blink_rate,
            "avg_blink_duration": avg_blink_duration,
            "long_eye_closure_count": len(self.long_closures),
            "yawn_count": len(self.yawns),
            "head_nod_count": len(self.nods),
        }

    def _update_eye_state(self, now_seconds: float, is_closed: bool) -> None:
        """根据 OPEN -> CLOSED -> OPEN 的状态变化识别眨眼和长闭眼。"""
        self.closed_frames.append((now_seconds, is_closed))
        if is_closed and self._closed_started_at is None:
            self._closed_started_at = now_seconds
            return

        if not is_closed and self._closed_started_at is not None:
            duration = now_seconds - self._closed_started_at
            if self.config["min_blink_seconds"] <= duration <= self.config["max_blink_seconds"]:
                self.blinks.append((now_seconds, duration))
            if duration >= self.config["long_eye_closure_seconds"]:
                self.long_closures.append((now_seconds, duration))
            self._closed_started_at = None

    def _update_yawn_state(self, now_seconds: float, is_yawning: bool) -> None:
        """嘴巴持续张开超过阈值时记为一次打哈欠。"""
        if is_yawning and self._yawn_started_at is None:
            self._yawn_started_at = now_seconds
            return

        if not is_yawning and self._yawn_started_at is not None:
            duration = now_seconds - self._yawn_started_at
            if duration >= self.config["yawn_min_seconds"]:
                self.yawns.append((now_seconds, duration))
            self._yawn_started_at = None

    def _update_nod_state(self, now_seconds: float, is_nodding: bool) -> None:
        """用 pitch 相对基线的快速下探近似识别点头。"""
        if is_nodding and self._nod_started_at is None:
            self._nod_started_at = now_seconds
            return

        if not is_nodding and self._nod_started_at is not None:
            duration = now_seconds - self._nod_started_at
            if 0.25 <= duration <= 2.0:
                self.nods.append((now_seconds, duration))
            self._nod_started_at = None

    def _debounced_state(
        self,
        now_seconds: float,
        candidate: bool,
        hold_seconds: float,
        immediate: bool = False,
    ) -> str:
        if immediate:
            self._fatigue_started_at = now_seconds
            return "fatigue"

        if candidate:
            if self._fatigue_started_at is None:
                self._fatigue_started_at = now_seconds
            if now_seconds - self._fatigue_started_at >= hold_seconds:
                return "fatigue"
            return "normal"

        self._fatigue_started_at = None
        return "normal"

    @staticmethod
    def _max_recent_duration(events: deque[tuple[float, float]]) -> float:
        if not events:
            return 0.0
        return max(duration for _, duration in events)
