from __future__ import annotations

from collections import deque

import numpy as np

from .attention import AttentionState
from .fatigue import FatigueState
from .math_utils import safe_mean, smooth_value
from .types import DMSResult, FrameFeatures


class DMSState:
    """管理跨帧状态：基线校准、平滑、注意力判断和疲劳判断。"""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.attention = AttentionState(config["attention"])
        self.fatigue = FatigueState(config["fatigue"])

        self.calibration_seconds = config["runtime"]["calibration_seconds"]
        self.smoothing_alpha = config["runtime"]["smoothing_alpha"]
        self.max_missing_face_seconds = config["runtime"]["max_missing_face_seconds"]

        self._start_seconds: float | None = None
        self._last_face_seen_seconds: float | None = None
        self._calibration_samples: deque[FrameFeatures] = deque()
        self._calibration_ready = False
        self._baseline: dict[str, float | None] = {
            "yaw": None,
            "pitch": None,
            "roll": None,
            "gaze_x": None,
            "gaze_y": None,
            "ear": None,
        }
        self._smoothed: dict[str, float | None] = {
            "yaw": None,
            "pitch": None,
            "roll": None,
            "gaze_x": None,
            "gaze_y": None,
            "ear": None,
            "mar": None,
        }

    def update(self, features: FrameFeatures) -> DMSResult:
        now_seconds = features.timestamp_ms / 1000.0
        if self._start_seconds is None:
            self._start_seconds = now_seconds

        self._update_smoothed_features(features)
        smoothed = self._smoothed

        if features.face_found:
            self._last_face_seen_seconds = now_seconds
            self._maybe_collect_calibration(features)

        missing_face_seconds = 0.0
        if self._last_face_seen_seconds is None:
            missing_face_seconds = now_seconds - (self._start_seconds or now_seconds)
        elif not features.face_found:
            missing_face_seconds = now_seconds - self._last_face_seen_seconds

        if missing_face_seconds >= self.max_missing_face_seconds:
            face_found_for_attention = False
        else:
            face_found_for_attention = features.face_found

        yaw_delta = self._delta("yaw")
        pitch_delta = self._delta("pitch")
        roll_delta = self._delta("roll")
        gaze_delta_x = self._delta("gaze_x")
        gaze_delta_y = self._delta("gaze_y")

        if not self._calibration_ready and features.face_found:
            # 校准期默认驾驶员在正常看前方，不用原始绝对角度触发分神。
            yaw_delta = 0.0
            pitch_delta = 0.0
            roll_delta = 0.0
            gaze_delta_x = 0.0
            gaze_delta_y = 0.0

        attention_result = self.attention.update(
            now_seconds=now_seconds,
            face_found=face_found_for_attention,
            yaw_delta=yaw_delta,
            pitch_delta=pitch_delta,
            roll_delta=roll_delta,
            gaze_delta_x=gaze_delta_x,
            gaze_delta_y=gaze_delta_y,
            missing_face_seconds=missing_face_seconds,
        )
        fatigue_result = self.fatigue.update(
            now_seconds=now_seconds,
            ear=smoothed["ear"],
            mar=smoothed["mar"],
            pitch_delta=pitch_delta,
            ear_baseline=self._baseline["ear"],
        )

        result_features = {
            "yaw": self._rounded(smoothed["yaw"]),
            "pitch": self._rounded(smoothed["pitch"]),
            "roll": self._rounded(smoothed["roll"]),
            "yaw_delta": self._rounded(yaw_delta),
            "pitch_delta": self._rounded(pitch_delta),
            "roll_delta": self._rounded(roll_delta),
            "gaze_x": self._rounded(smoothed["gaze_x"]),
            "gaze_y": self._rounded(smoothed["gaze_y"]),
            "gaze_delta_x": self._rounded(gaze_delta_x),
            "gaze_delta_y": self._rounded(gaze_delta_y),
            "ear": self._rounded(smoothed["ear"], 4),
            "mar": self._rounded(smoothed["mar"], 4),
            "perclos_30s": self._rounded(fatigue_result["perclos"], 4),
            "blink_rate_per_min": self._rounded(fatigue_result["blink_rate_per_min"], 2),
            "avg_blink_duration": self._rounded(fatigue_result["avg_blink_duration"], 3),
            "current_eye_closure_seconds": self._rounded(
                fatigue_result["current_eye_closure_seconds"],
                3,
            ),
            "long_eye_closure_count": fatigue_result["long_eye_closure_count"],
            "yawn_count": fatigue_result["yawn_count"],
            "head_nod_count": fatigue_result["head_nod_count"],
            "missing_face_seconds": self._rounded(missing_face_seconds, 3),
            "head_deviation": self._rounded(attention_result["head_deviation"], 4),
            "gaze_deviation": self._rounded(attention_result["gaze_deviation"], 4),
        }

        return DMSResult(
            timestamp_ms=features.timestamp_ms,
            driving_state=attention_result["driving_state"],
            fatigue_state=fatigue_result["fatigue_state"],
            attention_score=attention_result["attention_score"],
            fatigue_score=fatigue_result["fatigue_score"],
            features=result_features,
            calibration_ready=self._calibration_ready,
        )

    @property
    def baseline(self) -> dict[str, float | None]:
        return dict(self._baseline)

    def _update_smoothed_features(self, features: FrameFeatures) -> None:
        """对连续特征做指数平滑，降低关键点检测噪声。"""
        for name in self._smoothed:
            current = getattr(features, name)
            self._smoothed[name] = smooth_value(
                self._smoothed[name],
                current,
                self.smoothing_alpha,
            )

    def _maybe_collect_calibration(self, features: FrameFeatures) -> None:
        """收集开头几秒的正常驾驶样本，作为个人基线。"""
        if self._calibration_ready:
            return

        now_seconds = features.timestamp_ms / 1000.0
        self._calibration_samples.append(features)

        if self._start_seconds is None or now_seconds - self._start_seconds < self.calibration_seconds:
            return

        self._baseline = {
            "yaw": safe_mean([s.yaw for s in self._calibration_samples], 0.0),
            "pitch": safe_mean([s.pitch for s in self._calibration_samples], 0.0),
            "roll": safe_mean([s.roll for s in self._calibration_samples], 0.0),
            "gaze_x": safe_mean([s.gaze_x for s in self._calibration_samples], 0.0),
            "gaze_y": safe_mean([s.gaze_y for s in self._calibration_samples], 0.0),
            "ear": safe_mean([s.ear for s in self._calibration_samples], None),
        }
        self._calibration_ready = True

    def _delta(self, name: str) -> float | None:
        """返回当前平滑值相对个人基线的偏移。"""
        value = self._smoothed.get(name)
        if value is None or not np.isfinite(value):
            return None

        baseline = self._baseline.get(name)
        if baseline is None or not np.isfinite(baseline):
            baseline = 0.0
        return float(value - baseline)

    @staticmethod
    def _rounded(value: float | None, digits: int = 2) -> float | None:
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), digits)
