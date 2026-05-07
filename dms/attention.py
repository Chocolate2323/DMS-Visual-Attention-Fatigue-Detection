from __future__ import annotations

from .math_utils import clip


class AttentionState:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._distracted_started_at: float | None = None
        self._severe_started_at: float | None = None

    def update(
        self,
        now_seconds: float,
        face_found: bool,
        yaw_delta: float | None,
        pitch_delta: float | None,
        roll_delta: float | None,
        gaze_delta_x: float | None,
        gaze_delta_y: float | None,
        missing_face_seconds: float,
    ) -> dict:
        cfg = self.config

        if not face_found:
            head_deviation = 1.0
            gaze_deviation = 1.0
        else:
            yaw_norm = abs(yaw_delta or 0.0) / cfg["yaw_threshold_deg"]
            pitch = pitch_delta or 0.0
            pitch_threshold = cfg["pitch_down_threshold_deg"] if pitch < 0 else cfg["pitch_up_threshold_deg"]
            pitch_norm = abs(pitch) / pitch_threshold
            roll_norm = abs(roll_delta or 0.0) / cfg["roll_threshold_deg"]
            head_deviation = clip(max(yaw_norm, pitch_norm, roll_norm))

            gaze_x = gaze_delta_x or 0.0
            gaze_y = gaze_delta_y or 0.0
            gaze_deviation = clip(((gaze_x * gaze_x + gaze_y * gaze_y) ** 0.5) / cfg["gaze_threshold"])

        missing_face_score = clip(missing_face_seconds / 1.0)
        weights = cfg["weights"]
        distraction_score = clip(
            weights["head"] * head_deviation
            + weights["gaze"] * gaze_deviation
            + weights["missing_face"] * missing_face_score
        )
        attention_score = 1.0 - distraction_score

        candidate = attention_score < cfg["distracted_score_threshold"]
        severe_candidate = attention_score < cfg["severe_distracted_score_threshold"]
        driving_state = self._debounced_state(now_seconds, candidate, severe_candidate)

        return {
            "driving_state": driving_state,
            "attention_score": attention_score,
            "head_deviation": head_deviation,
            "gaze_deviation": gaze_deviation,
            "missing_face_score": missing_face_score,
        }

    def _debounced_state(self, now_seconds: float, candidate: bool, severe_candidate: bool) -> str:
        cfg = self.config

        if severe_candidate:
            if self._severe_started_at is None:
                self._severe_started_at = now_seconds
            if now_seconds - self._severe_started_at >= cfg["severe_hold_seconds"]:
                self._distracted_started_at = now_seconds
                return "distracted"
        else:
            self._severe_started_at = None

        if candidate:
            if self._distracted_started_at is None:
                self._distracted_started_at = now_seconds
            if now_seconds - self._distracted_started_at >= cfg["distracted_hold_seconds"]:
                return "distracted"
            return "normal"

        self._distracted_started_at = None
        return "normal"
