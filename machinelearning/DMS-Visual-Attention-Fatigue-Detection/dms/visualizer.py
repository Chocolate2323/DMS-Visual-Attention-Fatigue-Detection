from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lm
from .types import DMSResult, FrameFeatures


class Visualizer:
    """在输出视频上绘制关键点、头部坐标轴和状态面板。"""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.draw_landmarks = config["visualization"]["draw_landmarks"]
        self.draw_head_axis = config["visualization"]["draw_head_axis"]
        self.scale = config["visualization"]["overlay_scale"]

    def draw(self, frame: np.ndarray, features: FrameFeatures, result: DMSResult) -> np.ndarray:
        canvas = frame.copy()
        self._draw_driver_bbox(canvas, features)
        if self.draw_landmarks:
            self._draw_landmarks(canvas, features)
        if self.draw_head_axis:
            self._draw_head_axis(canvas, features)
        self._draw_overlay(canvas, features, result)
        return canvas

    def _draw_driver_bbox(self, frame: np.ndarray, features: FrameFeatures) -> None:
        driver_bbox = features.extra.get("driver_bbox")
        if driver_bbox is None:
            return

        x1, y1, x2, y2 = [int(value) for value in driver_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

    def _draw_landmarks(self, frame: np.ndarray, features: FrameFeatures) -> None:
        """绘制少量用于调试的眼部、嘴部和头姿关键点。"""
        landmarks = features.extra.get("landmarks")
        image_size = features.extra.get("image_size")
        if landmarks is None or image_size is None:
            return

        width, height = image_size
        indices = set(lm.LEFT_EYE_EAR + lm.RIGHT_EYE_EAR + lm.LEFT_IRIS + lm.RIGHT_IRIS)
        indices.update([lm.NOSE_TIP, lm.CHIN, lm.LEFT_MOUTH, lm.RIGHT_MOUTH, lm.MOUTH_TOP, lm.MOUTH_BOTTOM])
        for idx in indices:
            if idx >= len(landmarks):
                continue
            x = int(landmarks[idx, 0] * width)
            y = int(landmarks[idx, 1] * height)
            cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

    def _draw_head_axis(self, frame: np.ndarray, features: FrameFeatures) -> None:
        """绘制从鼻尖出发的 3D 坐标轴，辅助观察头姿估计效果。"""
        pose = features.head_pose
        landmarks = features.extra.get("landmarks")
        image_size = features.extra.get("image_size")
        if (
            pose is None
            or pose.rotation_vector is None
            or pose.translation_vector is None
            or pose.camera_matrix is None
            or landmarks is None
            or image_size is None
        ):
            return

        width, height = image_size
        nose = lm.pixel_points(landmarks, [lm.NOSE_TIP], width, height)[0].astype(int)
        axis = np.float64([[60, 0, 0], [0, 60, 0], [0, 0, 60]])
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        projected, _ = cv2.projectPoints(
            axis,
            pose.rotation_vector,
            pose.translation_vector,
            pose.camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2).astype(int)
        origin = tuple(nose)
        cv2.line(frame, origin, tuple(projected[0]), (0, 0, 255), 2)
        cv2.line(frame, origin, tuple(projected[1]), (0, 255, 0), 2)
        cv2.line(frame, origin, tuple(projected[2]), (255, 0, 0), 2)

    def _draw_overlay(self, frame: np.ndarray, features: FrameFeatures, result: DMSResult) -> None:
        # cv2.putText 默认字体不支持中文，所以视频上的标签保持英文。
        color_attention = (40, 210, 40) if result.driving_state == "normal" else (0, 165, 255)
        color_fatigue = (40, 210, 40) if result.fatigue_state == "normal" else (0, 0, 255)
        face_count = features.extra.get("face_count")
        track_id = features.extra.get("track_id")
        detector_backend = features.extra.get("detector_backend")
        landmark_source = features.extra.get("landmark_source")
        lines = [
            (f"Driving: {result.driving_state}", color_attention),
            (f"Fatigue: {result.fatigue_state}", color_fatigue),
            (f"Attention score: {result.attention_score:.2f}", (255, 255, 255)),
            (f"Fatigue score: {result.fatigue_score:.2f}", (255, 255, 255)),
            (
                "Yaw/Pitch/Roll: "
                f"{result.features.get('yaw')}/{result.features.get('pitch')}/{result.features.get('roll')}",
                (255, 255, 255),
            ),
            (
                "EAR/PERCLOS: "
                f"{result.features.get('ear')}/{result.features.get('perclos_30s')}",
                (255, 255, 255),
            ),
        ]
        if face_count is not None:
            lines.append((f"Faces/Track: {face_count}/{track_id}", (255, 255, 255)))
        if detector_backend:
            lines.append((f"Detector: {detector_backend}", (255, 255, 255)))
        if landmark_source:
            lines.append((f"Landmarks: {landmark_source}", (255, 255, 255)))
        if not result.calibration_ready:
            lines.append(("Calibrating baseline...", (0, 255, 255)))

        x, y = 16, 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.scale
        thickness = 2
        line_height = int(28 * self.scale) + 8

        max_width = 0
        for text, _ in lines:
            size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, size[0])

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (8, 8),
            (max_width + 32, 18 + line_height * len(lines)),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (x, y + i * line_height), font, font_scale, color, thickness, cv2.LINE_AA)
