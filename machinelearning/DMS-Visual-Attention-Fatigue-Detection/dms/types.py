from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceObservation:
    """单帧人脸检测结果。landmarks 使用 MediaPipe 的归一化坐标。"""

    found: bool
    landmarks: np.ndarray | None = None
    image_size: tuple[int, int] = (0, 0)
    detection_confidence: float | None = None

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]


@dataclass
class HeadPose:
    """头部姿态结果，角度单位为度。"""

    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    rotation_vector: np.ndarray | None = None
    translation_vector: np.ndarray | None = None
    camera_matrix: np.ndarray | None = None


@dataclass
class FrameFeatures:
    """一帧中提取出的全部中间特征。"""

    timestamp_ms: float
    face_found: bool
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    gaze_x: float | None = None
    gaze_y: float | None = None
    ear: float | None = None
    mar: float | None = None
    head_pose: HeadPose | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DMSResult:
    """最终对外输出的一帧 DMS 状态。"""

    timestamp_ms: float
    driving_state: str
    fatigue_state: str
    attention_score: float
    fatigue_score: float
    features: dict[str, Any]
    calibration_ready: bool

    def to_dict(self) -> dict[str, Any]:
        """转为可直接写入 JSON 的普通字典。"""
        return {
            "timestamp_ms": round(self.timestamp_ms, 2),
            "driving_state": self.driving_state,
            "fatigue_state": self.fatigue_state,
            "attention_score": round(float(self.attention_score), 4),
            "fatigue_score": round(float(self.fatigue_score), 4),
            "features": self.features,
            "calibration_ready": self.calibration_ready,
        }
