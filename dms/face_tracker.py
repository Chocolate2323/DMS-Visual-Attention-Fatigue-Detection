from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

from .types import FaceObservation


class MediaPipeFaceTracker:
    """MediaPipe Tasks FaceLandmarker 的轻量封装。"""

    def __init__(
        self,
        model_path: str | Path,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._timestamp_ms = -1
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"未找到 MediaPipe 人脸关键点模型：{model_path}。"
                "请下载 face_landmarker.task，并在配置中的 runtime.face_landmarker_model 指定路径。"
            )

        # VIDEO 模式会利用前后帧跟踪信息，比逐帧 IMAGE 模式更稳定。
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: float) -> FaceObservation:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # MediaPipe 要求视频时间戳严格递增，部分视频首帧时间戳会重复或为 0。
        timestamp = max(int(timestamp_ms), self._timestamp_ms + 1)
        self._timestamp_ms = timestamp
        result = self._landmarker.detect_for_video(mp_image, timestamp)

        if not result.face_landmarks:
            return FaceObservation(found=False, image_size=(width, height))

        face_landmarks = result.face_landmarks[0]
        # 保存为 numpy 数组，便于后续姿态、视线、疲劳特征统一计算。
        landmarks = np.array(
            [[point.x, point.y, point.z] for point in face_landmarks],
            dtype=np.float64,
        )
        return FaceObservation(found=True, landmarks=landmarks, image_size=(width, height))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeFaceTracker":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
