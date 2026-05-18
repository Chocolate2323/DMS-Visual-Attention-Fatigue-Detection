from __future__ import annotations

from dataclasses import dataclass

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 依赖缺失时仅做兜底
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - 可选依赖
    mp = None

MEDIAPIPE_SOLUTIONS_AVAILABLE = mp is not None and hasattr(mp, "solutions")
MEDIAPIPE_COMPAT_HINT = "当前 mediapipe 安装不包含 solutions API，请安装 mediapipe==0.10.14。"


@dataclass(slots=True)
class FaceMeshResult:
    landmarks: list[tuple[int, int]]
    roi_bbox: tuple[int, int, int, int] | None = None


class FaceMeshEstimator:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        bbox_margin: float = 0.15,
    ) -> None:
        self.bbox_margin = bbox_margin
        self.status_message = "FaceMesh 未启用。"
        self._face_mesh = None
        self.is_available = False

        if cv2 is None:
            self.status_message = "OpenCV 未安装，无法启用 FaceMesh。"
            return
        if mp is None:
            self.status_message = "MediaPipe 未安装，无法启用 FaceMesh。"
            return
        if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
            self.status_message = MEDIAPIPE_COMPAT_HINT
            return

        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.is_available = True
            self.status_message = "FaceMesh 已启用。"
        except Exception as exc:  # pragma: no cover - 依赖环境相关
            self.status_message = f"FaceMesh 初始化失败: {exc}"

    def estimate(
        self,
        frame_bgr,
        face_bbox: tuple[int, int, int, int] | None,
    ) -> FaceMeshResult:
        if face_bbox is None:
            return FaceMeshResult(landmarks=[])
        if cv2 is None:
            raise RuntimeError("OpenCV 未安装，无法执行关键点检测。") from _CV2_IMPORT_ERROR
        if self._face_mesh is None:
            return FaceMeshResult(landmarks=[], roi_bbox=face_bbox)

        image_height, image_width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = face_bbox
        margin_x = int((x2 - x1) * self.bbox_margin)
        margin_y = int((y2 - y1) * self.bbox_margin)
        roi_x1 = max(0, x1 - margin_x)
        roi_y1 = max(0, y1 - margin_y)
        roi_x2 = min(image_width, x2 + margin_x)
        roi_y2 = min(image_height, y2 + margin_y)

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return FaceMeshResult(landmarks=[], roi_bbox=face_bbox)

        face_roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        if face_roi.size == 0:
            return FaceMeshResult(landmarks=[], roi_bbox=face_bbox)

        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(face_roi_rgb)
        if not results.multi_face_landmarks:
            return FaceMeshResult(landmarks=[], roi_bbox=(roi_x1, roi_y1, roi_x2, roi_y2))

        roi_height, roi_width = face_roi.shape[:2]
        landmarks: list[tuple[int, int]] = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(round(landmark.x * roi_width + roi_x1))
            y = int(round(landmark.y * roi_height + roi_y1))
            landmarks.append(
                (
                    max(0, min(image_width - 1, x)),
                    max(0, min(image_height - 1, y)),
                )
            )

        return FaceMeshResult(
            landmarks=landmarks,
            roi_bbox=(roi_x1, roi_y1, roi_x2, roi_y2),
        )

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
