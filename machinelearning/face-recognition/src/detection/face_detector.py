from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocess.image_preprocessor import PreprocessResult

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - 可选依赖
    mp = None

MEDIAPIPE_SOLUTIONS_AVAILABLE = mp is not None and hasattr(mp, "solutions")
MEDIAPIPE_COMPAT_HINT = "当前 mediapipe 安装不包含 solutions API，请安装 mediapipe==0.10.14。"

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - 可选依赖
    YOLO = None


@dataclass(slots=True, frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_list(self) -> list[int]:
        return [int(value) for value in self.bbox]


class _BaseDetectorBackend:
    name = "base"

    def detect(self, preprocessed: PreprocessResult) -> list[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _NoOpDetectorBackend(_BaseDetectorBackend):
    name = "none"

    def detect(self, preprocessed: PreprocessResult) -> list[Detection]:
        return []


class _MediaPipeDetectorBackend(_BaseDetectorBackend):
    name = "mediapipe"

    def __init__(self, confidence_threshold: float) -> None:
        if not MEDIAPIPE_SOLUTIONS_AVAILABLE:
            raise RuntimeError(MEDIAPIPE_COMPAT_HINT)
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence_threshold,
        )

    def detect(self, preprocessed: PreprocessResult) -> list[Detection]:
        results = self._detector.process(preprocessed.processed_rgb)
        if not results.detections:
            return []

        image_height, image_width = preprocessed.processed_rgb.shape[:2]
        detections: list[Detection] = []
        for item in results.detections:
            bbox = item.location_data.relative_bounding_box
            x1 = int(round(bbox.xmin * image_width))
            y1 = int(round(bbox.ymin * image_height))
            x2 = int(round((bbox.xmin + bbox.width) * image_width))
            y2 = int(round((bbox.ymin + bbox.height) * image_height))
            mapped_bbox = preprocessed.map_bbox_to_original((x1, y1, x2, y2))
            detections.append(
                Detection(
                    bbox=_sanitize_bbox(mapped_bbox, preprocessed.original_size),
                    score=float(item.score[0]) if item.score else 0.0,
                )
            )
        return detections

    def close(self) -> None:
        self._detector.close()


class _YoloDetectorBackend(_BaseDetectorBackend):
    name = "yolo"

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        input_size: int,
    ) -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO 不可用。")
        self._model = YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold
        self._input_size = input_size

    def detect(self, preprocessed: PreprocessResult) -> list[Detection]:
        results = self._model.predict(
            source=preprocessed.processed_bgr,
            conf=self._confidence_threshold,
            imgsz=self._input_size,
            verbose=False,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        detections: list[Detection] = []
        for box in boxes:
            coords = box.xyxy[0].tolist()
            mapped_bbox = preprocessed.map_bbox_to_original(
                (
                    int(round(coords[0])),
                    int(round(coords[1])),
                    int(round(coords[2])),
                    int(round(coords[3])),
                )
            )
            score = float(box.conf[0]) if getattr(box, "conf", None) is not None else 0.0
            detections.append(
                Detection(
                    bbox=_sanitize_bbox(mapped_bbox, preprocessed.original_size),
                    score=score,
                )
            )
        return detections


def _sanitize_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return x1, y1, x2, y2


class FaceDetector:
    def __init__(
        self,
        backend: str = "auto",
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.35,
        input_size: int = 640,
    ) -> None:
        self.backend_name = "none"
        self.status_message = "未启用检测器。"
        self.last_error: str | None = None
        self._backend = self._build_backend(
            backend=backend,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            input_size=input_size,
        )

    def _build_backend(
        self,
        backend: str,
        model_path: str | Path | None,
        confidence_threshold: float,
        input_size: int,
    ) -> _BaseDetectorBackend:
        requested = backend.lower()
        resolved_model_path = Path(model_path) if model_path else None

        if requested in {"auto", "yolo"} and resolved_model_path is not None and resolved_model_path.exists():
            try:
                detector = _YoloDetectorBackend(
                    model_path=resolved_model_path,
                    confidence_threshold=confidence_threshold,
                    input_size=input_size,
                )
                self.backend_name = detector.name
                self.status_message = f"已加载 YOLO 人脸检测权重: {resolved_model_path.name}"
                return detector
            except Exception as exc:  # pragma: no cover - 依赖环境相关
                self.last_error = str(exc)
                self.status_message = f"YOLO 初始化失败，准备回退: {exc}"

        if requested in {"auto", "yolo", "mediapipe", "mp"}:
            if mp is not None and not MEDIAPIPE_SOLUTIONS_AVAILABLE:
                self.last_error = MEDIAPIPE_COMPAT_HINT
                self.status_message = MEDIAPIPE_COMPAT_HINT
            elif mp is not None:
                try:
                    detector = _MediaPipeDetectorBackend(confidence_threshold=confidence_threshold)
                    self.backend_name = detector.name
                    if resolved_model_path is not None and not resolved_model_path.exists():
                        self.status_message = (
                            f"未找到 YOLO 权重 {resolved_model_path}，已回退到 MediaPipe 人脸检测。"
                        )
                    else:
                        self.status_message = "已启用 MediaPipe 人脸检测。"
                    return detector
                except Exception as exc:  # pragma: no cover - 依赖环境相关
                    self.last_error = str(exc)
                    self.status_message = f"MediaPipe 初始化失败: {exc}"

        self.backend_name = "none"
        if self.status_message == "未启用检测器。":
            self.status_message = "未找到可用的人脸检测后端，检测结果将为空。"
        return _NoOpDetectorBackend()

    def detect(self, preprocessed: PreprocessResult) -> list[Detection]:
        try:
            detections = self._backend.detect(preprocessed)
        except Exception as exc:  # pragma: no cover - 运行环境相关
            self.last_error = str(exc)
            return []

        return sorted(
            detections,
            key=lambda item: (item.score, item.area),
            reverse=True,
        )

    def close(self) -> None:
        self._backend.close()
