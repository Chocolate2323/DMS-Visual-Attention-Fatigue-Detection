from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 依赖缺失时仅做兜底
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


@dataclass(slots=True)
class PreprocessResult:
    original_bgr: np.ndarray
    original_rgb: np.ndarray
    processed_bgr: np.ndarray
    processed_rgb: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale_x: float
    scale_y: float
    was_grayscale: bool

    def map_bbox_to_original(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        width, height = self.original_size
        mapped = (
            max(0, min(width - 1, int(round(x1 * self.scale_x)))),
            max(0, min(height - 1, int(round(y1 * self.scale_y)))),
            max(0, min(width - 1, int(round(x2 * self.scale_x)))),
            max(0, min(height - 1, int(round(y2 * self.scale_y)))),
        )
        return mapped

    def map_points_to_original(
        self,
        points: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        width, height = self.original_size
        mapped_points: list[tuple[int, int]] = []
        for x, y in points:
            mapped_points.append(
                (
                    max(0, min(width - 1, int(round(x * self.scale_x)))),
                    max(0, min(height - 1, int(round(y * self.scale_y)))),
                )
            )
        return mapped_points


class ImagePreprocessor:
    def __init__(self, target_width: int = 640, target_height: int = 480) -> None:
        self.target_width = target_width
        self.target_height = target_height

    @property
    def target_size(self) -> tuple[int, int]:
        return self.target_width, self.target_height

    def process(self, frame: np.ndarray) -> PreprocessResult:
        if cv2 is None:
            raise RuntimeError("OpenCV 未安装，无法执行图像预处理。") from _CV2_IMPORT_ERROR
        if frame is None:
            raise ValueError("输入帧不能为空。")

        was_grayscale = frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1)
        if was_grayscale:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame.copy()

        original_height, original_width = bgr.shape[:2]
        original_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        processed_bgr = cv2.resize(
            bgr,
            self.target_size,
            interpolation=cv2.INTER_LINEAR,
        )
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        processed_width, processed_height = self.target_size
        scale_x = original_width / processed_width
        scale_y = original_height / processed_height

        return PreprocessResult(
            original_bgr=bgr,
            original_rgb=original_rgb,
            processed_bgr=processed_bgr,
            processed_rgb=processed_rgb,
            original_size=(original_width, original_height),
            processed_size=(processed_width, processed_height),
            scale_x=scale_x,
            scale_y=scale_y,
            was_grayscale=was_grayscale,
        )
