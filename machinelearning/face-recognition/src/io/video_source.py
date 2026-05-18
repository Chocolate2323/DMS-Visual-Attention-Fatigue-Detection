from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 依赖缺失时仅做兜底
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


@dataclass(slots=True, frozen=True)
class VideoSourceInfo:
    source: str | int
    is_camera: bool
    width: int
    height: int
    fps: float


class VideoSource:
    def __init__(
        self,
        source: str | int = 0,
        width: int | None = None,
        height: int | None = None,
        target_fps: float | None = None,
    ) -> None:
        self.source = self._coerce_source(source)
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._capture = None
        self._opened_at: float | None = None
        self._frame_index = 0

    @staticmethod
    def _coerce_source(source: str | int) -> str | int:
        if isinstance(source, int):
            return source

        source_text = str(source).strip()
        if source_text.isdigit():
            return int(source_text)
        return source_text

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def open(self) -> "VideoSource":
        if cv2 is None:
            raise RuntimeError("OpenCV 未安装，无法打开视频源。") from _CV2_IMPORT_ERROR

        if self._capture is not None:
            return self

        if isinstance(self.source, str):
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"视频文件不存在: {path}")

        self._capture = cv2.VideoCapture(self.source)
        if not self._capture.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        if self.width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._opened_at = time.perf_counter()
        self._frame_index = 0
        return self

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            self.open()

        success, frame = self._capture.read()
        if not success:
            return None

        self._frame_index += 1
        return frame

    def iter_frames(self):
        while True:
            frame = self.read()
            if frame is None:
                break
            yield self._frame_index, frame

    def get_timestamp_ms(self) -> float:
        if self._capture is None or cv2 is None:
            return 0.0

        timestamp_ms = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if timestamp_ms > 0:
            return timestamp_ms

        if self._opened_at is None:
            return 0.0
        return (time.perf_counter() - self._opened_at) * 1000.0

    @property
    def info(self) -> VideoSourceInfo:
        if self._capture is None or cv2 is None:
            return VideoSourceInfo(
                source=self.source,
                is_camera=self.is_camera,
                width=0,
                height=0,
                fps=0.0,
            )

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return VideoSourceInfo(
            source=self.source,
            is_camera=self.is_camera,
            width=width,
            height=height,
            fps=fps,
        )

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
