from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from src.detection.face_detector import Detection


def _point_to_list(point: tuple[int, int] | None) -> list[int] | None:
    if point is None:
        return None
    return [int(point[0]), int(point[1])]


class FrameSerializer:
    def pack(
        self,
        frame_id: int,
        timestamp_ms: float,
        image_size: tuple[int, int],
        faces: list[Detection],
        driver_face: Detection | None,
        track_id: int | None,
        landmarks: list[tuple[int, int]],
        iris: dict[str, tuple[int, int] | None],
    ) -> dict[str, Any]:
        image_width, image_height = image_size
        face_payload = [
            {
                "bbox": face.to_list(),
                "score": round(float(face.score), 4),
            }
            for face in faces
        ]

        return {
            "frame_id": int(frame_id),
            "timestamp_ms": round(float(timestamp_ms), 2),
            "image_size": [int(image_width), int(image_height)],
            "image_width": int(image_width),
            "image_height": int(image_height),
            "face_count": len(faces),
            "face_bbox": [item["bbox"] for item in face_payload],
            "faces": face_payload,
            "driver_detected": driver_face is not None,
            "driver_bbox": driver_face.to_list() if driver_face else None,
            "driver_score": round(float(driver_face.score), 4) if driver_face else None,
            "track_id": int(track_id) if track_id is not None else None,
            "landmarks": [[int(x), int(y)] for x, y in landmarks],
            "iris_left": _point_to_list(iris.get("left")),
            "iris_right": _point_to_list(iris.get("right")),
            "confidence": round(float(driver_face.score), 4) if driver_face else None,
        }


class JsonlResultWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self._handle: TextIO | None = None

    def open(self) -> None:
        if self._handle is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        if self._handle is None:
            self.open()
        assert self._handle is not None
        json.dump(payload, self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "JsonlResultWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
