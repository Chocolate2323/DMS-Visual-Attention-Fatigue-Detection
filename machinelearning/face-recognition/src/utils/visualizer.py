from __future__ import annotations

from collections.abc import Sequence

from src.detection.face_detector import Detection

try:
    import cv2
except ImportError:  # pragma: no cover - 依赖缺失时仅做兜底
    cv2 = None


def draw_result(
    frame_bgr,
    faces: Sequence[Detection],
    driver_face: Detection | None = None,
    landmarks: Sequence[tuple[int, int]] | None = None,
    iris: dict[str, tuple[int, int] | None] | None = None,
    *,
    frame_id: int | None = None,
    track_id: int | None = None,
    detector_backend: str | None = None,
    draw_all_faces: bool = True,
    draw_landmarks: bool = True,
    draw_iris: bool = True,
    landmark_stride: int = 1,
):
    canvas = frame_bgr.copy()
    if cv2 is None:
        return canvas

    if draw_all_faces:
        for face in faces:
            _draw_bbox(canvas, face, color=(60, 220, 60), thickness=2)

    if driver_face is not None:
        _draw_bbox(canvas, driver_face, color=(0, 165, 255), thickness=3)

    if draw_landmarks and landmarks:
        stride = max(1, landmark_stride)
        for x, y in landmarks[::stride]:
            cv2.circle(canvas, (int(x), int(y)), 1, (255, 255, 0), -1)

    if draw_iris and iris:
        left_iris = iris.get("left")
        right_iris = iris.get("right")
        if left_iris is not None:
            cv2.circle(canvas, left_iris, 3, (255, 0, 255), -1)
        if right_iris is not None:
            cv2.circle(canvas, right_iris, 3, (0, 0, 255), -1)

    overlay_lines = [f"faces={len(faces)}"]
    if frame_id is not None:
        overlay_lines.insert(0, f"frame={frame_id}")
    if driver_face is not None:
        overlay_lines.append(f"driver_score={driver_face.score:.2f}")
    if track_id is not None:
        overlay_lines.append(f"track_id={track_id}")
    if detector_backend:
        overlay_lines.append(f"detector={detector_backend}")

    text_y = 28
    for line in overlay_lines:
        cv2.putText(
            canvas,
            line,
            (12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        text_y += 24

    return canvas


def _draw_bbox(frame_bgr, detection: Detection, color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = detection.bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
