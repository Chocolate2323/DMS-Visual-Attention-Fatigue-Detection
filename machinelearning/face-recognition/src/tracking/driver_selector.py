from __future__ import annotations

from collections.abc import Sequence

from src.detection.face_detector import Detection


def _bbox_iou(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class DriverSelector:
    def __init__(
        self,
        driver_side: str = "left",
        iou_threshold: float = 0.35,
        max_lost_frames: int = 8,
        position_weight: float = 0.6,
        area_weight: float = 0.4,
    ) -> None:
        self.driver_side = driver_side
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.position_weight = position_weight
        self.area_weight = area_weight
        self.last_bbox: tuple[int, int, int, int] | None = None
        self.lost_count = 0
        self.track_id = 0

    def reset(self) -> None:
        self.last_bbox = None
        self.lost_count = 0
        self.track_id = 0

    def select(
        self,
        faces: Sequence[Detection],
        image_size: tuple[int, int],
    ) -> tuple[Detection | None, int | None]:
        if not faces:
            self._handle_missing_faces()
            return None, None

        if self.last_bbox is not None:
            matched_face, best_iou = self._match_previous_driver(faces)
            if matched_face is not None and best_iou >= self.iou_threshold:
                self.last_bbox = matched_face.bbox
                self.lost_count = 0
                if self.track_id == 0:
                    self.track_id = 1
                return matched_face, self.track_id

            self.lost_count += 1
            if self.lost_count <= self.max_lost_frames:
                return None, None

        selected = self._select_new_driver(faces, image_size)
        if selected is None:
            return None, None

        self.track_id += 1
        self.last_bbox = selected.bbox
        self.lost_count = 0
        return selected, self.track_id

    def _handle_missing_faces(self) -> None:
        if self.last_bbox is None:
            return
        self.lost_count += 1
        if self.lost_count > self.max_lost_frames:
            self.last_bbox = None

    def _match_previous_driver(
        self,
        faces: Sequence[Detection],
    ) -> tuple[Detection | None, float]:
        if self.last_bbox is None:
            return None, 0.0

        best_face: Detection | None = None
        best_iou = 0.0
        for face in faces:
            iou = _bbox_iou(self.last_bbox, face.bbox)
            if iou > best_iou:
                best_face = face
                best_iou = iou
        return best_face, best_iou

    def _select_new_driver(
        self,
        faces: Sequence[Detection],
        image_size: tuple[int, int],
    ) -> Detection | None:
        width, _ = image_size
        if width <= 0:
            return None

        preferred_x = {
            "left": width * 0.35,
            "center": width * 0.50,
            "right": width * 0.65,
        }.get(self.driver_side, width * 0.35)

        max_area = max((face.area for face in faces), default=1)
        best_face: Detection | None = None
        best_score = float("-inf")

        for face in faces:
            center_x, _ = face.center
            position_score = 1.0 - min(abs(center_x - preferred_x) / width, 1.0)
            area_score = face.area / max_area if max_area > 0 else 0.0
            combined_score = (
                self.position_weight * position_score
                + self.area_weight * area_score
            )
            if combined_score > best_score:
                best_score = combined_score
                best_face = face
        return best_face
