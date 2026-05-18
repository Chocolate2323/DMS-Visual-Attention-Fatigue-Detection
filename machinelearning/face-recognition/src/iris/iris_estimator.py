from __future__ import annotations

from collections.abc import Sequence


class IrisEstimator:
    LEFT_IRIS_INDICES = (468, 469, 470, 471, 472)
    RIGHT_IRIS_INDICES = (473, 474, 475, 476, 477)

    def estimate(
        self,
        landmarks: Sequence[tuple[int, int]],
    ) -> dict[str, tuple[int, int] | None]:
        return {
            "left": self._mean_point(landmarks, self.LEFT_IRIS_INDICES),
            "right": self._mean_point(landmarks, self.RIGHT_IRIS_INDICES),
        }

    @staticmethod
    def _mean_point(
        landmarks: Sequence[tuple[int, int]],
        indices: Sequence[int],
    ) -> tuple[int, int] | None:
        points = [landmarks[index] for index in indices if index < len(landmarks)]
        if not points:
            return None
        avg_x = int(round(sum(point[0] for point in points) / len(points)))
        avg_y = int(round(sum(point[1] for point in points) / len(points)))
        return avg_x, avg_y
