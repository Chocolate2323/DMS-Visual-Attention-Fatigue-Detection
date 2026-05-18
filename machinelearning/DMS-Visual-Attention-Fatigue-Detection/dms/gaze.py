from __future__ import annotations

import numpy as np

from . import landmarks as lm
from .math_utils import safe_mean
from .types import FaceObservation


class GazeEstimator:
    """根据虹膜在眼眶内的相对位置估计粗略视线方向。"""

    def estimate(self, observation: FaceObservation) -> tuple[float | None, float | None]:
        if not observation.found or observation.landmarks is None:
            return None, None

        width, height = observation.image_size
        landmarks = observation.landmarks
        # 单眼视线容易受遮挡和眨眼影响，因此左右眼分别估计后再取均值。
        gaze_values = []

        left = self._single_eye_gaze(
            landmarks,
            width,
            height,
            eye_corners=[lm.LEFT_EYE_INNER, lm.LEFT_EYE_OUTER],
            upper_indices=[386, 387],
            lower_indices=[374, 373],
            iris_indices=lm.LEFT_IRIS,
        )
        right = self._single_eye_gaze(
            landmarks,
            width,
            height,
            eye_corners=[lm.RIGHT_EYE_OUTER, lm.RIGHT_EYE_INNER],
            upper_indices=[159, 158],
            lower_indices=[145, 153],
            iris_indices=lm.RIGHT_IRIS,
        )

        for gaze in (left, right):
            if gaze[0] is not None and gaze[1] is not None:
                gaze_values.append(gaze)

        if not gaze_values:
            return None, None

        gaze_x = safe_mean([v[0] for v in gaze_values])
        gaze_y = safe_mean([v[1] for v in gaze_values])
        return gaze_x, gaze_y

    @staticmethod
    def _single_eye_gaze(
        landmarks: np.ndarray,
        width: int,
        height: int,
        eye_corners: list[int],
        upper_indices: list[int],
        lower_indices: list[int],
        iris_indices: list[int],
    ) -> tuple[float | None, float | None]:
        if max(iris_indices) >= len(landmarks):
            return None, None

        # gaze_x/gaze_y 是归一化偏移量，不是物理角度。
        corners = lm.pixel_points(landmarks, eye_corners, width, height)
        iris = lm.mean_pixel_point(landmarks, iris_indices, width, height)
        upper = lm.mean_pixel_point(landmarks, upper_indices, width, height)
        lower = lm.mean_pixel_point(landmarks, lower_indices, width, height)

        min_x = float(np.min(corners[:, 0]))
        max_x = float(np.max(corners[:, 0]))
        eye_width = max(max_x - min_x, 1.0)
        mid_x = (min_x + max_x) / 2.0

        eye_height = max(abs(float(lower[1] - upper[1])), 1.0)
        mid_y = (float(lower[1]) + float(upper[1])) / 2.0

        gaze_x = (float(iris[0]) - mid_x) / eye_width * 2.0
        gaze_y = (float(iris[1]) - mid_y) / eye_height
        return gaze_x, gaze_y
