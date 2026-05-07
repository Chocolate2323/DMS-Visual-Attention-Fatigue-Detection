from __future__ import annotations

import numpy as np


# MediaPipe Face Mesh indices used by this project.
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_MOUTH = 291
RIGHT_MOUTH = 61

LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def pixel_points(landmarks: np.ndarray, indices: list[int], width: int, height: int) -> np.ndarray:
    pts = landmarks[indices, :2].copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts.astype(np.float64)


def mean_pixel_point(landmarks: np.ndarray, indices: list[int], width: int, height: int) -> np.ndarray:
    return pixel_points(landmarks, indices, width, height).mean(axis=0)
