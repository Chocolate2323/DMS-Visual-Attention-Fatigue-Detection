from __future__ import annotations

import numpy as np


# 本项目使用的 MediaPipe Face Mesh 关键点索引。
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_MOUTH = 291
RIGHT_MOUTH = 61

# EAR 眼睛纵横比使用的左右眼 6 点。
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]

# MediaPipe FaceLandmarker 输出的虹膜点。
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]

# MAR 嘴部纵横比使用的嘴角和上下唇点。
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def pixel_points(landmarks: np.ndarray, indices: list[int], width: int, height: int) -> np.ndarray:
    """把归一化关键点转换为像素坐标。"""
    pts = landmarks[indices, :2].copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts.astype(np.float64)


def mean_pixel_point(landmarks: np.ndarray, indices: list[int], width: int, height: int) -> np.ndarray:
    """返回多个关键点的像素坐标中心。"""
    return pixel_points(landmarks, indices, width, height).mean(axis=0)
