from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lm
from .types import FaceObservation, HeadPose


class HeadPoseEstimator:
    """使用 2D 人脸关键点和通用 3D 人脸模型估计头部姿态。"""

    def __init__(self) -> None:
        # 通用 3D 人脸模型，单位近似为毫米。solvePnP 主要依赖相对几何关系，
        # 绝对尺度不影响最终欧拉角，但点位顺序必须和 2D 关键点严格对应。
        self.model_points = np.array(
            [
                [0.0, 0.0, 0.0],          # 鼻尖
                [0.0, -63.6, -12.5],      # 下巴
                [-43.3, 32.7, -26.0],     # 左眼外眼角
                [43.3, 32.7, -26.0],      # 右眼外眼角
                [-28.9, -28.9, -24.1],    # 左嘴角
                [28.9, -28.9, -24.1],     # 右嘴角
            ],
            dtype=np.float64,
        )
        self.indices = [
            lm.NOSE_TIP,
            lm.CHIN,
            lm.LEFT_EYE_OUTER,
            lm.RIGHT_EYE_OUTER,
            lm.LEFT_MOUTH,
            lm.RIGHT_MOUTH,
        ]

    def estimate(self, observation: FaceObservation) -> HeadPose:
        if not observation.found or observation.landmarks is None:
            return HeadPose()

        width, height = observation.image_size
        image_points = lm.pixel_points(observation.landmarks, self.indices, width, height)

        # 普通单目视频通常没有相机内参。这里用图像宽度近似焦距，
        # 适合课程项目原型；若有真实相机标定参数，应替换此矩阵。
        focal_length = float(width)
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return HeadPose(camera_matrix=camera_matrix)

        # OpenCV 返回旋转向量，转换成欧拉角后分别作为 pitch/yaw/roll。
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        pitch, yaw, roll = (float(angles[0]), float(angles[1]), float(angles[2]))

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
            camera_matrix=camera_matrix,
        )
