from __future__ import annotations

import cv2
import numpy as np

from . import landmarks as lm
from .types import FaceObservation, HeadPose


class HeadPoseEstimator:
    """Estimate yaw, pitch and roll from six Face Mesh landmarks."""

    def __init__(self) -> None:
        # Generic 3D face model in millimeters. The coordinate scale is arbitrary
        # for solvePnP; relative geometry is what matters.
        self.model_points = np.array(
            [
                [0.0, 0.0, 0.0],          # nose tip
                [0.0, -63.6, -12.5],      # chin
                [-43.3, 32.7, -26.0],     # left eye outer corner
                [43.3, 32.7, -26.0],      # right eye outer corner
                [-28.9, -28.9, -24.1],    # left mouth corner
                [28.9, -28.9, -24.1],     # right mouth corner
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
