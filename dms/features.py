from __future__ import annotations

import numpy as np

from .face_tracker import MediaPipeFaceTracker
from .fatigue import FatigueFeatureExtractor
from .gaze import GazeEstimator
from .head_pose import HeadPoseEstimator
from .types import FrameFeatures


class FeatureExtractor:
    def __init__(self, config: dict) -> None:
        self.face_tracker = MediaPipeFaceTracker(config["runtime"]["face_landmarker_model"])
        self.head_pose = HeadPoseEstimator()
        self.gaze = GazeEstimator()
        self.fatigue = FatigueFeatureExtractor()

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: float) -> FrameFeatures:
        observation = self.face_tracker.detect(frame_bgr, timestamp_ms)
        if not observation.found:
            return FrameFeatures(timestamp_ms=timestamp_ms, face_found=False)

        head_pose = self.head_pose.estimate(observation)
        gaze_x, gaze_y = self.gaze.estimate(observation)
        ear, mar = self.fatigue.extract(observation)

        return FrameFeatures(
            timestamp_ms=timestamp_ms,
            face_found=True,
            yaw=head_pose.yaw,
            pitch=head_pose.pitch,
            roll=head_pose.roll,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            ear=ear,
            mar=mar,
            head_pose=head_pose,
            extra={"landmarks": observation.landmarks, "image_size": observation.image_size},
        )

    def close(self) -> None:
        self.face_tracker.close()

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
