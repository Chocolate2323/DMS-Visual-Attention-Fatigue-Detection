from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.detection.face_detector import Detection, FaceDetector
from src.iris.iris_estimator import IrisEstimator
from src.landmark.face_mesh_estimator import FaceMeshEstimator
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.tracking.driver_selector import DriverSelector
from src.utils.serializer import FrameSerializer


@dataclass(slots=True)
class FrameProcessorOutput:
    result: dict[str, Any]
    faces: list[Detection]
    driver_face: Detection | None
    track_id: int | None
    landmarks: list[tuple[int, int]]
    iris: dict[str, tuple[int, int] | None]


class FrameProcessor:
    def __init__(
        self,
        preprocessor: ImagePreprocessor,
        face_detector: FaceDetector,
        driver_selector: DriverSelector,
        face_mesh_estimator: FaceMeshEstimator,
        iris_estimator: IrisEstimator,
        serializer: FrameSerializer,
    ) -> None:
        self.preprocessor = preprocessor
        self.face_detector = face_detector
        self.driver_selector = driver_selector
        self.face_mesh_estimator = face_mesh_estimator
        self.iris_estimator = iris_estimator
        self.serializer = serializer

    @property
    def detector_backend(self) -> str:
        return self.face_detector.backend_name

    def process(
        self,
        frame,
        frame_id: int,
        timestamp_ms: float,
    ) -> FrameProcessorOutput:
        preprocessed = self.preprocessor.process(frame)
        faces = self.face_detector.detect(preprocessed)
        driver_face, track_id = self.driver_selector.select(
            faces,
            preprocessed.original_size,
        )

        landmarks: list[tuple[int, int]] = []
        iris = {"left": None, "right": None}
        if driver_face is not None:
            mesh_result = self.face_mesh_estimator.estimate(
                preprocessed.original_bgr,
                driver_face.bbox,
            )
            landmarks = mesh_result.landmarks
            if landmarks:
                iris = self.iris_estimator.estimate(landmarks)

        result = self.serializer.pack(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            image_size=preprocessed.original_size,
            faces=faces,
            driver_face=driver_face,
            track_id=track_id,
            landmarks=landmarks,
            iris=iris,
        )
        return FrameProcessorOutput(
            result=result,
            faces=faces,
            driver_face=driver_face,
            track_id=track_id,
            landmarks=landmarks,
            iris=iris,
        )

    def close(self) -> None:
        self.face_detector.close()
        self.face_mesh_estimator.close()
