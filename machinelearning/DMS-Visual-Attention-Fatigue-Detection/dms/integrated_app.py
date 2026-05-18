from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import load_config
from .face_tracker import MediaPipeFaceTracker
from .output import JsonArrayWriter
from .state import DMSState
from .types import FaceObservation, FrameFeatures
from .visualizer import Visualizer
from .fatigue import FatigueFeatureExtractor
from .gaze import GazeEstimator
from .head_pose import HeadPoseEstimator


class IntegratedFeatureExtractor:
    def __init__(self, face_landmarker_model: str | Path) -> None:
        self.head_pose = HeadPoseEstimator()
        self.gaze = GazeEstimator()
        self.fatigue = FatigueFeatureExtractor()
        self.landmark_status = "使用 face-recognition FaceMesh 关键点。"
        self.fallback_tracker = None
        try:
            self.fallback_tracker = MediaPipeFaceTracker(face_landmarker_model)
            self.landmark_status = "FaceMesh 不可用时将回退到 DMS Face Landmarker。"
        except Exception as exc:
            self.landmark_status = f"DMS Face Landmarker 不可用: {exc}"

    def extract(
        self,
        *,
        frame_bgr: np.ndarray,
        timestamp_ms: float,
        image_size: tuple[int, int],
        landmarks: list[tuple[int, int]],
        driver_bbox: tuple[int, int, int, int] | None,
        driver_score: float | None,
        track_id: int | None,
        face_count: int,
        detector_backend: str,
    ) -> FrameFeatures:
        extra: dict[str, Any] = {
            "image_size": image_size,
            "driver_bbox": driver_bbox,
            "driver_score": driver_score,
            "track_id": track_id,
            "face_count": face_count,
            "detector_backend": detector_backend,
            "landmark_source": None,
            "landmarks": None,
        }

        if driver_bbox is None:
            return FrameFeatures(timestamp_ms=timestamp_ms, face_found=False, extra=extra)

        observation = None
        if landmarks:
            observation = self._to_observation(image_size, landmarks, driver_score)
            extra["landmark_source"] = "face_mesh"
        elif self.fallback_tracker is not None:
            fallback = self.fallback_tracker.detect(frame_bgr, timestamp_ms)
            if fallback.found:
                observation = fallback
                extra["landmark_source"] = "face_landmarker"

        if observation is None:
            return FrameFeatures(timestamp_ms=timestamp_ms, face_found=True, extra=extra)

        head_pose = self.head_pose.estimate(observation)
        gaze_x, gaze_y = self.gaze.estimate(observation)
        ear, mar = self.fatigue.extract(observation)
        extra["landmarks"] = observation.landmarks

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
            extra=extra,
        )

    def close(self) -> None:
        if self.fallback_tracker is not None:
            self.fallback_tracker.close()

    @staticmethod
    def _to_observation(
        image_size: tuple[int, int],
        landmarks: list[tuple[int, int]],
        driver_score: float | None,
    ) -> FaceObservation:
        width, height = image_size
        scale_x = float(max(width, 1))
        scale_y = float(max(height, 1))
        normalized = np.zeros((len(landmarks), 3), dtype=np.float64)
        points = np.asarray(landmarks, dtype=np.float64)
        normalized[:, 0] = points[:, 0] / scale_x
        normalized[:, 1] = points[:, 1] / scale_y
        return FaceObservation(
            found=True,
            landmarks=normalized,
            image_size=image_size,
            detection_confidence=driver_score,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 YOLO/FaceMesh + DMS 状态机的联合后端。",
    )
    parser.add_argument("--input", "-i", required=True, help="输入视频路径或摄像头编号，例如 0。")
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/integrated_annotated.mp4",
        help="标注视频输出路径。传入 none 可关闭视频输出。",
    )
    parser.add_argument(
        "--json",
        default="outputs/integrated_results.json",
        help="JSON 结果输出路径。传入 none 可关闭 JSON 输出。",
    )
    parser.add_argument("--config", default=None, help="DMS YAML 配置路径。")
    parser.add_argument(
        "--face-config",
        default=None,
        help="face-recognition YAML 配置路径。默认使用其内置配置。",
    )
    parser.add_argument(
        "--face-repo",
        default=None,
        help="face-recognition 仓库路径。默认查找当前项目同级目录下的 face-recognition。",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "yolo", "mediapipe"],
        default="auto",
        help="人脸检测后端。",
    )
    parser.add_argument("--weights", default=None, help="YOLO 人脸权重路径。")
    parser.add_argument(
        "--driver-side",
        choices=["left", "center", "right"],
        default="left",
        help="主驾驶员所在侧。",
    )
    parser.add_argument("--resize-width", type=int, default=None, help="检测前缩放宽度。")
    parser.add_argument("--resize-height", type=int, default=None, help="检测前缩放高度。")
    parser.add_argument("--camera-width", type=int, default=None, help="摄像头采集宽度。")
    parser.add_argument("--camera-height", type=int, default=None, help="摄像头采集高度。")
    parser.add_argument("--display", action="store_true", help="在 OpenCV 窗口中显示联合结果。")
    parser.add_argument("--mirror", action="store_true", help="水平镜像画面，适合摄像头预览。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧。")
    parser.add_argument("--frame-stride", type=int, default=1, help="每隔 N 帧处理一帧。")
    return parser.parse_args(argv)


def _resolve_face_repo_root(face_repo: str | None) -> Path:
    if face_repo:
        repo_root = Path(face_repo).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[2] / "face-recognition"

    if not repo_root.exists():
        raise FileNotFoundError(
            f"未找到 face-recognition 仓库：{repo_root}。"
            "请通过 --face-repo 指定仓库路径。"
        )
    return repo_root


def _load_face_modules(face_repo_root: Path) -> tuple[type[Any], Any, Any]:
    if str(face_repo_root) not in sys.path:
        sys.path.insert(0, str(face_repo_root))

    from src.config import load_config as load_face_config
    from src.io.video_source import VideoSource
    from src.main import build_processor

    return VideoSource, load_face_config, build_processor


def _coerce_source(value: str) -> str | int:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def _make_video_writer(path: str | None, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    if not path or path.lower() == "none":
        return None

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def _apply_face_overrides(config: Any, args: argparse.Namespace) -> Any:
    config.input.source = _coerce_source(args.input)
    config.detection.backend = args.backend
    if args.weights is not None:
        config.detection.model_path = args.weights
    config.tracking.driver_side = args.driver_side
    if args.resize_width is not None:
        config.preprocess.resize_width = args.resize_width
    if args.resize_height is not None:
        config.preprocess.resize_height = args.resize_height
    if args.camera_width is not None:
        config.input.width = args.camera_width
    if args.camera_height is not None:
        config.input.height = args.camera_height
    config.visualization.enabled = False
    config.output.save_jsonl = False
    return config


def _build_json_record(result: dict[str, Any], features: FrameFeatures) -> dict[str, Any]:
    extra = features.extra
    driver_bbox = extra.get("driver_bbox")
    return {
        **result,
        "detector": {
            "backend": extra.get("detector_backend"),
            "landmark_source": extra.get("landmark_source"),
            "face_count": int(extra.get("face_count") or 0),
            "driver_detected": bool(features.face_found),
            "driver_bbox": list(driver_bbox) if driver_bbox is not None else None,
            "driver_score": round(float(extra["driver_score"]), 4)
            if extra.get("driver_score") is not None
            else None,
            "track_id": extra.get("track_id"),
        },
    }


def run(args: argparse.Namespace) -> int:
    dms_config = load_config(args.config)
    dms_root = Path(__file__).resolve().parents[1]
    face_repo_root = _resolve_face_repo_root(args.face_repo)
    VideoSource, load_face_config, build_processor = _load_face_modules(face_repo_root)
    face_config = _apply_face_overrides(load_face_config(args.face_config), args)
    processor = build_processor(face_config)

    dms_state = DMSState(dms_config)
    visualizer = Visualizer(dms_config)
    face_landmarker_model = dms_root / dms_config["runtime"]["face_landmarker_model"]
    feature_extractor = IntegratedFeatureExtractor(face_landmarker_model)
    frame_stride = max(1, int(args.frame_stride))
    json_path = None if args.json.lower() == "none" else args.json
    processed_frame_count = 0
    writer = None

    try:
        with VideoSource(
            source=_coerce_source(args.input),
            width=face_config.input.width,
            height=face_config.input.height,
            target_fps=face_config.input.target_fps,
        ) as video_source, JsonArrayWriter(json_path) as json_writer:
            source_info = video_source.info
            fps = source_info.fps if source_info.fps > 1e-3 else 25.0
            writer_fps = fps / frame_stride if frame_stride > 1 else fps
            writer = _make_video_writer(args.output, writer_fps, source_info.width, source_info.height)

            print(f"[INFO] 视频源已打开: {args.input}")
            print(f"[INFO] 检测器状态: {processor.face_detector.status_message}")
            print(f"[INFO] 关键点状态: {processor.face_mesh_estimator.status_message}")
            print(f"[INFO] 关键点回退: {feature_extractor.landmark_status}")
            print(
                "[INFO] 联合后端: face-recognition -> DMS "
                f"({source_info.width}x{source_info.height}, FPS={fps:.2f})"
            )

            while True:
                frame = video_source.read()
                if frame is None:
                    break

                frame_id = video_source.frame_index
                if (frame_id - 1) % frame_stride != 0:
                    continue

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = video_source.get_timestamp_ms()
                output = processor.process(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                )
                image_size = tuple(output.result["image_size"])
                driver_score = output.driver_face.score if output.driver_face is not None else None
                driver_bbox = output.driver_face.bbox if output.driver_face is not None else None
                features = feature_extractor.extract(
                    frame_bgr=frame,
                    timestamp_ms=timestamp_ms,
                    image_size=image_size,
                    landmarks=output.landmarks,
                    driver_bbox=driver_bbox,
                    driver_score=driver_score,
                    track_id=output.track_id,
                    face_count=len(output.faces),
                    detector_backend=processor.detector_backend,
                )
                result = dms_state.update(features)
                json_writer.write(_build_json_record(result.to_dict(), features))
                processed_frame_count += 1

                annotated = visualizer.draw(frame, features, result)
                if writer is not None:
                    writer.write(annotated)

                if args.display:
                    cv2.imshow("DMS Integrated Backend", annotated)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

                if args.max_frames is not None and processed_frame_count >= args.max_frames:
                    break
    finally:
        feature_extractor.close()
        processor.close()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"[INFO] 已读取 {processed_frame_count} 帧联合结果。")
    if args.output.lower() != "none":
        print(f"[INFO] 标注视频: {args.output}")
    if args.json.lower() != "none":
        print(f"[INFO] JSON 结果: {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())