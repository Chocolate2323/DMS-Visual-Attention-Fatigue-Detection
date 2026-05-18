from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 依赖缺失时仅做兜底
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from src.config import AppConfig, load_config
from src.detection.face_detector import FaceDetector
from src.io.video_source import VideoSource
from src.iris.iris_estimator import IrisEstimator
from src.landmark.face_mesh_estimator import FaceMeshEstimator
from src.pipeline.frame_processor import FrameProcessor
from src.preprocess.image_preprocessor import ImagePreprocessor
from src.tracking.driver_selector import DriverSelector
from src.utils.serializer import FrameSerializer, JsonlResultWriter
from src.utils.visualizer import draw_result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DMS 输入与核心处理流水线 MVP")
    parser.add_argument("--config", type=str, default=None, help="可选的 YAML 配置文件路径")
    parser.add_argument("--source", type=str, default=None, help="摄像头编号或视频文件路径")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "yolo", "mediapipe"],
        default=None,
        help="人脸检测后端",
    )
    parser.add_argument("--weights", type=str, default=None, help="YOLO 人脸权重路径")
    parser.add_argument(
        "--driver-side",
        type=str,
        choices=["left", "center", "right"],
        default=None,
        help="主驾驶员所在侧",
    )
    parser.add_argument("--resize-width", type=int, default=None, help="预处理宽度")
    parser.add_argument("--resize-height", type=int, default=None, help="预处理高度")
    parser.add_argument("--camera-width", type=int, default=None, help="摄像头采集宽度")
    parser.add_argument("--camera-height", type=int, default=None, help="摄像头采集高度")
    parser.add_argument("--output-jsonl", type=str, default=None, help="结构化结果输出路径")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    parser.add_argument("--no-display", action="store_true", help="不显示可视化窗口")
    return parser.parse_args(argv)


def _coerce_source(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.source is not None:
        config.input.source = _coerce_source(args.source)
    if args.backend is not None:
        config.detection.backend = args.backend
    if args.weights is not None:
        config.detection.model_path = args.weights
    if args.driver_side is not None:
        config.tracking.driver_side = args.driver_side
    if args.resize_width is not None:
        config.preprocess.resize_width = args.resize_width
    if args.resize_height is not None:
        config.preprocess.resize_height = args.resize_height
    if args.camera_width is not None:
        config.input.width = args.camera_width
    if args.camera_height is not None:
        config.input.height = args.camera_height
    if args.output_jsonl is not None:
        config.output.save_jsonl = True
        config.output.output_path = args.output_jsonl
    if args.no_display:
        config.visualization.enabled = False
    return config


def _resolve_input_source(config: AppConfig) -> str | int:
    source = config.input.source
    if isinstance(source, int):
        return source
    if str(source).isdigit():
        return int(str(source))
    return str(config.resolve_path(source))


def build_processor(config: AppConfig) -> FrameProcessor:
    preprocessor = ImagePreprocessor(
        target_width=config.preprocess.resize_width,
        target_height=config.preprocess.resize_height,
    )
    detector = FaceDetector(
        backend=config.detection.backend,
        model_path=config.resolve_path(config.detection.model_path),
        confidence_threshold=config.detection.confidence_threshold,
        input_size=config.detection.input_size,
    )
    driver_selector = DriverSelector(
        driver_side=config.tracking.driver_side,
        iou_threshold=config.tracking.iou_threshold,
        max_lost_frames=config.tracking.max_lost_frames,
        position_weight=config.tracking.position_weight,
        area_weight=config.tracking.area_weight,
    )
    face_mesh = FaceMeshEstimator(
        static_image_mode=config.landmark.static_image_mode,
        max_num_faces=config.landmark.max_num_faces,
        refine_landmarks=config.landmark.refine_landmarks,
        min_detection_confidence=config.landmark.min_detection_confidence,
        min_tracking_confidence=config.landmark.min_tracking_confidence,
        bbox_margin=config.landmark.bbox_margin,
    )
    iris_estimator = IrisEstimator()
    serializer = FrameSerializer()
    return FrameProcessor(
        preprocessor=preprocessor,
        face_detector=detector,
        driver_selector=driver_selector,
        face_mesh_estimator=face_mesh,
        iris_estimator=iris_estimator,
        serializer=serializer,
    )


def run(config: AppConfig, max_frames: int | None = None) -> int:
    if config.visualization.enabled and cv2 is None:
        raise RuntimeError("OpenCV 未安装，无法显示可视化窗口。") from _CV2_IMPORT_ERROR

    input_source = _resolve_input_source(config)
    processor = build_processor(config)
    writer = None
    if config.output.save_jsonl:
        writer = JsonlResultWriter(config.resolve_path(config.output.output_path))

    processed_frames = 0
    try:
        with VideoSource(
            source=input_source,
            width=config.input.width,
            height=config.input.height,
            target_fps=config.input.target_fps,
        ) as video_source:
            print(f"[INFO] 视频源已打开: {input_source}")
            print(f"[INFO] 检测器状态: {processor.face_detector.status_message}")
            print(f"[INFO] 关键点状态: {processor.face_mesh_estimator.status_message}")
            source_info = video_source.info
            print(
                "[INFO] 输入分辨率: "
                f"{source_info.width}x{source_info.height}, FPS={source_info.fps:.2f}"
            )

            while True:
                frame = video_source.read()
                if frame is None:
                    break

                output = processor.process(
                    frame=frame,
                    frame_id=video_source.frame_index,
                    timestamp_ms=video_source.get_timestamp_ms(),
                )
                processed_frames += 1

                if writer is not None:
                    writer.write(output.result)

                if config.visualization.enabled and cv2 is not None:
                    annotated = draw_result(
                        frame,
                        output.faces,
                        output.driver_face,
                        output.landmarks,
                        output.iris,
                        frame_id=video_source.frame_index,
                        track_id=output.track_id,
                        detector_backend=processor.detector_backend,
                        draw_all_faces=config.visualization.draw_all_faces,
                        draw_landmarks=config.visualization.draw_landmarks,
                        draw_iris=config.visualization.draw_iris,
                        landmark_stride=config.visualization.landmark_stride,
                    )
                    cv2.imshow(config.visualization.window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if max_frames is not None and processed_frames >= max_frames:
                    break
    finally:
        if writer is not None:
            writer.close()
        processor.close()
        if config.visualization.enabled and cv2 is not None:
            cv2.destroyAllWindows()

    print(f"[INFO] 处理完成，总帧数: {processed_frames}")
    if writer is not None:
        print(f"[INFO] 结构化结果已写入: {writer.output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(load_config(args.config), args)
    try:
        return run(config, max_frames=args.max_frames)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
