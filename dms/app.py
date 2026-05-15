from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .config import load_config
from .features import FeatureExtractor
from .output import JsonArrayWriter
from .state import DMSState
from .visualizer import Visualizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从驾驶员视频中检测视觉注意力和疲劳状态。",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入视频路径或摄像头编号，例如 0。",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/annotated.mp4",
        help="标注视频输出路径。传入 none 可关闭视频输出。",
    )
    parser.add_argument(
        "--json",
        default="outputs/results.json",
        help="JSON 结果输出路径。传入 none 可关闭 JSON 输出。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选 YAML 配置路径，会覆盖 configs/default.yaml 中的同名字段。",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="在本地 OpenCV 窗口中实时显示标注画面。",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="水平镜像画面，适合摄像头预览。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多处理 N 帧，适合快速测试。",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="每隔 N 帧处理一帧。默认 1 表示不抽帧，数据集快速评估可设为 5 或 10。",
    )
    return parser.parse_args(argv)


def open_capture(source: str) -> cv2.VideoCapture:
    """打开视频文件或摄像头。纯数字字符串会被视为摄像头编号。"""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def make_video_writer(path: str | None, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    """按输出后缀创建 OpenCV 视频写入器。"""
    if not path or path.lower() == "none":
        return None

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".avi"}:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cap = open_capture(args.input)
    if not cap.isOpened():
        print(f"无法打开输入：{args.input}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_stride = max(1, int(args.frame_stride))
    writer_fps = fps / frame_stride if frame_stride > 1 else fps
    writer = make_video_writer(args.output, writer_fps, width, height)
    json_path = None if args.json.lower() == "none" else args.json

    dms_state = DMSState(config)
    visualizer = Visualizer(config)
    frame_index = 0
    processed_frame_count = 0

    with FeatureExtractor(config) as extractor, JsonArrayWriter(json_path) as json_writer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            should_process = frame_index % frame_stride == 0
            frame_index += 1
            if not should_process:
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms <= 0:
                # 有些摄像头或编码器不给时间戳，用帧号和 FPS 估算。
                timestamp_ms = (frame_index - 1) * 1000.0 / fps

            features = extractor.extract(frame, timestamp_ms)
            result = dms_state.update(features)
            json_writer.write(result.to_dict())
            processed_frame_count += 1

            annotated = visualizer.draw(frame, features, result)
            if writer is not None:
                writer.write(annotated)

            if args.display:
                cv2.imshow("DMS", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.max_frames is not None and processed_frame_count >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"已读取 {frame_index} 帧，实际处理 {processed_frame_count} 帧。")
    if args.output.lower() != "none":
        print(f"标注视频：{args.output}")
    if args.json.lower() != "none":
        print(f"JSON 结果：{args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
