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
        description="DMS visual attention and fatigue detection from driver video.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input video path or camera index, for example 0.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/annotated.mp4",
        help="Annotated output video path. Use 'none' to disable video writing.",
    )
    parser.add_argument(
        "--json",
        default="outputs/results.json",
        help="JSON result path. Use 'none' to disable JSON writing.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config path. Values override configs/default.yaml.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show annotated frames in a local OpenCV window.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror frames horizontally, useful for webcam preview.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames, useful for quick tests.",
    )
    return parser.parse_args(argv)


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def make_video_writer(path: str | None, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
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
        print(f"Cannot open input: {args.input}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-3:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = make_video_writer(args.output, fps, width, height)
    json_path = None if args.json.lower() == "none" else args.json

    dms_state = DMSState(config)
    visualizer = Visualizer(config)
    frame_index = 0

    with FeatureExtractor(config) as extractor, JsonArrayWriter(json_path) as json_writer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms <= 0:
                timestamp_ms = frame_index * 1000.0 / fps

            features = extractor.extract(frame, timestamp_ms)
            result = dms_state.update(features)
            json_writer.write(result.to_dict())

            annotated = visualizer.draw(frame, features, result)
            if writer is not None:
                writer.write(annotated)

            if args.display:
                cv2.imshow("DMS", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames.")
    if args.output.lower() != "none":
        print(f"Annotated video: {args.output}")
    if args.json.lower() != "none":
        print(f"JSON results: {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
