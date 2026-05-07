from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np


SAMPLE_FACE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal DMS feasibility smoke test.")
    parser.add_argument(
        "--output-dir",
        default="outputs/smoke_test",
        help="Directory for generated temporary videos and DMS outputs.",
    )
    parser.add_argument(
        "--face-image",
        default=None,
        help="Optional local face image path. If omitted with --download-face, a sample image is downloaded.",
    )
    parser.add_argument(
        "--download-face",
        action="store_true",
        help="Download a small public sample face image and run the face-detection smoke test.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate source videos in the output directory.",
    )
    return parser.parse_args()


def make_blank_video(path: Path, frame_count: int = 20, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create test video: {path}")

    for index in range(frame_count):
        frame = np.full((240, 320, 3), 32, dtype=np.uint8)
        cv2.putText(
            frame,
            f"DMS smoke {index:02d}",
            (42, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def make_face_video(image_path: Path, video_path: Path, frame_count: int = 30, fps: float = 10.0) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read face image: {image_path}")

    image = cv2.resize(image, (512, 512))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (512, 512))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create test video: {video_path}")

    for _ in range(frame_count):
        writer.write(image)
    writer.release()


def download_sample_face(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SAMPLE_FACE_URL, path)
    return path


def run_dms(input_video: Path, output_video: Path, output_json: Path, max_frames: int) -> None:
    command = [
        sys.executable,
        "-m",
        "dms.app",
        "--input",
        str(input_video),
        "--output",
        str(output_video),
        "--json",
        str(output_json),
        "--max-frames",
        str(max_frames),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"DMS command failed with exit code {completed.returncode}")


def load_results(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise AssertionError(f"Result JSON is empty or invalid: {path}")
    return data


def assert_common_outputs(video_path: Path, results: list[dict]) -> None:
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise AssertionError(f"Annotated video was not created: {video_path}")

    required = {"timestamp_ms", "driving_state", "fatigue_state", "attention_score", "fatigue_score", "features"}
    missing = required.difference(results[0])
    if missing:
        raise AssertionError(f"Result JSON missing fields: {sorted(missing)}")

    states = {"normal", "distracted"}
    fatigue_states = {"normal", "fatigue"}
    for record in results:
        if record["driving_state"] not in states:
            raise AssertionError(f"Unexpected driving_state: {record['driving_state']}")
        if record["fatigue_state"] not in fatigue_states:
            raise AssertionError(f"Unexpected fatigue_state: {record['fatigue_state']}")


def assert_face_features(results: list[dict]) -> None:
    has_face_features = any(
        record.get("features", {}).get("ear") is not None
        and record.get("features", {}).get("yaw") is not None
        for record in results
    )
    if not has_face_features:
        raise AssertionError("No face features were detected in the face smoke test.")


def run_blank_test(output_dir: Path) -> None:
    source_video = output_dir / "blank_input.mp4"
    annotated_video = output_dir / "blank_annotated.mp4"
    result_json = output_dir / "blank_results.json"

    make_blank_video(source_video)
    run_dms(source_video, annotated_video, result_json, max_frames=8)
    results = load_results(result_json)
    assert_common_outputs(annotated_video, results)
    print("[OK] Blank-video pipeline test passed.")


def run_face_test(output_dir: Path, face_image: Path | None, download_face: bool) -> None:
    if face_image is None:
        if not download_face:
            print("[SKIP] Face test skipped. Use --download-face or --face-image path/to/image.jpg.")
            return
        face_image = download_sample_face(output_dir / "sample_face.jpg")

    source_video = output_dir / "face_input.mp4"
    annotated_video = output_dir / "face_annotated.mp4"
    result_json = output_dir / "face_results.json"

    make_face_video(face_image, source_video)
    run_dms(source_video, annotated_video, result_json, max_frames=8)
    results = load_results(result_json)
    assert_common_outputs(annotated_video, results)
    assert_face_features(results)
    print("[OK] Face-landmark pipeline test passed.")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.keep_temp:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_blank_test(output_dir)
        run_face_test(output_dir, Path(args.face_image) if args.face_image else None, args.download_face)
        print(f"Smoke-test outputs kept in: {output_dir}")
        return 0

    with tempfile.TemporaryDirectory(prefix="dms_smoke_") as temp_dir:
        temp_output = Path(temp_dir)
        run_blank_test(temp_output)
        run_face_test(temp_output, Path(args.face_image) if args.face_image else None, args.download_face)

    print("Smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
