from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# MediaPipe 导入链路会触发 matplotlib/fontconfig 缓存。把缓存放到 /tmp，
# 可以避免在只读 home 配置目录下反复等待或打印大量警告。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dms.config import load_config
from dms.features import FeatureExtractor
from dms.state import DMSState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 YawDD Mirror 子集上的打哈欠检测效果。")
    parser.add_argument(
        "--input-dir",
        default="data/local/archive/Mirror",
        help="YawDD Mirror 子集目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/yawdd_mirror_eval",
        help="评估结果输出目录。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选 YAML 配置路径。",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="每隔 N 帧处理一帧。默认 5，用于加速全量评估。",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="最多评估多少个视频，适合快速调试。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个视频最多处理多少帧，适合快速调试。",
    )
    parser.add_argument(
        "--write-frame-json",
        action="store_true",
        help="保存每个视频的逐帧 JSON。默认只保存按视频汇总，速度更快、占用更小。",
    )
    return parser.parse_args()


def label_from_name(path: Path) -> str:
    """从 YawDD Mirror 文件名解析动作标签。"""
    name = path.stem.lower()
    if "talkingyawning" in name:
        return "TalkingYawning"
    if "yawning" in name:
        return "Yawning"
    if "talking" in name:
        return "Talking"
    if "normal" in name:
        return "Normal"
    return "Unknown"


def is_yawn_label(label: str) -> bool:
    return label in {"Yawning", "TalkingYawning"}


def discover_videos(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.avi"))


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def summarize_frames(video_path: Path, label: str, records: list[dict[str, Any]], input_dir: Path) -> dict[str, Any]:
    """把逐帧结果压缩成一个视频级结果。"""
    if not records:
        return {
            "video": str(video_path.relative_to(input_dir)),
            "label": label,
            "true_yawn": int(is_yawn_label(label)),
            "processed_frames": 0,
            "face_found_frames": 0,
            "face_found_ratio": 0.0,
            "max_mar": 0.0,
            "mean_mar": 0.0,
            "max_yawn_count": 0,
            "pred_yawn_by_event": 0,
            "pred_yawn_by_mar": 0,
            "max_fatigue_score": 0.0,
            "fatigue_frames": 0,
            "error": "empty_records",
        }

    mar_values = [
        safe_float(record.get("features", {}).get("mar"))
        for record in records
        if record.get("features", {}).get("mar") is not None
    ]
    yawn_counts = [int(record.get("features", {}).get("yawn_count") or 0) for record in records]
    fatigue_scores = [safe_float(record.get("fatigue_score")) for record in records]
    fatigue_frames = sum(1 for record in records if record.get("fatigue_state") == "fatigue")
    face_found_frames = sum(1 for record in records if record.get("features", {}).get("mar") is not None)

    max_mar = max(mar_values) if mar_values else 0.0
    mean_mar = sum(mar_values) / len(mar_values) if mar_values else 0.0
    max_yawn_count = max(yawn_counts) if yawn_counts else 0

    return {
        "video": str(video_path.relative_to(input_dir)),
        "label": label,
        "true_yawn": int(is_yawn_label(label)),
        "processed_frames": len(records),
        "face_found_frames": face_found_frames,
        "face_found_ratio": round(face_found_frames / len(records), 4),
        "max_mar": round(max_mar, 4),
        "mean_mar": round(mean_mar, 4),
        "max_yawn_count": max_yawn_count,
        "pred_yawn_by_event": int(max_yawn_count > 0),
        "pred_yawn_by_mar": int(max_mar >= 0.60),
        "max_fatigue_score": round(max(fatigue_scores), 4) if fatigue_scores else 0.0,
        "fatigue_frames": fatigue_frames,
        "error": "",
    }


def process_video(
    video_path: Path,
    input_dir: Path,
    extractor: FeatureExtractor,
    config: dict[str, Any],
    frame_stride: int,
    max_frames: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = label_from_name(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (
            {
                "video": str(video_path.relative_to(input_dir)),
                "label": label,
                "true_yawn": int(is_yawn_label(label)),
                "processed_frames": 0,
                "face_found_frames": 0,
                "face_found_ratio": 0.0,
                "max_mar": 0.0,
                "mean_mar": 0.0,
                "max_yawn_count": 0,
                "pred_yawn_by_event": 0,
                "pred_yawn_by_mar": 0,
                "max_fatigue_score": 0.0,
                "fatigue_frames": 0,
                "error": "cannot_open_video",
            },
            [],
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    processed_count = 0
    records: list[dict[str, Any]] = []
    dms_state = DMSState(config)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp_ms <= 0:
            timestamp_ms = frame_index * 1000.0 / fps

        features = extractor.extract(frame, timestamp_ms)
        result = dms_state.update(features)
        records.append(result.to_dict())

        processed_count += 1
        frame_index += 1
        if max_frames is not None and processed_count >= max_frames:
            break

    cap.release()
    return summarize_frames(video_path, label, records, input_dir), records


def binary_metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("error")]
    tp = sum(1 for row in valid_rows if row["true_yawn"] == 1 and row[pred_key] == 1)
    fp = sum(1 for row in valid_rows if row["true_yawn"] == 0 and row[pred_key] == 1)
    tn = sum(1 for row in valid_rows if row["true_yawn"] == 0 and row[pred_key] == 0)
    fn = sum(1 for row in valid_rows if row["true_yawn"] == 1 and row[pred_key] == 0)
    total = max(len(valid_rows), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "prediction": pred_key,
        "videos": len(valid_rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def best_mar_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """在本次评估结果上扫描 MAR 阈值，找到 F1 最高的阈值。"""
    valid_rows = [row for row in rows if not row.get("error")]
    best = {"threshold": 0.0, "f1": -1.0}
    for threshold in [round(x / 100, 2) for x in range(20, 121)]:
        temp_rows = []
        for row in valid_rows:
            copy = dict(row)
            copy["pred_tmp"] = int(float(row["max_mar"]) >= threshold)
            temp_rows.append(copy)
        metrics = binary_metrics(temp_rows, "pred_tmp")
        if metrics["f1"] > best["f1"]:
            best = {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
    return best


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "label",
        "true_yawn",
        "processed_frames",
        "face_found_frames",
        "face_found_ratio",
        "max_mar",
        "mean_mar",
        "max_yawn_count",
        "pred_yawn_by_event",
        "pred_yawn_by_mar",
        "max_fatigue_score",
        "fatigue_frames",
        "error",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    frame_stride = max(1, int(args.frame_stride))

    videos = discover_videos(input_dir)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        print(f"没有找到 YawDD Mirror 视频：{input_dir}")
        return 0

    print(f"开始评估 {len(videos)} 个视频，frame_stride={frame_stride}。")
    rows: list[dict[str, Any]] = []

    frame_json_dir = output_dir / "frames"
    with FeatureExtractor(config) as extractor:
        for index, video_path in enumerate(videos, start=1):
            print(f"[{index}/{len(videos)}] {video_path.relative_to(input_dir)}")
            summary, records = process_video(
                video_path=video_path,
                input_dir=input_dir,
                extractor=extractor,
                config=config,
                frame_stride=frame_stride,
                max_frames=args.max_frames,
            )
            rows.append(summary)
            if args.write_frame_json:
                frame_json_path = frame_json_dir / Path(summary["video"]).with_suffix(".json")
                frame_json_path.parent.mkdir(parents=True, exist_ok=True)
                with frame_json_path.open("w", encoding="utf-8") as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)

    label_counts = Counter(row["label"] for row in rows)
    metrics = {
        "input_dir": str(input_dir),
        "frame_stride": frame_stride,
        "label_counts": dict(label_counts),
        "event_rule": binary_metrics(rows, "pred_yawn_by_event"),
        "mar_0_60_rule": binary_metrics(rows, "pred_yawn_by_mar"),
        "best_mar_threshold_on_this_run": best_mar_threshold(rows),
    }
    write_outputs(output_dir, rows, metrics)

    print(f"汇总结果：{output_dir / 'summary.csv'}")
    print(f"总体指标：{output_dir / 'metrics.json'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
