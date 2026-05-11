from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量处理本地视频数据集。")
    parser.add_argument(
        "--input-dir",
        default="data/local",
        help="本地数据集视频目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dataset_runs",
        help="标注视频、JSON 结果和汇总文件的输出目录。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="可选 YAML 配置路径，会传给 dms.app。",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="需要处理的视频扩展名。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="每个视频最多处理的帧数，适合快速检查。",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成标注视频，只生成 JSON 和汇总文件。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果某个视频的 JSON 结果已经存在，则跳过。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任意视频失败时立即停止。",
    )
    return parser.parse_args()


def discover_videos(input_dir: Path, extensions: list[str]) -> list[Path]:
    """递归查找指定扩展名的视频文件。"""
    normalized = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


def output_paths(input_dir: Path, output_dir: Path, video_path: Path) -> tuple[Path, Path]:
    """保持输入目录层级，为每个视频生成独立输出目录。"""
    relative = video_path.relative_to(input_dir)
    video_output_dir = output_dir / relative.parent / relative.stem
    annotated_path = video_output_dir / f"{relative.stem}_annotated.mp4"
    json_path = video_output_dir / f"{relative.stem}_results.json"
    return annotated_path, json_path


def run_single_video(
    video_path: Path,
    annotated_path: Path,
    json_path: Path,
    config: str | None,
    max_frames: int | None,
    write_video: bool,
) -> None:
    """调用 dms.app 处理单个视频。"""
    annotated_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "dms.app",
        "--input",
        str(video_path),
        "--output",
        str(annotated_path if write_video else "none"),
        "--json",
        str(json_path),
    ]
    if config:
        command.extend(["--config", config])
    if max_frames is not None:
        command.extend(["--max-frames", str(max_frames)])

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())


def load_records(json_path: Path) -> list[dict[str, Any]]:
    """读取单个视频的逐帧 JSON 结果。"""
    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Result file is not a JSON array: {json_path}")
    return records


def summarize_records(input_dir: Path, video_path: Path, json_path: Path, status: str, error: str = "") -> dict[str, Any]:
    """把逐帧结果压缩成一行汇总，便于批量查看。"""
    row: dict[str, Any] = {
        "video": str(video_path.relative_to(input_dir)),
        "status": status,
        "frames": 0,
        "final_driving_state": "",
        "final_fatigue_state": "",
        "distracted_frames": 0,
        "fatigue_frames": 0,
        "min_attention_score": "",
        "max_fatigue_score": "",
        "json": str(json_path),
        "error": error,
    }
    if status != "ok":
        return row

    records = load_records(json_path)
    if not records:
        row["error"] = "empty result json"
        row["status"] = "failed"
        return row

    driving_counts = Counter(record.get("driving_state") for record in records)
    fatigue_counts = Counter(record.get("fatigue_state") for record in records)
    attention_scores = [float(record.get("attention_score", 0.0)) for record in records]
    fatigue_scores = [float(record.get("fatigue_score", 0.0)) for record in records]

    row.update(
        {
            "frames": len(records),
            "final_driving_state": records[-1].get("driving_state", ""),
            "final_fatigue_state": records[-1].get("fatigue_state", ""),
            "distracted_frames": driving_counts.get("distracted", 0),
            "fatigue_frames": fatigue_counts.get("fatigue", 0),
            "min_attention_score": round(min(attention_scores), 4),
            "max_fatigue_score": round(max(fatigue_scores), 4),
        }
    )
    return row


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """同时写出 CSV 和 JSON 两种汇总格式。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"

    fieldnames = [
        "video",
        "status",
        "frames",
        "final_driving_state",
        "final_fatigue_state",
        "distracted_frames",
        "fatigue_frames",
        "min_attention_score",
        "max_fatigue_score",
        "json",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"输入目录不存在：{input_dir}", file=sys.stderr)
        return 2

    videos = discover_videos(input_dir, args.extensions)
    if not videos:
        print(f"在 {input_dir} 下没有找到视频。扩展名：{', '.join(args.extensions)}")
        return 0

    print(f"找到 {len(videos)} 个视频。")
    rows: list[dict[str, Any]] = []

    for index, video_path in enumerate(videos, start=1):
        annotated_path, json_path = output_paths(input_dir, output_dir, video_path)
        print(f"[{index}/{len(videos)}] {video_path.relative_to(input_dir)}")

        if args.skip_existing and json_path.exists():
            rows.append(summarize_records(input_dir, video_path, json_path, status="ok"))
            print("  已跳过已有结果")
            continue

        try:
            run_single_video(
                video_path=video_path,
                annotated_path=annotated_path,
                json_path=json_path,
                config=args.config,
                max_frames=args.max_frames,
                write_video=not args.no_video,
            )
            rows.append(summarize_records(input_dir, video_path, json_path, status="ok"))
            print(f"  成功 -> {json_path}")
        except Exception as exc:
            message = str(exc)
            rows.append(summarize_records(input_dir, video_path, json_path, status="failed", error=message))
            print(f"  失败：{message}", file=sys.stderr)
            if args.stop_on_error:
                write_summary(output_dir, rows)
                return 1

    write_summary(output_dir, rows)
    print(f"汇总文件：{output_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
