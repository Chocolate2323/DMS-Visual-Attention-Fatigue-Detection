from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ImportError:  # pragma: no cover - 可选依赖
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DriverSide = Literal["left", "center", "right"]


@dataclass(slots=True)
class InputConfig:
    source: str | int = 0
    width: int | None = None
    height: int | None = None
    target_fps: float | None = None


@dataclass(slots=True)
class PreprocessConfig:
    resize_width: int = 640
    resize_height: int = 480


@dataclass(slots=True)
class DetectionConfig:
    backend: str = "auto"
    model_path: str = "models/face_detection/yolo_face.pt"
    confidence_threshold: float = 0.35
    input_size: int = 640


@dataclass(slots=True)
class TrackingConfig:
    driver_side: DriverSide = "left"
    iou_threshold: float = 0.35
    max_lost_frames: int = 8
    position_weight: float = 0.6
    area_weight: float = 0.4


@dataclass(slots=True)
class LandmarkConfig:
    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    bbox_margin: float = 0.15


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = True
    window_name: str = "DMS Pipeline"
    draw_all_faces: bool = True
    draw_landmarks: bool = True
    draw_iris: bool = True
    landmark_stride: int = 1


@dataclass(slots=True)
class OutputConfig:
    save_jsonl: bool = False
    output_path: str = "data/outputs/frame_results.jsonl"


@dataclass(slots=True)
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    landmark: LandmarkConfig = field(default_factory=LandmarkConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path | None) -> "AppConfig":
        if config_path is None:
            return cls()

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        if yaml is None:
            raise RuntimeError("未安装 PyYAML，无法读取 YAML 配置文件。")

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            input=InputConfig(**raw.get("input", {})),
            preprocess=PreprocessConfig(**raw.get("preprocess", {})),
            detection=DetectionConfig(**raw.get("detection", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
            landmark=LandmarkConfig(**raw.get("landmark", {})),
            visualization=VisualizationConfig(**raw.get("visualization", {})),
            output=OutputConfig(**raw.get("output", {})),
        )

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    return AppConfig.from_yaml(config_path)
