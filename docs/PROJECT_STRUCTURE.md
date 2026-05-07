# Project Structure

```text
DMS-Visual-Attention-Fatigue-Detection/
  README.md
  requirements.txt
  configs/
    default.yaml
  dms/
    app.py
    face_tracker.py
    head_pose.py
    gaze.py
    fatigue.py
    attention.py
    state.py
    visualizer.py
  models/
    face_landmarker.task
  scripts/
    smoke_test.py
    run_dataset.py
  data/
    README.md
    local/
    samples/
  outputs/
  docs/
    DMS_视觉注意力分析与疲劳检测方案.md
    PROJECT_STRUCTURE.md
```

## Directory Roles

- `dms/`: main Python package and algorithm implementation.
- `configs/`: threshold and runtime configuration.
- `models/`: lightweight model files required at runtime.
- `scripts/`: utility scripts for smoke tests and dataset batch processing.
- `data/local/`: your local dataset videos. Ignored by Git.
- `data/samples/`: small local samples. Ignored by Git except the placeholder.
- `outputs/`: generated annotated videos, JSON results, and summaries. Ignored by Git.
- `docs/`: design notes and project documentation.

## Local Dataset Workflow

1. Put videos under `data/local/`.
2. Run `python scripts/smoke_test.py --download-face --keep-temp` once to check the environment.
3. Run `python scripts/run_dataset.py --input-dir data/local --output-dir outputs/dataset_runs`.
4. Inspect `outputs/dataset_runs/summary.csv` first, then open individual JSON or annotated videos.
