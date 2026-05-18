# 项目结构

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
    PROJECT_REVIEW.md
```

## 目录作用

- `dms/`：核心 Python 包和算法实现。
- `configs/`：阈值与运行配置。
- `models/`：运行时需要的轻量模型文件。
- `scripts/`：最小测试和数据集批处理脚本。
- `data/local/`：本地数据集视频目录，内容被 Git 忽略。
- `data/samples/`：少量本地样例目录，内容被 Git 忽略，只保留占位符。
- `outputs/`：标注视频、JSON 结果和汇总文件，内容被 Git 忽略。
- `docs/`：设计方案、审查记录和项目说明。

## 本地数据集流程

1. 将视频放到 `data/local/`。
2. 先运行 `python scripts/smoke_test.py --download-face --keep-temp` 检查环境。
3. 再运行 `python scripts/run_dataset.py --input-dir data/local --output-dir outputs/dataset_runs`。
4. 优先查看 `outputs/dataset_runs/summary.csv`，再打开单个 JSON 或标注视频检查细节。
