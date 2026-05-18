# 本地数据目录

把本地测试视频放在这里。本目录中的数据集内容会被 Git 忽略，因此大体积视频不会上传到 GitHub。

推荐目录结构：

```text
data/
  local/
    normal/
      driver_001.mp4
    distracted/
      phone_001.mp4
    fatigue/
      fatigue_001.mp4
  samples/
    small_demo.mp4
```

运行单个视频：

```bash
python -m dms.app --input data/local/normal/driver_001.mp4 --output outputs/single/driver_001_annotated.mp4 --json outputs/single/driver_001_results.json
```

批量运行整个本地数据集：

```bash
python scripts/run_dataset.py --input-dir data/local --output-dir outputs/dataset_runs
```
