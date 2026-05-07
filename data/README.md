# Local Data Directory

Put local test videos here. The contents of this directory are ignored by Git, so large datasets will not be uploaded to GitHub.

Recommended layout:

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

Run a single video:

```bash
python -m dms.app --input data/local/normal/driver_001.mp4 --output outputs/single/driver_001_annotated.mp4 --json outputs/single/driver_001_results.json
```

Run the whole local dataset:

```bash
python scripts/run_dataset.py --input-dir data/local --output-dir outputs/dataset_runs
```
