# DMS 视觉注意力分析与疲劳检测

本目录实现了一个基于视频的 DMS MVP：输入驾驶员视频，输出驾驶员状态 `normal / distracted` 和疲劳状态 `normal / fatigue`，并生成标注视频和 JSON 结果。

## 环境

代码已按 conda 环境 `machine_learning` 实现和验证。依赖见 `requirements.txt`。

```bash
conda activate machine_learning
pip install -r requirements.txt
```

项目已包含 `models/face_landmarker.task`。如果换机器后缺失该文件，可以从 MediaPipe 官方模型地址下载，并在 `configs/default.yaml` 的 `runtime.face_landmarker_model` 中指定路径。

## 运行

处理视频：

```bash
python -m dms.app --input data/driver.mp4 --output outputs/annotated.mp4 --json outputs/results.json
```

使用摄像头：

```bash
python -m dms.app --input 0 --mirror --display
```

快速检查前 100 帧：

```bash
python -m dms.app --input data/driver.mp4 --max-frames 100
```

## 最小可行性测试

在跑自己的数据集前，可以先运行 smoke test，确认环境、模型、视频读取、JSON 输出和标注视频流程可用。

只测试基础视频管线：

```bash
python scripts/smoke_test.py
```

下载一张公开人脸样例，并验证人脸关键点、头姿、EAR 等特征能输出：

```bash
python scripts/smoke_test.py --download-face --keep-temp
```

如果不想联网，也可以使用本地任意正脸图片：

```bash
python scripts/smoke_test.py --face-image path/to/face.jpg --keep-temp
```

测试结果会放在 `outputs/smoke_test/`，其中包括输入测试视频、标注视频和 JSON 结果。

## 输出

`outputs/results.json` 是 JSON 数组，每个元素包含：

- `driving_state`：`normal` 或 `distracted`
- `fatigue_state`：`normal` 或 `fatigue`
- `attention_score`：注意力评分，越高越专注
- `fatigue_score`：疲劳评分，越高越疲劳
- `features`：头部姿态、视线、EAR、PERCLOS、眨眼频率等中间特征

## 实现思路

- `dms/face_tracker.py`：MediaPipe Face Mesh 人脸关键点检测
- `dms/head_pose.py`：OpenCV `solvePnP` 估计 `yaw / pitch / roll`
- `dms/gaze.py`：根据虹膜在眼部区域的位置估计粗略视线方向
- `dms/fatigue.py`：EAR、眨眼、长闭眼、PERCLOS、打哈欠、点头检测
- `dms/attention.py`：头姿、视线、人脸缺失融合为注意力状态
- `dms/state.py`：基线校准、平滑、状态融合
- `dms/visualizer.py`：视频标注

默认会使用视频前 5 秒进行基线校准。阈值可在 `configs/default.yaml` 中调整。
