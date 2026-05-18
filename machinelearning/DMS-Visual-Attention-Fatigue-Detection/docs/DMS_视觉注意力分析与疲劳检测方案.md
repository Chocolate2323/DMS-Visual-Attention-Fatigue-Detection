# DMS 视觉注意力分析与疲劳检测方案

本方案对应课程 PPT 第 12-19 页中“视觉注意力分析”和“疲劳检测”的要求。系统输入为驾驶员侧视频，输出驾驶员当前驾驶状态（正常驾驶 / 分神）和疲劳状态（正常 / 疲劳），并保留中间特征用于可视化、调参和汇报。

## 1. 目标与输出

### 输入

- 离线视频文件：`mp4 / avi / mov`
- 或实时摄像头流：车内摄像头、电脑摄像头
- 推荐视角：仪表盘或 A 柱附近，能看到驾驶员正脸、眼睛、嘴部和头部动作

### 输出

每一帧或每 0.5 秒输出一次结构化结果：

```json
{
  "timestamp_ms": 12340,
  "driving_state": "normal|distracted",
  "fatigue_state": "normal|fatigue",
  "attention_score": 0.82,
  "fatigue_score": 0.18,
  "features": {
    "yaw": 4.2,
    "pitch": -3.1,
    "roll": 1.5,
    "gaze_x": 0.03,
    "gaze_y": -0.02,
    "ear": 0.29,
    "mar": 0.31,
    "perclos_30s": 0.06,
    "blink_rate_per_min": 17.0
  }
}
```

同时可输出带标注的视频，显示人脸关键点、头部姿态、视线方向、EAR/PERCLOS、当前状态。

## 2. 总体架构

```text
视频输入
  -> 帧读取与预处理
  -> 人脸与关键点检测
  -> 头部姿态估计
  -> 视线方向估计
  -> 眼部/嘴部行为特征提取
  -> 时间窗口统计与平滑
  -> 注意力状态判断
  -> 疲劳状态判断
  -> 结果 JSON + 可视化视频
```

推荐先做一个不依赖训练的规则版 MVP，再扩展为机器学习版。

- MVP：MediaPipe / OpenFace + OpenCV + 规则阈值 + 滑动窗口。
- 增强版：用 SVM、随机森林、XGBoost 或轻量 LSTM 对时间序列特征分类。

## 3. 推荐技术路线

### 方案 A：MediaPipe + OpenCV，推荐用于课程项目

优点是部署简单、速度快、Python 代码量少。MediaPipe Face Landmarker 可直接处理图片、视频和实时流，并输出 3D 人脸关键点、表情 blendshape 和人脸变换矩阵。OpenCV 的 `solvePnP` 可用于 3D-2D 点求解头部姿态。

依赖建议：

```bash
pip install opencv-python mediapipe numpy scipy scikit-learn pyyaml
```

### 方案 B：OpenFace 2.0，推荐作为对照或增强

OpenFace 已经包含人脸关键点、头部姿态、眼动视线和 AU 表情分析，适合直接跑视频生成 CSV，然后用我们自己的规则或分类器做状态判断。缺点是安装比 MediaPipe 麻烦，课程项目里更适合作为参考基线。

## 4. 视觉注意力分析设计

视觉注意力分析的核心思想：驾驶员是否看向前方道路，可以由头部朝向、眼睛视线和持续时间共同判断。短暂转头不一定是分神，持续偏离才判断为分神。

### 4.1 人脸关键点检测

每帧检测一张驾驶员人脸，提取：

- 脸部关键点：鼻尖、下巴、左右眼角、嘴角
- 眼部关键点：眼睑、眼角、虹膜中心
- 嘴部关键点：上下唇、嘴角

如果连续多帧检测不到人脸，可以视为高风险分神，例如低头、遮挡、离开座位。

### 4.2 头部姿态估计

用 2D 人脸关键点与预设 3D 人脸模型做 PnP 求解，得到旋转向量和平移向量，再换算为欧拉角：

- `yaw`：左右转头，偏航角
- `pitch`：抬头 / 低头，俯仰角
- `roll`：歪头，滚转角

建议使用的关键点：

| 语义点 | 用途 |
|---|---|
| 鼻尖 | 姿态中心点 |
| 下巴 | 约束 pitch |
| 左右眼角 | 约束 yaw / roll |
| 左右嘴角 | 稳定脸部平面 |

初始阈值建议：

| 条件 | 判断含义 |
|---|---|
| `abs(yaw) <= 25°` 且 `-15° <= pitch <= 20°` | 大概率看向前方 |
| `abs(yaw) > 30°` 持续 `> 1.5s` | 左右分神 |
| `pitch < -25°` 持续 `> 1.5s` | 低头分神，例如看手机 |
| 连续无脸 `> 1.0s` | 视为分神或严重遮挡 |

实际项目中应在视频前 3-5 秒采集“正常驾驶基线”，把驾驶员正常看前方时的 yaw/pitch 均值作为 `yaw0/pitch0`，再判断偏离程度。

### 4.3 视线方向估计

优先使用 MediaPipe 的眼部和虹膜关键点，不需要手写复杂瞳孔检测。计算虹膜中心相对眼角和眼睑的位置：

```text
gaze_x = iris_x 在左右眼角之间的归一化位置 - 正常基线
gaze_y = iris_y 在上下眼睑之间的归一化位置 - 正常基线
```

若课程要求贴合 PPT 中“眼部 ROI + 瞳孔检测”的流程，也可以作为备选：

1. 根据眼部关键点裁剪眼部 ROI。
2. 灰度化、直方图均衡、滤波。
3. Canny 边缘检测或阈值分割。
4. 霍夫圆或轮廓检测定位瞳孔中心。
5. 由眼球中心到瞳孔中心得到视线向量。

项目实现上建议用 MediaPipe 作为主线，传统瞳孔检测作为说明或备用。

### 4.4 注意力评分

注意力不是单帧判断，而是时间窗口评分：

```text
head_deviation = max(
  abs(yaw - yaw0) / yaw_threshold,
  abs(pitch - pitch0) / pitch_threshold
)

gaze_deviation = sqrt(gaze_x^2 + gaze_y^2) / gaze_threshold

attention_score = 1 - clip(
  0.55 * head_deviation +
  0.35 * gaze_deviation +
  0.10 * face_missing_score,
  0, 1
)
```

决策规则：

- `attention_score >= 0.60`：正常驾驶
- `attention_score < 0.60` 持续 `> 1.5s`：分神
- `attention_score < 0.40` 持续 `> 0.8s`：强分神，可立即报警

加入滑动平均和滞回机制，避免状态在正常 / 分神之间快速抖动。

## 5. 疲劳检测设计

疲劳检测主要使用眼睛闭合行为、眨眼频率、闭眼持续时间、PERCLOS、打哈欠和点头动作。视频输入下不使用心率、EEG 等生理信号，避免超出项目硬件范围。

### 5.1 EAR 眼睛纵横比

EAR 用眼部关键点计算眼睛开合程度：

```text
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

左右眼取平均：

```text
EAR_avg = (EAR_left + EAR_right) / 2
```

闭眼阈值建议：

- 固定阈值：`EAR < 0.20 ~ 0.25`
- 更稳的做法：前 3-5 秒估计个人基线 `EAR_open`，使用 `EAR < 0.65 * EAR_open` 判断闭眼

### 5.2 眨眼事件识别

用状态机处理 EAR 时间序列：

```text
OPEN -> CLOSED -> OPEN
```

判定：

- 连续闭眼 `2-12` 帧：一次正常眨眼
- 闭眼持续 `> 0.5s`：长眨眼，疲劳风险上升
- 闭眼持续 `> 1.0s`：强疲劳信号

输出特征：

- `blink_rate_per_min`：每分钟眨眼次数
- `avg_blink_duration`：平均眨眼持续时间
- `long_eye_closure_count`：长闭眼次数

### 5.3 PERCLOS

PERCLOS 表示一段时间内眼睛处于闭合状态的比例，是疲劳驾驶中常用指标。

```text
PERCLOS = 闭眼帧数 / 时间窗口内总有效帧数
```

建议窗口：

- 快速响应：`10s`
- 稳定判断：`30s`
- 报告指标：`60s`

初始阈值：

| PERCLOS | 疲劳判断 |
|---|---|
| `< 0.15` | 正常 |
| `0.15 - 0.20` | 疲劳预警 |
| `> 0.20` | 疲劳 |

课程输出只要求正常 / 疲劳时，可把 `> 0.20` 作为疲劳主阈值，`0.15 - 0.20` 作为内部 warning。

### 5.4 打哈欠检测，可选但推荐

用嘴部纵横比 MAR 判断嘴巴张开程度：

```text
MAR = mouth_vertical_distance / mouth_horizontal_distance
```

规则：

- `MAR > 0.60` 持续 `> 1.0s`：一次打哈欠
- 60 秒内多次打哈欠：疲劳分提高

### 5.5 点头检测，可选

如果只有视频没有加速度计，可用头部 `pitch` 时间序列近似检测点头：

- 低头角度突然增大
- 随后快速恢复
- 在 0.5-3 秒内形成一次周期性动作

规则示例：

- `pitch` 向下偏离基线超过 `20°`
- 峰值持续时间 `0.3-2.0s`
- 1 分钟内出现多次，则疲劳分提高

### 5.6 疲劳评分

```text
fatigue_score = clip(
  0.45 * perclos_score +
  0.25 * long_eye_closure_score +
  0.15 * yawn_score +
  0.15 * head_nod_score,
  0, 1
)
```

决策规则：

- `fatigue_score < 0.60`：正常
- `fatigue_score >= 0.60` 持续 `> 3s`：疲劳
- 任意一次闭眼 `> 1.5s`：可直接判定疲劳

## 6. 状态融合逻辑

驾驶状态和疲劳状态分开输出：

```text
driving_state = normal / distracted
fatigue_state = normal / fatigue
```

例子：

| 情况 | driving_state | fatigue_state |
|---|---|---|
| 看前方，眼睛正常 | normal | normal |
| 长时间低头看手机 | distracted | normal |
| 看前方但频繁长闭眼 | normal | fatigue |
| 低头且闭眼、打哈欠 | distracted | fatigue |

这样比只输出一个总状态更清晰，也符合项目要求。

## 7. 实现模块划分

推荐目录：

```text
dms/
  app.py                 # 主入口：读取视频、写结果
  face_tracker.py        # MediaPipe / OpenFace 封装
  head_pose.py           # PnP、欧拉角、姿态平滑
  gaze.py                # 视线方向估计
  fatigue.py             # EAR、PERCLOS、眨眼、打哈欠
  attention.py           # 注意力评分与分神判断
  fusion.py              # 状态融合、滞回、报警
  visualizer.py          # 视频标注
  config.yaml            # 阈值配置
```

主流程伪代码：

```python
cap = cv2.VideoCapture(input_video)
state = DMSState(config)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    landmarks = face_tracker.detect(frame, timestamp_ms)

    features = feature_extractor.compute(frame, landmarks)
    result = state.update(features, timestamp_ms)

    json_writer.write(result)
    annotated = visualizer.draw(frame, landmarks, features, result)
    video_writer.write(annotated)
```

命令行示例：

```bash
python -m dms.app --input data/driver.mp4 --output outputs/result.mp4 --json outputs/result.json
```

## 8. 阈值校准方案

因为摄像头位置、驾驶员脸型、眼睛大小差异很大，建议引入自动校准：

1. 视频前 3-5 秒要求驾驶员正常看前方。
2. 记录 `yaw0/pitch0/roll0`、`EAR_open`、`gaze_x0/gaze_y0`。
3. 后续特征使用相对偏移，而不是绝对值。
4. 如果前 5 秒检测不稳定，则回退到默认阈值。

这样可以显著降低误报。

## 9. 机器学习增强方案

如果团队希望体现“机器学习项目”特色，可以在规则版之后增加分类器：

### 特征向量

每 1 秒或 3 秒聚合一次：

```text
[
  mean_yaw, std_yaw, max_abs_yaw,
  mean_pitch, std_pitch, min_pitch,
  mean_gaze_x, mean_gaze_y,
  mean_ear, min_ear,
  perclos_10s, perclos_30s,
  blink_rate, avg_blink_duration,
  yawn_count, head_nod_count,
  face_missing_ratio
]
```

### 模型选择

- SVM：适合小数据，便于课程讲解。
- Random Forest：抗噪声强，能输出特征重要性。
- LSTM / TCN：适合视频时间序列，但实现成本更高。

建议课程项目采用 Random Forest 或 SVM，把规则版作为 baseline。

### 标签

两个二分类任务：

- 注意力：`normal / distracted`
- 疲劳：`normal / fatigue`

最后两个模型分别输出概率，再用融合模块生成最终状态。

## 10. 可参考开源项目与资料

- MediaPipe Face Landmarker：用于人脸关键点、3D landmarks、视频模式和实时流处理。官方文档：https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
- OpenCV solvePnP：用于通过 3D-2D 点估计头部姿态。官方文档：https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- OpenFace 2.0：开源人脸行为分析工具，包含关键点、头部姿态、眼动视线等能力。GitHub：https://github.com/TadasBaltrusaitis/OpenFace
- Soukupova & Cech 眼睛眨眼检测：EAR 方法的经典参考。资料页：https://dspace.cvut.cz/entities/publication/fbcbe690-daaf-4037-9edd-41a51703ecec
- NTHU Driver Drowsiness Detection Dataset：疲劳检测数据集，可用于训练或实验对比。官网：https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- Driver Distraction Dataset：包含安全驾驶、打电话、发短信、喝水、操作收音机等分神类别，可作为分神检测参考。GitHub：https://github.com/AmalEzzouhri/Driver-Distraction-Dataset

## 11. PPT 可讲述结构

### 视觉注意力分析

1. 输入驾驶员视频。
2. 使用 MediaPipe/OpenFace 检测人脸和关键点。
3. 用 PnP 计算头部姿态 yaw/pitch/roll。
4. 用虹膜与眼部关键点估计视线方向。
5. 将头部偏转、视线偏离和人脸缺失输入时间窗口评分。
6. 持续偏离阈值则输出“分神”，否则输出“正常驾驶”。

### 疲劳检测

1. 从眼部关键点计算 EAR。
2. 通过 EAR 时间序列检测眨眼和长闭眼。
3. 在 30 秒窗口计算 PERCLOS。
4. 可选加入 MAR 打哈欠和 pitch 点头检测。
5. 通过加权疲劳分或 RF/SVM 分类器输出“正常 / 疲劳”。

### 项目亮点

- 使用单目普通摄像头，硬件要求低。
- 规则版可实时运行，机器学习版可进一步提升鲁棒性。
- 输出包含可解释特征，便于调参和演示。
- 注意力状态与疲劳状态独立输出，结果更符合真实驾驶场景。
