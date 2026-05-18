# 人脸识别与关键点定位

实现 DMS（驾驶员监控系统）方案图左侧与中间部分的核心流水线：

- 数据输入（摄像头 / 视频文件）
- 图像预处理
- 人脸检测（YOLOv8n-face）
- 主驾驶员识别与跟踪
- 人脸关键点检测（MediaPipe FaceMesh，478 点）
- 虹膜 / 瞳孔定位
- 结构化数据输出（JSONL）

**当前阶段**：MVP 链路已跑通，正在做链路稳定性验证与结果质量检查。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.10（推荐，高版本 mediapipe 兼容性差） |
| mediapipe | **必须 == 0.10.14**（高版本移除了 solutions API） |
| ultralytics | >= 8.3，当前验证版本 8.4.48 |
| OpenCV | >= 4.10，当前验证版本 4.13.0 |

---

## 快速开始

### 第一步：激活环境并安装依赖

```powershell
conda activate dms-face
pip install -r requirements.txt
```

### 第二步：下载 YOLO 人脸权重

从 [akanametov/yolo-face Releases](https://github.com/akanametov/yolo-face/releases) 下载 `yolov8n-face.pt`，放到：

```
models/face_detection/yolov8n-face.pt
```

### 第三步：运行

```powershell
# 摄像头实时检测
python ./src/main.py --backend yolo --weights models/face_detection/yolov8n-face.pt

# 视频文件检测
python ./src/main.py --source data/demo_videos/demo.mp4 --backend yolo --weights models/face_detection/yolov8n-face.pt

# 视频检测 + 保存结构化结果
python ./src/main.py --source data/demo_videos/demo.mp4 --backend yolo --weights models/face_detection/yolov8n-face.pt --output-jsonl data/outputs/result.jsonl

# 无界面快速自检（前 100 帧）
python ./src/main.py --source data/demo_videos/demo.mp4 --no-display --max-frames 100 --backend yolo --weights models/face_detection/yolov8n-face.pt

# 无权重回退模式（MediaPipe，不需要 .pt 文件）
python ./src/main.py --source data/demo_videos/demo.mp4 --backend mediapipe
```

---

## 命令行参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--source` | 摄像头编号（0/1/2）或视频文件路径 | `0`（摄像头） |
| `--backend` | 检测后端：`auto` / `yolo` / `mediapipe` | `auto` |
| `--weights` | YOLO 权重路径 | `models/face_detection/yolo_face.pt` |
| `--driver-side` | 主驾驶员位置：`left` / `center` / `right` | `left` |
| `--output-jsonl` | JSONL 结果输出路径（指定后自动落盘） | 不输出 |
| `--max-frames` | 最多处理帧数（用于快速测试） | 不限制 |
| `--no-display` | 不弹出可视化窗口 | 显示 |
| `--resize-width` | 预处理缩放宽度 | `640` |
| `--resize-height` | 预处理缩放高度 | `480` |
| `--config` | 可选 YAML 配置文件路径 | 无 |

---

## 项目结构

```
人脸识别与关键点定位/
├── data/
│   ├── demo_videos/          # 测试视频（放这里）
│   └── outputs/              # JSONL 输出结果
├── docs/
│   └── 从0到1实操指南.md
├── models/
│   └── face_detection/
│       └── yolov8n-face.pt   # YOLO 人脸权重（需自行下载）
├── src/
│   ├── io/
│   │   └── video_source.py           # 摄像头 / 视频输入
│   ├── preprocess/
│   │   └── image_preprocessor.py    # BGR↔RGB 转换与缩放
│   ├── detection/
│   │   └── face_detector.py          # YOLOv8 + MediaPipe 双后端
│   ├── tracking/
│   │   └── driver_selector.py        # 主驾驶员选择与 IoU 跟踪
│   ├── landmark/
│   │   └── face_mesh_estimator.py    # FaceMesh 478 点关键点
│   ├── iris/
│   │   └── iris_estimator.py         # 虹膜中心估计
│   ├── pipeline/
│   │   └── frame_processor.py        # 整条流水线串联
│   ├── utils/
│   │   ├── serializer.py             # 结构化输出与 JSONL 写入
│   │   └── visualizer.py            # 检测框 / 关键点可视化
│   ├── config.py                     # 配置数据类
│   └── main.py                       # 程序入口
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 输出数据格式

每帧输出一行 JSON（JSONL 格式，每行独立可解析）：

```json
{
  "frame_id": 42,
  "timestamp_ms": 1400.0,
  "image_size": [1280, 720],
  "face_count": 1,
  "face_bbox": [[120, 80, 320, 340]],
  "driver_detected": true,
  "driver_bbox": [120, 80, 320, 340],
  "driver_score": 0.91,
  "track_id": 1,
  "landmarks": [[150, 120], [155, 125]],
  "iris_left": [175, 200],
  "iris_right": [240, 198],
  "confidence": 0.91
}
```

后续头姿估计、视线分析、EAR/MAR 等分析模块直接读取此结构，无需重新处理图像。

---

## 检测后端对比

| 后端 | 需要权重文件 | 推荐场景 |
|---|---|---|
| `yolo`（yolov8n-face） | 是，需下载 `.pt` | 正式使用，检测质量更高 |
| `mediapipe` | 否，内置模型 | 无权重时快速验证链路 |
| `auto` | 自动判断 | 默认，有权重用 YOLO，否则回退 MediaPipe |

---

## ⚠ MediaPipe 版本说明

`mediapipe >= 0.10.18` 移除了 `solutions` API，本项目当前实现依赖此 API。

**必须使用 `mediapipe==0.10.14`**，否则会报错：

```
module 'mediapipe' has no attribute 'solutions'
```

修复方法：

```powershell
pip install mediapipe==0.10.14
```

---

## 当前阶段完成状态

- [x] 摄像头输入与实时可视化
- [x] 视频文件回放检测
- [x] YOLOv8n-face 人脸检测接入
- [x] MediaPipe FaceMesh 478 点关键点提取
- [x] 主驾驶员选择与 IoU 跟踪
- [x] 虹膜中心估计
- [x] JSONL 结构化结果输出
- [ ] 链路稳定性测试记录（进行中）
- [ ] 头姿估计（下一阶段）
- [ ] 视线分析（下一阶段）
- [ ] 疲劳检测 EAR / MAR（下一阶段）

---
