# face_detection

把你的 YOLO 人脸检测权重放到这个目录下，例如：

- `yolo_face.pt`
- `yolov8-face.pt`

当前项目默认会尝试读取 `models/face_detection/yolo_face.pt`。
如果该文件不存在，程序会自动回退到 MediaPipe 人脸检测，以便先把整条链路跑通。
