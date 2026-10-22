# YOLOv8 Real-time Detection
基于 YOLOv8 和 OpenCV 的实时目标检测与保存系统。

## 功能
- 实时读取摄像头视频流进行目标检测
- 类别过滤（仅显示 person）
- 实时画框与 FPS 显示
- 将检测结果保存为 mp4 视频

## 踩坑记录
1. **张量维度不匹配**：遇到 `RuntimeError: shape '[-1, 1014]' is invalid`。排查发现输入尺寸变更为 32×32 后，特征图变为 15×15。改用 `x.flatten(1)` 替代硬编码 `view`。
2. **视频保存失败/损坏**：`VideoWriter` 宽高必须与 `frame.shape` 严格一致（宽在前，高在后），且必须调用 `release()` 释放文件头。
3. **环境兼容性**：VS Code 内置播放器不支持 `mp4v` 编码，改用系统播放器验证通过。

## 运行方式
`pip install -r requirements.txt`
`python yolo_test.py`

虽然大部分过程是在ai的帮助下进行调试，但是还算是又成就感吧。对这方面的知识又了解了一点。
