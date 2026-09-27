from ultralytics import YOLO
import cv2

print("1. 正在加载模型（首次运行会自动下载约6MB的权重）...")
# 加载预训练模型（ yolov8n 是最轻量级的，适合新手跑通）
model = YOLO("yolov8n.pt")

print("2. 正在进行推理...")
# 读取 test.jpg 进行推理
results = model("test.jpg")

print("3. 正在画框...")
# 遍历结果，画框并显示
for result in results:
    annotated_frame = result.plot() # 在图像上绘制检测框
    cv2.imshow("YOLOv8 Test", annotated_frame)
    
    print("4. 检测完成！按键盘任意键关闭窗口。")
    cv2.waitKey(0) # 等待用户按任意键
    cv2.destroyAllWindows()