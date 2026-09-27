from ultralytics import YOLO
import cv2

print("1. 加载模型...")
model = YOLO("yolov8n.pt")

print("2. 推理图片...")
# 这一步得到的结果 results 是一堆张量数据，不是画好的图
results = model("test.jpg")

print("3. 自己动手画框...")
# ① 用 OpenCV 重新读取原图（我们要在干净的底图上画）
img = cv2.imread("test.jpg")

# ② 提取第一张图的结果
result = results[0]
boxes = result.boxes # 这里面装着所有的检测框、置信度、类别

# ③ 遍历每一个检测框
for box in boxes:
    # 提取坐标：神经网络吐出的坐标是浮点数（如 102.3），画图需要整数，用 map(int, ...) 转换
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 提取置信度（0.81之类的）
    conf = float(box.conf[0])
    
    # 提取类别ID（比如 0 代表人，5 代表公交车）
    cls_id = int(box.cls[0])
    
    # 把数字ID翻译成人类能看懂的文字（例如把 5 翻译成 "bus"）
    class_name = model.names[cls_id]

    # 【画框】调用 OpenCV 的画笔：
    # 参数：原图, 左上角坐标, 右下角坐标, 颜色(紫色BGR), 线宽
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # 【写字】在框的左上角写标签：
    label = f"{class_name} {conf:.2f}" 
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 0, 255), 2)

print("4. 展示结果！按任意键关闭。")
# 此时 img 已经被我们用 OpenCV 画满了框和字
cv2.imshow("My Own Drawing", img)
cv2.waitKey(0)
cv2.destroyAllWindows()