from ultralytics import YOLO
import cv2
import time

print("1. 加载模型...")
model = YOLO("yolov8n.pt")
print("设备:", next(model.model.parameters()).device)

print("2. 打开摄像头...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("摄像头打不开！")
    exit()

cv2.namedWindow("YOLO Video", cv2.WINDOW_NORMAL)

prev_time = time.time()
frame_count = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter('output.mp4',fourcc,20.0,(640,480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理：imgsz=320 是最有效的提速手段
    results = model(frame, imgsz=320, verbose=False)

    # 画框
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if cls_id!=0:
            continue
        label = f"{model.names[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # FPS：前10帧不显示，避开预热
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    frame_count += 1
    if frame_count > 10:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("YOLO Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
h, w = frame.shape[:2]
print(f"宽={w}, 高={h}")
cap.release()
out.release()
cv2.destroyAllWindows()
