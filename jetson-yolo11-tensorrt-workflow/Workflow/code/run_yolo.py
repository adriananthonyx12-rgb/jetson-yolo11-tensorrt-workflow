from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.engine", task="detect")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read camera frame")
        break

    results = model.predict(frame, device="cuda", imgsz=320, verbose=False, task="detect")
    annotated = results[0].plot()

    cv2.imshow("YOLO11 TensorRT", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
