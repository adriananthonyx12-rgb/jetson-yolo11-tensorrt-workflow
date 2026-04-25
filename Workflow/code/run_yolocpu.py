from ultralytics import YOLO
import cv2

# Load PyTorch model (NOT .engine)
model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera failed")
        break

    # CPU inference
    results = model.predict(frame, device="cpu", imgsz=320, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("YOLO11 CPU", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
