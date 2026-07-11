from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("assets/videos/9sec.mp4")
cv2.namedWindow("YOLO Video Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Video Detection", 1200, 700)
# Read video frame by frame
while True:

    ret, frame = cap.read()

    if not ret:
        break

    
    results = model(frame, verbose=False)
    print(results[0].boxes)
    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("YOLO Video Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()