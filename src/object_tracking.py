from ultralytics import YOLO
import cv2
import os
import time 

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("assets/videos/9sec.mp4")
#video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)
#CREATE VIDEO WRITER 



os.makedirs("outputs", exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    "outputs/tracked_output.mp4",
    fourcc,
    fps,
    (frame_width, frame_height)
)

print("VideoWriter opened:", out.isOpened())
if not out.isOpened():
    print("Error: VideoWriter could not open!")
else:
    print("VideoWriter opened successfully!")
cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Tracking",1200,700)

unique_car_ids = set()

allowed_classes = ["car", "person", "bus", "truck","motorcycle"]

prev_time = time.time()

while True:
    ret , frame = cap.read()

    if not ret:
        break
    results= model.track(frame, persist= True, verbose= False)
    object_count = {}
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        if class_name == "car" and box.id is not None:
            unique_car_ids.add(int(box.id))
        if class_name in allowed_classes:
            if class_name in object_count:
                object_count[class_name] += 1
            else:
                object_count[class_name] = 1

    annotated_frame = results[0].plot()

    current_time = time.time()
    fps_display = 1/ (current_time - prev_time)
    prev_time = current_time

    y = 30
    for name, count in object_count.items():
        display_names = {
            "car": "Cars",
            "person": "Persons",
            "bus": "Buses",
            "truck": "Trucks",
            "motorcycle": "Motorcycles"
        }

        text = f"{display_names[name]}: {count}"
        cv2.putText(
            annotated_frame,
            f"Unique Cars: {len(unique_car_ids)}",
            
            (20, y+20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        y += 40
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps_display)}",
            (900, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    out.write(annotated_frame)

    cv2.imshow("Object Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print("Video saved successfully!")
cap.release()
out.release()
cv2.destroyAllWindows()