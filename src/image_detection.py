from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("assets/images/cat.jpg")
results[0].show()
results[0].save(filename="outputs/cat_result.jpg")
