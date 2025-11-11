from ultralytics import YOLO

model = YOLO("yolov8n")
results = model.train(data ="config.yaml")


