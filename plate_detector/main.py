from ultralytics import YOLO

model = YOLO("yolo11n.yaml")

results = model.train(data="config.yaml", epochs=100)