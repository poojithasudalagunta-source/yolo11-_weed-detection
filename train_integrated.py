from ultralytics import YOLO

# Load custom model via YAML (points to Python class)
model = YOLO("models/yolo11_ds_hgnet.yaml", task="detect")

model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640
)
