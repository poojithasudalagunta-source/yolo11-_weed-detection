from ultralytics import YOLO
from custom_models.yolo11_custom import YOLO11_Custom

# 1. Create empty YOLO engine
yolo = YOLO(model=None, task="detect")

# 2. Attach your custom model
yolo.model = YOLO11_Custom(num_classes=15)

# 3. Train
yolo.train(
    data="data.yaml",
    epochs=50,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=4,
    name="yolo11_ds_hgnet_bifpn"
)
