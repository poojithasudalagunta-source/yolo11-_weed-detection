from ultralytics import YOLO

# Load YOLO11 Nano model
model = YOLO("C:/yolo11_pro/yolo11n.pt")  

# Train on CPU 
model.train(
    data="C:/yolo11_pro/data.yaml",
    epochs=10,       # small for testing
    imgsz=320,       # CPU-friendly
    batch=1,
    device='cpu'
)
