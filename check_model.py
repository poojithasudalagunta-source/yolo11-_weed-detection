import torch
from models.yolo11_ds_hgnetv2 import YOLO11_DS_HGNetV2

model = YOLO11_DS_HGNetV2(nc=15, ch=3)

x = torch.randn(1, 3, 640, 640)
outputs = model(x)

print("Model ran successfully!")
print([o.shape for o in outputs])
