🌱 YOLOv11-LightDetect-BiFPN Weed Detection

A lightweight multi-scale weed detection framework built on YOLOv11 for detecting multiple weed species in real agricultural environments.
The model improves both efficiency and detection performance by integrating a Light-Detect backbone (DSHGNetV2) and a BiFPN feature fusion neck.

📌 Overview

Weed detection in agricultural fields is challenging due to:

small object sizes

dense plant distribution

heavy occlusion

class imbalance

This project enhances YOLOv11 to better handle these conditions using:

Component	Purpose
Light-Detect (DSHGNetV2)	Efficient feature extraction
BiFPN	Multi-scale feature fusion
Progressive Fine-Tuning	Stable convergence
🧠 Model Architecture
Input Image
     ↓
DSHGNetV2 Backbone (Light-Detect)
     ↓
Multi-scale Features (P3, P4, P5)
     ↓
BiFPN Neck (Bidirectional Fusion)
     ↓
YOLO Detection Head
     ↓
Bounding Boxes + Weed Class
📂 Dataset

MH-Weed16: Indian Multiclass Annotated Weed Dataset

15 weed classes

YOLO format annotations

Field-captured real environment images

Strong class imbalance

Dense multi-object scenes

Images resized to 640×640 during training.

⚙️ Training Configuration
Parameter	Value
Framework	Ultralytics YOLO
Image Size	640×640
Epochs	Progressive fine-tuning (~40-50 effective epochs)
Hardware	CPU
Augmentation	Default YOLO augmentations (mosaic, flip, scale, color)
🚀 Training
Baseline YOLOv11
yolo detect train model=yolo11.yaml data=data.yaml epochs=40 imgsz=640
Proposed Model (LightDetect + BiFPN)
yolo detect train model=models/yolo11_ds_hgnet.yaml data=data.yaml epochs=40 imgsz=640
Continue Training From Checkpoint
yolo detect train model=runs/detect/trainX/weights/last.pt data=data.yaml epochs=10 imgsz=640 name=trainY
Resume Interrupted Training
yolo detect train resume=True model=runs/detect/trainX/weights/last.pt
📊 Evaluation Metrics

Precision

Recall

mAP@0.5

mAP@0.5:0.95

Confusion Matrix

PR Curve

F1-Confidence Curve

🔍 Key Observations

Improved small-object detection due to BiFPN

Better localization stability over training

Dataset imbalance affects classification confidence

Progressive fine-tuning improves convergence

🧪 Research Contribution

This work proposes a lightweight yet accurate agricultural detection framework by combining efficient feature extraction and multi-scale feature fusion within YOLOv11, suitable for resource-limited environments.

🛠 Requirements

Python 3.11

PyTorch

Ultralytics YOLO

OpenCV

NumPy

Matplotlib

Install dependencies:

pip install ultralytics opencv-python matplotlib numpy
