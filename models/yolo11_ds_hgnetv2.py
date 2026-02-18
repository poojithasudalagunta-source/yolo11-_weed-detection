import torch
import torch.nn as nn
from models.backbone_ds_hgnetv2 import DSHGNetV2
from models.common import Detect, Conv


# -----------------------------
# Lightweight BiFPN Module
# -----------------------------
class BiFPN(nn.Module):
    def __init__(self, c3, c4, c5):
        super().__init__()
        self.eps = 1e-4

        # Channel alignment
        self.p3_conv = Conv(c3, 128, 1, 1)
        self.p4_conv = Conv(c4, 256, 1, 1)
        self.p5_conv = Conv(c5, 512, 1, 1)

        # Learnable fusion weights
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(2))

        # Output convolutions
        self.out3 = Conv(128, 128, 3, 1)
        self.out4 = Conv(256, 256, 3, 1)
        self.out5 = Conv(512, 512, 3, 1)

    def forward(self, p3, p4, p5):
        # Align channels
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)

        # Normalize weights
        w1 = torch.relu(self.w1)
        w1 = w1 / (torch.sum(w1) + self.eps)

        # Top-down fusion
        p4_td = w1[0] * p4 + w1[1] * nn.functional.interpolate(p5, scale_factor=2, mode="nearest")
        p3_td = w1[0] * p3 + w1[1] * nn.functional.interpolate(p4_td, scale_factor=2, mode="nearest")

        # Normalize weights
        w2 = torch.relu(self.w2)
        w2 = w2 / (torch.sum(w2) + self.eps)

        # Bottom-up fusion
        p4_out = w2[0] * p4_td + w2[1] * nn.functional.max_pool2d(p3_td, 2)
        p5_out = w2[0] * p5 + w2[1] * nn.functional.max_pool2d(p4_out, 2)

        return (
            self.out3(p3_td),
            self.out4(p4_out),
            self.out5(p5_out),
        )


# -----------------------------
# YOLOv11 + Light Detect + BiFPN
# -----------------------------
class YOLO11_DSHGNetV2(nn.Module):
    def __init__(self, nc=15):
        super().__init__()

        # Backbone (Light Detect)
        self.backbone = DSHGNetV2()

        # Neck (BiFPN)
        self.bifpn = BiFPN(128, 256, 512)

        # Head (YOLO Detect)
        self.detect = Detect(nc, ch=[128, 256, 512])

    def forward(self, x):
        # Backbone features
        p3, p4, p5 = self.backbone(x)

        # BiFPN feature fusion
        p3, p4, p5 = self.bifpn(p3, p4, p5)

        # Detection
        return self.detect([p3, p4, p5])
