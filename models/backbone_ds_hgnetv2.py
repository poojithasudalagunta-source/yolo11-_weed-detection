import torch
import torch.nn as nn

class DSHGNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        # Example stages (adjust channels to your design)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.stem(x)
        p3 = self.stage1(x)   # stride 8
        p4 = self.stage2(p3)  # stride 16
        p5 = self.stage3(p4)  # stride 32
        return p3, p4, p5
