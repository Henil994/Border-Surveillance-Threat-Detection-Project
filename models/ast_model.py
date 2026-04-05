import timm
import torch.nn as nn


class ASTModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            img_size=160
        )

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.backbone(x)
        return self.head(x)