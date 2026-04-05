import timm
import torch
import torch.nn as nn


class ASTTemporalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, padding=1),
            nn.ReLU()
        )

        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            img_size=160
        )

        # 🔥 FREEZE VIT (IMPORTANT)
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 1, 160, 160)

        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        x = self.cnn(x)
        x = self.backbone(x)

        x = x.view(B, T, -1)

        _, (h, _) = self.lstm(x)

        return self.head(h[-1])