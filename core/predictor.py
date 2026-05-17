import torch
import torch.nn.functional as F

from models.ast_model import ASTModel
from config import DEVICE, CLASS_NAMES

# 🔥 USE EXISTING EXTRACTOR
from core.precompute_features import extract_ast


class ThreatPredictor:

    def __init__(self):

        self.model = ASTModel().to(DEVICE)

        self.model.load_state_dict(
            torch.load(
                "best_model.pth",
                map_location=DEVICE,
                weights_only=True
            ),
            strict=False
        )

        self.model.eval()

    def predict(self, audio_path):

        # =========================
        # FEATURE EXTRACTION
        # =========================
        features = extract_ast(audio_path)

        features = features.unsqueeze(0).to(DEVICE)

        # =========================
        # MODEL PREDICTION
        # =========================
        with torch.no_grad():

            outputs = self.model(features)

            # 🔥 TEMPERATURE SCALING
            probs = F.softmax(outputs / 3.0, dim=1)

            confidence, pred = torch.max(probs, dim=1)

        pred_idx = pred.item()

        confidence = confidence.item() * 100

        # =========================
        # CLASS PROBABILITIES
        # =========================
        prob_dict = {
            CLASS_NAMES[i]: round(
                probs[0][i].item() * 100,
                2
            )
            for i in range(len(CLASS_NAMES))
        }

        return (
            CLASS_NAMES[pred_idx],
            round(confidence, 2),
            prob_dict
        )