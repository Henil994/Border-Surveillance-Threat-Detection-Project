import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models.ast_temporal_model import ASTTemporalModel
from models.stage2_model import Stage2Model
from core.feature_extractor import extract_sequence
import config


CLASS_NAMES = {
    0: "Human Intrusion",
    1: "Drone Threat",
    2: "Animal Threat",
    3: "Environmental Noise"
}


class ThreatPredictor:

    def __init__(self):

        self.device = config.DEVICE

        self.model = ASTTemporalModel().to(self.device)
        self.model.load_state_dict(
            torch.load("models/ast_model.pth", map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.stage2_model = Stage2Model().to(self.device)

        try:
            self.stage2_model.load_state_dict(
                torch.load("models/stage2_model.pth", map_location=self.device, weights_only=True)
            )
            self.stage2_model.eval()
            self.use_stage2 = True
        except:
            self.use_stage2 = False

    # ---------------------------------------------------

    def predict_from_audio(self, path):

        spec_seq = extract_sequence(path, segments=2).to(self.device)

        with torch.no_grad():
            out = self.model(spec_seq.unsqueeze(0))
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))

        # 🔥 ONLY refine Noise (correct use of stage2)
        if self.use_stage2 and pred == 3:

            spec_single = spec_seq[0].unsqueeze(0)

            with torch.no_grad():
                out2 = self.stage2_model(spec_single)
                pred2 = out2.argmax(1).item()

            pred = 1 if pred2 == 0 else 3

        label = CLASS_NAMES[pred]
        confidence = float(np.max(probs) * 100)

        probs_dict = {
            "Human Intrusion": float(probs[0] * 100),
            "Drone Threat": float(probs[1] * 100),
            "Animal Threat": float(probs[2] * 100),
            "Environmental Noise": float(probs[3] * 100)
        }

        return label, confidence, probs_dict