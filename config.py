import torch

# DEVICE
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# AUDIO
SAMPLE_RATE = 16000

# CLASS LABELS
CLASS_NAMES = [
    "Human Intrusion",
    "Drone Threat",
    "Animal Threat",
    "Environmental Noise"
]