import torch 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160
TARGET_LENGTH = 1024

BATCH_SIZE = 16
EPOCHS = 6   # 🔥 reduced
LR = 5e-5