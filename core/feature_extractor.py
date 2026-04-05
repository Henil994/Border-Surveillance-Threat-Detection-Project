import torch
import torchaudio
import torch.nn.functional as F
import config

# 🔥 CLEAN WARNINGS (optional but useful)
import warnings
warnings.filterwarnings("ignore")

mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS
)

MIN_AUDIO_LEN = config.N_FFT  # critical for STFT stability


# =====================================================
# 🔹 SINGLE FEATURE (used in some places)
# =====================================================

def extract_features(path, train_mode=True):

    wave, sr = torchaudio.load(path)

    # mono
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # resample
    if sr != config.SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, config.SAMPLE_RATE)

    # 🔥 normalize waveform (removes recording bias)
    wave = wave / (wave.abs().max() + 1e-6)

    # 🔥 ensure minimum length
    if wave.shape[-1] < MIN_AUDIO_LEN:
        wave = F.pad(wave, (0, MIN_AUDIO_LEN - wave.shape[-1]))

    spec = mel(wave)
    spec = torch.log1p(spec)

    # normalization
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)

    # pad/trim
    if spec.shape[-1] > config.TARGET_LENGTH:
        spec = spec[:, :, :config.TARGET_LENGTH]
    else:
        spec = F.pad(spec, (0, config.TARGET_LENGTH - spec.shape[-1]))

    # resize
    spec = F.interpolate(
        spec.unsqueeze(0),
        size=(160, 160),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    spec = spec[:, :160, :160].contiguous()

    return spec


# =====================================================
# 🔹 TEMPORAL SEQUENCE (MAIN IMPORTANT FUNCTION)
# =====================================================

def extract_sequence(path, segments=2):

    wave, sr = torchaudio.load(path)

    # mono
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # resample
    if sr != config.SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, config.SAMPLE_RATE)

    # 🔥 normalize waveform
    wave = wave / (wave.abs().max() + 1e-6)

    # 🔥 ensure minimum length (VERY IMPORTANT)
    if wave.shape[-1] < MIN_AUDIO_LEN:
        wave = F.pad(wave, (0, MIN_AUDIO_LEN - wave.shape[-1]))

    total_len = wave.shape[-1]
    seg_len = max(total_len // segments, MIN_AUDIO_LEN)

    specs = []

    for i in range(segments):

        start = i * seg_len
        end = start + seg_len

        seg = wave[:, start:end]

        # 🔥 pad EACH segment if too small
        if seg.shape[-1] < MIN_AUDIO_LEN:
            seg = F.pad(seg, (0, MIN_AUDIO_LEN - seg.shape[-1]))

        spec = mel(seg)
        spec = torch.log1p(spec)

        # 🔥 GLOBAL normalization (not per segment mean/std)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        spec = F.interpolate(
            spec.unsqueeze(0),
            size=(160, 160),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        specs.append(spec)

    return torch.stack(specs)  # (T, 1, 160, 160)