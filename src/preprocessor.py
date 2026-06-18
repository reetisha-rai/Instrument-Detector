"""
src/preprocessor.py — audio loading, mel-spectrogram computation,
caching, augmentation, and windowed segmentation.
All functions are identical to the Colab Cell 4 versions.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX,
    WINDOW_SECONDS, HOP_SECONDS, FRAMES_PER_WIN, CACHE_DIR,
)


# ── Core audio loading ────────────────────────────────────────────────────────

def load_audio(path) -> np.ndarray:
    """Load any audio file → mono numpy array at SAMPLE_RATE."""
    w, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return w


def audio_to_melspec(wav: np.ndarray) -> np.ndarray:
    """Waveform → log-mel spectrogram (N_MELS × T)."""
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def normalize_spec(s: np.ndarray) -> np.ndarray:
    """Normalize spectrogram to [0, 1]."""
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-8)


def spec_to_tensor(s: np.ndarray) -> torch.Tensor:
    """Pad/trim spectrogram to FRAMES_PER_WIN and return (1, N_MELS, T) tensor."""
    if s.shape[1] < FRAMES_PER_WIN:
        s = np.pad(s, ((0, 0), (0, FRAMES_PER_WIN - s.shape[1])))
    else:
        s = s[:, :FRAMES_PER_WIN]
    return torch.tensor(s, dtype=torch.float32).unsqueeze(0)


# ── Cache-aware random crop (used during training) ────────────────────────────

def random_crop(path) -> torch.Tensor:
    """
    Load a random 1-second crop from an audio file.
    Uses cached .npy files if available (much faster after precompute_cache()).
    Returns tensor of shape (1, N_MELS, FRAMES_PER_WIN).
    """
    key    = Path(path).stem
    cached = CACHE_DIR / f"{key}.npy"
    if cached.exists():
        s = np.load(str(cached))
        return torch.tensor(s, dtype=torch.float32).unsqueeze(0)

    # fallback: compute on the fly
    wav = load_audio(path)
    win = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(wav) <= win:
        wav   = np.pad(wav, (0, win - len(wav)))
        start = 0
    else:
        start = np.random.randint(0, len(wav) - win)
    return spec_to_tensor(normalize_spec(audio_to_melspec(wav[start:start + win])))


# ── Spectrogram augmentation (used during training) ──────────────────────────

def augment_spec(s: torch.Tensor) -> torch.Tensor:
    """
    SpecAugment-style augmentation: random frequency mask,
    time mask, and small Gaussian noise.
    Input/output shape: (1, N_MELS, FRAMES_PER_WIN).
    """
    s = s.clone()
    _, nm, nf = s.shape
    if np.random.rand() > 0.5:
        f  = np.random.randint(1, 15)
        fs = np.random.randint(0, nm - f)
        s[0, fs:fs + f, :] = 0.0
    if np.random.rand() > 0.5:
        t  = np.random.randint(1, 20)
        ts = np.random.randint(0, nf - t)
        s[0, :, ts:ts + t] = 0.0
    if np.random.rand() > 0.5:
        s = (s + torch.randn_like(s) * 0.005).clamp(0, 1)
    return s


# ── Windowed segmentation (used during inference) ────────────────────────────

def windowed_segments(path) -> Tuple[List[Tuple[float, torch.Tensor]], float]:
    """
    Slide a 1-second window across the full audio track.
    Returns:
        segments : list of (start_time_seconds, tensor (1, N_MELS, FRAMES_PER_WIN))
        duration : total track duration in seconds
    """
    wav  = load_audio(path)
    win  = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop  = int(HOP_SECONDS * SAMPLE_RATE)
    segs = []
    i    = 0
    while i + win <= len(wav):
        chunk = wav[i:i + win]
        segs.append((
            i / SAMPLE_RATE,
            spec_to_tensor(normalize_spec(audio_to_melspec(chunk))),
        ))
        i += hop
    return segs, len(wav) / SAMPLE_RATE


# ── One-time cache precomputation ─────────────────────────────────────────────

def precompute_cache(audio_paths: list, cache_dir: Path = CACHE_DIR):
    """
    Precompute and save spectrograms for all audio files.
    Run once before training to make each epoch much faster.

    Usage:
        from src.preprocessor import precompute_cache
        precompute_cache(all_audio_paths)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for path in audio_paths:
        key  = Path(path).stem
        dest = cache_dir / f"{key}.npy"
        if dest.exists():
            skipped += 1
            continue
        try:
            wav = load_audio(path)
            win = int(WINDOW_SECONDS * SAMPLE_RATE)
            if len(wav) < win:
                wav = np.pad(wav, (0, win - len(wav)))
            start = np.random.randint(0, max(1, len(wav) - win))
            chunk = wav[start:start + win]
            s     = normalize_spec(audio_to_melspec(chunk))
            if s.shape[1] < FRAMES_PER_WIN:
                s = np.pad(s, ((0, 0), (0, FRAMES_PER_WIN - s.shape[1])))
            else:
                s = s[:, :FRAMES_PER_WIN]
            np.save(dest, s.astype(np.float32))
        except Exception as e:
            print(f"  Skipping {path}: {e}")
    print(f"Cache complete. {len(audio_paths) - skipped} computed, {skipped} already existed.")
