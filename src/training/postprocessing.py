import numpy as np
import torch
from typing import List, Tuple


def moving_average_filter(probs: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth a probability sequence with a symmetric moving-average kernel.

    window_size=5 at 1-second stride → 5-second smoothing window.
    'same' mode preserves sequence length; edges are padded with zeros.
    """
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(probs, kernel, mode='same').astype(np.float32)


def apply_threshold(smoothed_probs: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Convert smoothed probabilities to binary predictions."""
    return (smoothed_probs >= thresh).astype(np.int8)


def apply_collar(
    binary_preds: np.ndarray,
    stride_sec: float = 1.0,
    collar_sec: float = 5.0,
) -> np.ndarray:
    """
    Extend every detected seizure segment by collar_sec on both sides,
    then merge overlapping extended segments.

    Args:
        binary_preds: (N,) binary prediction sequence.
        stride_sec:   time (seconds) between consecutive elements.
        collar_sec:   extension applied to each side of every detection.

    Returns:
        (N,) extended binary array of the same length.
    """
    collar_steps = int(collar_sec / stride_sec)
    result = binary_preds.copy()

    padded = np.concatenate([[0], binary_preds.astype(np.int32), [0]])
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    for s, e in zip(starts, ends):
        lo = max(0, s - collar_steps)
        hi = min(len(result), e + collar_steps)
        result[lo:hi] = 1

    return result


def postprocess(
    probs: np.ndarray,
    maf_window: int = 5,
    threshold: float = 0.5,
    collar_sec: float = 5.0,
    stride_sec: float = 1.0,
) -> np.ndarray:
    """Full post-processing pipeline: MAF → threshold → collar."""
    smoothed = moving_average_filter(probs, maf_window)
    binary = apply_threshold(smoothed, threshold)
    return apply_collar(binary, stride_sec, collar_sec)


@torch.no_grad()
def infer_uniform_stride(
    model: torch.nn.Module,
    eeg_filtered: np.ndarray,
    win_len: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run the model over a filtered EEG recording with a uniform stride.

    Args:
        model:        trained SeizureDetector in eval mode.
        eeg_filtered: (n_channels, T) filtered signal (float32).
        win_len:      window length in samples (1024).
        stride:       step size in samples (256 = 1 s at 256 Hz).
        device:       torch device.

    Returns:
        probs: (n_windows,) float32 array of seizure probabilities.
    """
    model.eval()
    T = eeg_filtered.shape[1]
    probs: List[float] = []

    pos = 0
    while pos + win_len <= T:
        window = eeg_filtered[:, pos:pos + win_len].copy()
        # Per-channel z-score normalisation (mirrors EEGDataset.__getitem__)
        mean = window.mean(axis=-1, keepdims=True)
        std = window.std(axis=-1, keepdims=True) + 1e-8
        window = (window - mean) / std

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        probs.append(prob)
        pos += stride

    return np.array(probs, dtype=np.float32)
