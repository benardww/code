import numpy as np
import pywt
from typing import List, Tuple


def wavelet_filter(signal_1d: np.ndarray, wavelet: str = 'db4', level: int = 5) -> np.ndarray:
    """
    Filter a 1-D EEG signal with Db4 wavelet decomposition.

    Decomposition at fs=256 Hz (level=5):
      coeffs[0] = cA5:  0– 4 Hz  (delta)     → KEEP
      coeffs[1] = cD5:  4– 8 Hz  (theta)     → KEEP
      coeffs[2] = cD4:  8–16 Hz  (alpha)     → KEEP
      coeffs[3] = cD3: 16–32 Hz  (beta)      → KEEP
      coeffs[4] = cD2: 32–64 Hz  (high-freq) → DISCARD
      coeffs[5] = cD1: 64–128 Hz (noise)     → DISCARD

    Reconstructed signal retains frequencies 0–32 Hz.
    """
    orig_len = len(signal_1d)
    coeffs = pywt.wavedec(signal_1d, wavelet=wavelet, level=level)
    coeffs[4] = np.zeros_like(coeffs[4])  # zero cD2
    coeffs[5] = np.zeros_like(coeffs[5])  # zero cD1
    reconstructed = pywt.waverec(coeffs, wavelet=wavelet)
    return reconstructed[:orig_len].astype(np.float32)


def filter_all_channels(eeg: np.ndarray, cfg) -> np.ndarray:
    """Apply wavelet_filter to every channel. Input/output: (n_channels, T)."""
    return np.stack(
        [wavelet_filter(eeg[ch], cfg.wavelet, cfg.wavelet_level) for ch in range(eeg.shape[0])]
    )


def sliding_window(
    eeg: np.ndarray,
    seizure_intervals: List[Tuple[int, int]],
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training windows using differential stride to balance classes.

    Seizure windows : 4 s, 50% overlap (stride = 512), only inside [s, e].
    Normal windows  : 4 s, no overlap (stride = 1024), only where sum==0.

    Args:
        eeg:               (n_channels, T) filtered signal.
        seizure_intervals: list of (start_sample, end_sample) pairs.
        cfg:               Config object.

    Returns:
        X: (N, n_channels, n_timesteps) float32
        y: (N,) int64
    """
    T = eeg.shape[1]
    win_len = cfg.n_timesteps  # 1024

    # Build per-sample seizure label array for fast lookup
    sample_labels = np.zeros(T, dtype=np.int8)
    for s, e in seizure_intervals:
        s = max(0, min(s, T))
        e = max(0, min(e, T))
        sample_labels[s:e] = 1

    X_seizure: List[np.ndarray] = []
    X_normal: List[np.ndarray] = []

    # Seizure windows: slide within each seizure interval
    for s, e in seizure_intervals:
        s = max(0, s)
        e = min(e, T)
        pos = s
        while pos + win_len <= e:
            X_seizure.append(eeg[:, pos:pos + win_len])
            pos += cfg.seizure_stride

    # Normal windows: slide across the full recording, skip any with seizure samples
    pos = 0
    while pos + win_len <= T:
        if sample_labels[pos:pos + win_len].sum() == 0:
            X_normal.append(eeg[:, pos:pos + win_len])
        pos += cfg.normal_stride

    if not X_seizure and not X_normal:
        return np.empty((0, cfg.n_channels, win_len), dtype=np.float32), np.empty(0, dtype=np.int64)

    X_all = np.array(X_seizure + X_normal, dtype=np.float32)
    y_all = np.array(
        [1] * len(X_seizure) + [0] * len(X_normal), dtype=np.int64
    )
    return X_all, y_all
