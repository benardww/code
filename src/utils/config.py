from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data
    fs: int = 256
    win_sec: int = 4
    n_channels: int = 18
    n_timesteps: int = 1024  # win_sec * fs

    # Excluded subjects
    excluded_subjects: List[str] = field(default_factory=lambda: ['chb06', 'chb16'])

    # Wavelet
    wavelet: str = 'db4'
    wavelet_level: int = 5

    # Sliding window strides (in samples)
    seizure_stride: int = 512   # 50% overlap for seizure windows
    normal_stride: int = 1024   # no overlap for normal windows
    infer_stride: int = 256     # 1s uniform stride for test inference

    # Patch Embedding
    patch_size: int = 16
    patch_stride: int = 8       # num_patches = floor((1024-16)/8)+1 = 127

    # ModernTCN
    d_model: int = 128
    kernel_size: int = 51
    num_tcn_blocks: int = 3
    expansion: int = 4
    tcn_dropout: float = 0.1

    # SE-Net
    se_reduction: int = 16

    # BiLSTM
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3

    # Training
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    grad_clip: float = 5.0
    early_stop_patience: int = 15
    num_workers: int = 4

    # Post-processing
    maf_window: int = 5       # smoothing over 5 windows = 5s at 1s stride
    threshold: float = 0.5
    collar_sec: float = 5.0   # extend each detection region by ±5s

    # Paths (relative to project root)
    data_raw_dir: str = '/root/autodl-tmp/chb-mit-scalp-eeg-database-1.0.0'
    data_processed_dir: str = '/root/autodl-tmp/data/processed'

    # Splits (by EDF file count)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test gets the remainder (~0.15)

    # Other
    device: str = 'cuda'
    seed: int = 42
