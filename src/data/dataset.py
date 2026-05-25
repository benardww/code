import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    """PyTorch Dataset wrapping pre-windowed EEG arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()  # (n_channels, n_timesteps)
        # Per-channel z-score normalisation
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-8
        x = (x - mean) / std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def _make_loaders(train_X, train_y, val_X, val_y, batch_size, num_workers):
    train_ds = EEGDataset(train_X, train_y)
    val_ds = EEGDataset(val_X, val_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_loaders_from_arrays(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray, val_y: np.ndarray,
    batch_size: int,
    num_workers: int = 4,
):
    """从内存数组构建 DataLoader，不依赖磁盘文件。"""
    return _make_loaders(train_X, train_y, val_X, val_y, batch_size, num_workers)


def get_loaders(
    subject_dir: str,
    batch_size: int,
    num_workers: int = 4,
):
    """从磁盘 .npy 文件构建 DataLoader（--mode preprocess 缓存后使用）。"""
    train_X = np.load(os.path.join(subject_dir, 'train_X.npy'))
    train_y = np.load(os.path.join(subject_dir, 'train_y.npy'))
    val_X = np.load(os.path.join(subject_dir, 'val_X.npy'))
    val_y = np.load(os.path.join(subject_dir, 'val_y.npy'))
    return _make_loaders(train_X, train_y, val_X, val_y, batch_size, num_workers)

