import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Input/output: (B, C, T).  The block recalibrates channel-wise feature responses.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        s = self.gap(x).squeeze(-1)        # (B, C)
        s = self.relu(self.fc1(s))          # (B, mid)
        s = self.sigmoid(self.fc2(s))       # (B, C)
        return x * s.unsqueeze(-1)          # (B, C, T)
