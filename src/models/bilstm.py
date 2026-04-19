import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM that maps a feature sequence to a fixed-size vector.

    Input : (B, C, T)  — channel-first from the attention block.
    Output: (B, 2 * hidden_size)  — last-layer forward + backward hidden states.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C) for LSTM
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, B, hidden_size)
        # Last layer: h_n[-2] = forward, h_n[-1] = backward
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2 * hidden_size)
