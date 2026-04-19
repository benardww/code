import torch
import torch.nn as nn

from .modern_tcn import ModernTCN
from .channel_attention import SEBlock
from .bilstm import BiLSTMClassifier


class SeizureDetector(nn.Module):
    """
    Full detection model: ModernTCN → SE-Net → BiLSTM → classifier.

    Input : (B, 18, 1024)
    Output: (B, 2)  — logits for [normal, seizure]

    Dimension trace (default config):
      (B, 18, 1024) → patch embed → (B, 128, 127)
                    → ModernTCN ×3 → (B, 128, 127)
                    → SE-Net       → (B, 128, 127)
                    → BiLSTM       → (B, 512)
                    → classifier   → (B, 2)
    """

    def __init__(self, cfg):
        super().__init__()
        self.tcn = ModernTCN(
            in_channels=cfg.n_channels,
            d_model=cfg.d_model,
            patch_size=cfg.patch_size,
            patch_stride=cfg.patch_stride,
            num_blocks=cfg.num_tcn_blocks,
            kernel_size=cfg.kernel_size,
            expansion=cfg.expansion,
            dropout=cfg.tcn_dropout,
        )
        self.se = SEBlock(cfg.d_model, cfg.se_reduction)
        self.bilstm = BiLSTMClassifier(
            input_size=cfg.d_model,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.lstm_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)           # (B, 128, 127)
        x = self.se(x)            # (B, 128, 127)
        x = self.bilstm(x)        # (B, 512)
        return self.classifier(x) # (B, 2)
