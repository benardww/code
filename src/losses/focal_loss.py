import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary focal loss for seizure (class 1) vs normal (class 0).

    alpha: weight for the positive (seizure) class.
    gamma: focusing exponent; higher values down-weight easy examples.
    """

    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 2), targets: (B,) in {0, 1}
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=1)
        p_t = probs[torch.arange(len(targets), device=targets.device), targets]

        # Per-class alpha weighting
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(p_t, self.alpha),
            torch.full_like(p_t, 1.0 - self.alpha),
        )

        loss = alpha_t * (1.0 - p_t) ** self.gamma * ce_loss
        return loss.mean()
