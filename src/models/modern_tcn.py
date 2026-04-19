import torch
import torch.nn as nn


class ModernTCNBlock(nn.Module):
    """
    One ModernTCN block:
      DWConv (large-kernel depthwise) → InstanceNorm → FFN (two pointwise convs) → residual.
    """

    def __init__(self, d_model: int, kernel_size: int = 51, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        padding = kernel_size // 2

        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding, groups=d_model)
        self.norm = nn.InstanceNorm1d(d_model, affine=True)
        self.pw_conv1 = nn.Conv1d(d_model, d_model * expansion, kernel_size=1)
        self.pw_conv2 = nn.Conv1d(d_model * expansion, d_model, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, T)
        residual = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pw_conv2(x)
        return x + residual


class ModernTCN(nn.Module):
    """
    ModernTCN: patch embedding + N ModernTCNBlocks.

    Input : (B, in_channels, T)
    Output: (B, d_model, T') where T' = floor((T - patch_size) / patch_stride) + 1
    """

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int,
        patch_stride: int,
        num_blocks: int,
        kernel_size: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_stride)
        self.embed_norm = nn.InstanceNorm1d(d_model, affine=True)
        self.blocks = nn.ModuleList(
            [ModernTCNBlock(d_model, kernel_size, expansion, dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)   # (B, d_model, T')
        x = self.embed_norm(x)
        for block in self.blocks:
            x = block(x)
        return x
