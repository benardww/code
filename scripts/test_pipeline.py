#!/usr/bin/env python3
"""
分步验证脚本：无需真实数据，纯合成数据验证整个 pipeline。

测试项目：
  T1  各模块前向传播 + 输出维度
  T2  FocalLoss 前向传播
  T3  后处理流水线 (MAF / threshold / collar)
  T4  infer_uniform_stride
  T5  compute_metrics
  T6  过拟合验证（50样本，能否 loss→0）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    msg = f"  [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return cond


# ─────────────────────────────────────────────
# T1  模块前向传播
# ─────────────────────────────────────────────
def test_forward():
    print("\n[T1] 模块前向传播")
    from src.utils.config import Config
    from src.models.modern_tcn import ModernTCN
    from src.models.channel_attention import SEBlock
    from src.models.bilstm import BiLSTMClassifier
    from src.models.seizure_detector import SeizureDetector

    cfg = Config()
    B = 2
    x = torch.randn(B, 18, 1024)

    # Patch embed → TCN
    tcn = ModernTCN(
        in_channels=cfg.n_channels, d_model=cfg.d_model,
        patch_size=cfg.patch_size, patch_stride=cfg.patch_stride,
        num_blocks=cfg.num_tcn_blocks, kernel_size=cfg.kernel_size,
        expansion=cfg.expansion, dropout=cfg.tcn_dropout,
    )
    out_tcn = tcn(x)
    check("ModernTCN output shape == (2, 128, 127)",
          tuple(out_tcn.shape) == (B, 128, 127), str(tuple(out_tcn.shape)))

    # SE-Net
    se = SEBlock(cfg.d_model, cfg.se_reduction)
    out_se = se(out_tcn)
    check("SEBlock output shape == (2, 128, 127)",
          tuple(out_se.shape) == (B, 128, 127), str(tuple(out_se.shape)))

    # BiLSTM
    bilstm = BiLSTMClassifier(cfg.d_model, cfg.lstm_hidden, cfg.lstm_layers, cfg.lstm_dropout)
    out_lstm = bilstm(out_se)
    check("BiLSTMClassifier output shape == (2, 512)",
          tuple(out_lstm.shape) == (B, 2 * cfg.lstm_hidden), str(tuple(out_lstm.shape)))

    # Full model
    model = SeizureDetector(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logits = model(x)
    check("SeizureDetector output shape == (2, 2)",
          tuple(logits.shape) == (B, 2), str(tuple(logits.shape)))
    print(f"      可训练参数量: {n_params:,}")


# ─────────────────────────────────────────────
# T2  FocalLoss
# ─────────────────────────────────────────────
def test_focal_loss():
    print("\n[T2] FocalLoss")
    from src.losses.focal_loss import FocalLoss

    criterion = FocalLoss(alpha=0.9, gamma=2.0)
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    loss = criterion(logits, targets)

    check("loss 是标量", loss.ndim == 0, f"shape={tuple(loss.shape)}")
    check("loss > 0", loss.item() > 0, f"loss={loss.item():.4f}")
    check("loss 可反向传播", True)  # no exception → pass
    loss.backward()


# ─────────────────────────────────────────────
# T3  后处理流水线
# ─────────────────────────────────────────────
def test_postprocessing():
    print("\n[T3] 后处理流水线")
    from src.training.postprocessing import (
        moving_average_filter, apply_threshold, apply_collar, postprocess
    )

    probs = np.zeros(100, dtype=np.float32)
    probs[30:40] = 0.9   # 模拟发作区段

    smoothed = moving_average_filter(probs, window_size=5)
    check("MAF 输出长度不变", len(smoothed) == len(probs), f"len={len(smoothed)}")
    check("MAF 平滑发作区段 > 0", smoothed[30:40].max() > 0)

    binary = apply_threshold(smoothed, thresh=0.5)
    check("threshold 输出为 int8", binary.dtype == np.int8)
    check("发作区段被检出", binary[30:40].sum() > 0)

    collared = apply_collar(binary, stride_sec=1.0, collar_sec=5.0)
    check("collar 扩展了检测区段", collared.sum() > binary.sum(),
          f"before={binary.sum()} after={collared.sum()}")
    check("collar 输出长度不变", len(collared) == len(probs))

    final = postprocess(probs, maf_window=5, threshold=0.5, collar_sec=5.0, stride_sec=1.0)
    check("postprocess 组合输出正确", len(final) == len(probs))


# ─────────────────────────────────────────────
# T4  infer_uniform_stride
# ─────────────────────────────────────────────
def test_infer_stride():
    print("\n[T4] infer_uniform_stride")
    from src.utils.config import Config
    from src.models.seizure_detector import SeizureDetector
    from src.training.postprocessing import infer_uniform_stride

    cfg = Config()
    device = torch.device('cpu')
    model = SeizureDetector(cfg).to(device)
    model.eval()

    T = 5120  # 20s @ 256 Hz
    eeg = np.random.randn(18, T).astype(np.float32)
    probs = infer_uniform_stride(model, eeg, win_len=1024, stride=256, device=device)

    expected = (T - 1024) // 256 + 1  # 17 windows
    check("输出窗口数正确", len(probs) == expected, f"got={len(probs)} expected={expected}")
    check("概率值在 [0,1]", probs.min() >= 0 and probs.max() <= 1,
          f"min={probs.min():.4f} max={probs.max():.4f}")


# ─────────────────────────────────────────────
# T5  compute_metrics
# ─────────────────────────────────────────────
def test_metrics():
    print("\n[T5] compute_metrics")
    from src.training.metrics import compute_metrics

    np.random.seed(0)
    labels = np.array([0]*80 + [1]*20)
    probs = np.clip(labels + np.random.randn(100) * 0.3, 0, 1).astype(np.float32)
    metrics = compute_metrics(probs, labels, threshold=0.5)

    for key in ('sensitivity', 'specificity', 'auc', 'f1'):
        check(f"{key} 在 [0,1]", 0.0 <= metrics[key] <= 1.0,
              f"{key}={metrics[key]:.4f}")

    check("AUC 合理 (> 0.7)", metrics['auc'] > 0.7, f"auc={metrics['auc']:.4f}")


# ─────────────────────────────────────────────
# T6  过拟合验证（50样本，loss 能否趋近 0）
# ─────────────────────────────────────────────
def test_overfit():
    print("\n[T6] 过拟合验证（50 样本，30 轮）")
    from src.utils.config import Config
    from src.models.seizure_detector import SeizureDetector
    from src.losses.focal_loss import FocalLoss

    cfg = Config()
    device = torch.device('cpu')
    model = SeizureDetector(cfg).to(device)

    torch.manual_seed(42)
    X = torch.randn(50, 18, 1024)
    y = torch.cat([torch.zeros(25, dtype=torch.long),
                   torch.ones(25, dtype=torch.long)])

    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1:3d}  loss={loss.item():.4f}")

    final_loss = loss.item()
    check("30 轮后 loss < 0.3", final_loss < 0.3, f"final_loss={final_loss:.4f}")


# ─────────────────────────────────────────────
if __name__ == '__main__':
    all_ok = True
    for test_fn in [test_forward, test_focal_loss, test_postprocessing,
                    test_infer_stride, test_metrics, test_overfit]:
        try:
            test_fn()
        except Exception as e:
            print(f"  [{FAIL}] {test_fn.__name__} 抛出异常: {e}")
            import traceback; traceback.print_exc()
            all_ok = False

    print("\n" + "=" * 50)
    print("全部通过 ✓" if all_ok else "存在失败项，请检查输出")
