#!/usr/bin/env python3
"""Train the SeizureDetector on one CHB-MIT subject."""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch

from src.utils.config import Config
from src.data.dataset import get_loaders
from src.models.seizure_detector import SeizureDetector
from src.losses.focal_loss import FocalLoss
from src.training.trainer import Trainer


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_subject(subject_name: str, cfg: Config, experiments_dir: str = 'experiments') -> str:
    set_seed(cfg.seed)

    subject_dir = os.path.join(cfg.data_processed_dir, subject_name)
    if not os.path.isdir(subject_dir):
        raise FileNotFoundError(
            f"Processed data not found at {subject_dir}. "
            "Run scripts/preprocess_all.py first."
        )

    # Load per-subject focal_alpha if available
    stats_path = os.path.join(subject_dir, 'stats.json')
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        cfg.focal_alpha = stats.get('focal_alpha', cfg.focal_alpha)
        print(f"Using focal_alpha={cfg.focal_alpha:.4f} from stats.json")

    save_dir = os.path.join(experiments_dir, subject_name)
    print(f"\nTraining on subject: {subject_name}  |  device: {cfg.device}")

    train_loader, val_loader = get_loaders(subject_dir, cfg.batch_size, cfg.num_workers)
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)}"
    )

    model = SeizureDetector(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    trainer = Trainer(model, criterion, cfg, save_dir)
    best_ckpt = trainer.fit(train_loader, val_loader)

    print(f"Best checkpoint saved: {best_ckpt}")
    return best_ckpt


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train seizure detector on a CHB-MIT subject')
    parser.add_argument('--subject', type=str, required=True, help='e.g. chb01')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    args = parser.parse_args()

    cfg = Config(device=args.device)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    train_subject(args.subject, cfg)
