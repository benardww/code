import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .metrics import compute_metrics


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, cfg, save_dir: str):
        self.cfg = cfg
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=1e-6
        )
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        self.early_stopper = EarlyStopping(patience=cfg.early_stop_patience)

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp:
                with autocast(device_type='cuda'):
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def _validate(self, loader) -> dict:
        self.model.eval()
        all_probs, all_labels = [], []
        for X, y in loader:
            X = X.to(self.device)
            logits = self.model(X)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())
        return compute_metrics(np.array(all_probs), np.array(all_labels))

    def fit(self, train_loader, val_loader) -> str:
        best_auc = -np.inf
        best_ckpt = os.path.join(self.save_dir, 'best_model.pth')

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            self.scheduler.step()

            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"sens={val_metrics['sensitivity']:.4f} | "
                f"spec={val_metrics['specificity']:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                torch.save(self.model.state_dict(), best_ckpt)

            if self.early_stopper.step(val_metrics['auc']):
                print(f"Early stopping at epoch {epoch} (best val_auc={best_auc:.4f})")
                break

        print(f"Training complete. Best val_auc={best_auc:.4f}")
        return best_ckpt
