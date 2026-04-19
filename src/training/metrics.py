import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def compute_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics.

    Args:
        probs:     (N,) seizure-class probability scores (float in [0, 1]).
        labels:    (N,) ground-truth binary labels (0=normal, 1=seizure).
        threshold: decision threshold for binary predictions.

    Returns:
        dict with keys: sensitivity, specificity, auc, f1, tp, tn, fp, fn.
    """
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float('nan')

    f1 = float(f1_score(labels, preds, zero_division=0))

    return {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'auc': auc,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
