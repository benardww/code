#!/usr/bin/env python3
"""
Evaluate a trained SeizureDetector on the test split of one CHB-MIT subject.

Inference uses a uniform stride (1 s) to produce a continuous probability
sequence per recording. Post-processing (MAF → threshold → collar) is then
applied, and metrics are reported before and after post-processing.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.utils.config import Config
from src.data.edf_reader import parse_summary, read_edf
from src.data.preprocessing import filter_all_channels
from src.models.seizure_detector import SeizureDetector
from src.training.postprocessing import infer_uniform_stride, postprocess
from src.training.metrics import compute_metrics


def _build_gt_sequence(T: int, seizure_intervals, stride: int, win_len: int) -> np.ndarray:
    """
    Build a binary ground-truth array aligned with the uniform-stride window sequence.
    A window is labelled 1 if it contains any seizure sample.
    """
    sample_labels = np.zeros(T, dtype=np.int8)
    for s, e in seizure_intervals:
        sample_labels[max(0, s):min(T, e)] = 1

    gt = []
    pos = 0
    while pos + win_len <= T:
        gt.append(1 if sample_labels[pos:pos + win_len].sum() > 0 else 0)
        pos += stride
    return np.array(gt, dtype=np.int8)


def evaluate_subject(subject_name: str, cfg: Config) -> dict:
    subject_proc_dir = os.path.join(cfg.data_processed_dir, subject_name)
    subject_raw_dir = os.path.join(cfg.data_raw_dir, subject_name)
    ckpt_path = os.path.join('experiments', subject_name, 'best_model.pth')

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = SeizureDetector(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test file list from stats.json
    with open(os.path.join(subject_proc_dir, 'stats.json')) as f:
        stats = json.load(f)
    test_files = stats.get('test_files', [])
    if not test_files:
        print(f"  [{subject_name}] No test files.")
        return {}

    # Parse seizure annotations
    summary_files = [f for f in os.listdir(subject_raw_dir) if f.endswith('-summary.txt')]
    seizure_map = parse_summary(os.path.join(subject_raw_dir, summary_files[0]))

    stride_sec = cfg.infer_stride / cfg.fs  # seconds per window step

    all_probs_raw, all_gt_raw = [], []   # for AUC (pre-post-processing)
    all_final, all_gt_post = [], []       # for post-processing metrics

    for edf_name in test_files:
        edf_path = os.path.join(subject_raw_dir, edf_name)
        if not os.path.isfile(edf_path):
            print(f"  [{subject_name}] EDF not found: {edf_name}")
            continue

        data, ok = read_edf(edf_path)
        if not ok:
            print(f"  [{subject_name}] Skipping {edf_name} (channel mismatch)")
            continue

        filtered = filter_all_channels(data, cfg)
        T = filtered.shape[1]

        intervals_sec = seizure_map.get(edf_name, [])
        intervals_samples = [
            (max(0, int(s * cfg.fs)), min(T, int(e * cfg.fs)))
            for s, e in intervals_sec
        ]

        # Raw probabilities from uniform-stride inference
        probs = infer_uniform_stride(model, filtered, cfg.n_timesteps, cfg.infer_stride, device)
        gt = _build_gt_sequence(T, intervals_samples, cfg.infer_stride, cfg.n_timesteps)

        min_len = min(len(probs), len(gt))
        probs, gt = probs[:min_len], gt[:min_len]

        all_probs_raw.extend(probs.tolist())
        all_gt_raw.extend(gt.tolist())

        # Post-process this file's probability sequence independently
        final = postprocess(probs, cfg.maf_window, cfg.threshold, cfg.collar_sec, stride_sec)
        all_final.extend(final.tolist())
        all_gt_post.extend(gt.tolist())

    if not all_gt_raw:
        print(f"  [{subject_name}] No test data processed.")
        return {}

    probs_arr = np.array(all_probs_raw, dtype=np.float32)
    gt_arr = np.array(all_gt_raw, dtype=np.int8)
    final_arr = np.array(all_final, dtype=np.int8)
    gt_post_arr = np.array(all_gt_post, dtype=np.int8)

    raw_metrics = compute_metrics(probs_arr, gt_arr, threshold=cfg.threshold)

    # Post-processing metrics (binary predictions, no AUC)
    cm = confusion_matrix(gt_post_arr, final_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    post_metrics = {
        'sensitivity': tp / (tp + fn + 1e-8),
        'specificity': tn / (tn + fp + 1e-8),
        'f1': f1_score(gt_post_arr, final_arr, zero_division=0),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

    sep = '=' * 55
    print(f"\n{sep}")
    print(f"Subject : {subject_name}  |  test files : {len(test_files)}")
    print(f"Pre-post  | AUC={raw_metrics['auc']:.4f}  "
          f"Sens={raw_metrics['sensitivity']:.4f}  Spec={raw_metrics['specificity']:.4f}")
    print(f"Post-proc | F1={post_metrics['f1']:.4f}  "
          f"Sens={post_metrics['sensitivity']:.4f}  Spec={post_metrics['specificity']:.4f}  "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")
    print(sep)

    return {'raw': raw_metrics, 'post': post_metrics}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate seizure detector on a CHB-MIT subject')
    parser.add_argument('--subject', type=str, required=True, help='e.g. chb01')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    cfg = Config(device=args.device)
    evaluate_subject(args.subject, cfg)
