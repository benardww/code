#!/usr/bin/env python3
"""
Evaluate a trained SeizureDetector on the test split of one CHB-MIT subject.

Two evaluation schemes:
  Segment-based — sensitivity, specificity, accuracy (per sliding window)
  Event-based    — duration, n_testing, seizures_number, true_detection_sensitivity,
                   FDR (/h), latency (s)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.utils.config import Config
from src.data.edf_reader import parse_summary, read_edf
from src.data.preprocessing import filter_all_channels
from src.data.build_dataset import _seizure_aware_split
from src.models.seizure_detector import SeizureDetector
from src.training.postprocessing import infer_uniform_stride, postprocess


def _build_gt_sequence(T: int, seizure_intervals, stride: int, win_len: int) -> np.ndarray:
    sample_labels = np.zeros(T, dtype=np.int8)
    for s, e in seizure_intervals:
        sample_labels[max(0, s):min(T, e)] = 1
    gt = []
    pos = 0
    while pos + win_len <= T:
        gt.append(1 if sample_labels[pos:pos + win_len].sum() > 0 else 0)
        pos += stride
    return np.array(gt, dtype=np.int8)


def _extract_events(binary_seq: np.ndarray, stride_sec: float):
    """Extract contiguous 1-segments from a binary sequence.

    Returns list of (start_sec, end_sec).
    """
    events = []
    in_event = False
    start = 0
    for i, val in enumerate(binary_seq):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start * stride_sec, i * stride_sec))
    if in_event:
        events.append((start * stride_sec, len(binary_seq) * stride_sec))
    return events


def _compute_event_metrics(detected_events, gt_events, test_duration_sec):
    """Compute event-level metrics.

    Matching rule: a predicted event matches a GT event if they overlap.
    Each GT event is matched at most once (greedy, by prediction order).

    Returns dict with:
        duration_test_h, number_of_testing, seizures_number,
        true_detection_sensitivity, FDR_per_h, latency_s
    """
    test_duration_h = test_duration_sec / 3600.0 if test_duration_sec > 0 else 0.0
    n_testing = len(detected_events)
    seizures_number = len(gt_events)

    matched_gt = set()
    latencies = []

    for p_start, p_end in detected_events:
        best_gt_idx = None
        for idx, (g_start, g_end) in enumerate(gt_events):
            if idx in matched_gt:
                continue
            if p_start < g_end and p_end > g_start:  # overlap
                best_gt_idx = idx
                break
        if best_gt_idx is not None:
            matched_gt.add(best_gt_idx)
            latencies.append(max(0.0, p_start - gt_events[best_gt_idx][0]))

    TP_event = len(matched_gt)
    FP_event = n_testing - TP_event

    true_detection_sensitivity = TP_event / seizures_number if seizures_number > 0 else float('nan')
    FDR_per_h = FP_event / test_duration_h if test_duration_h > 0 else 0.0
    latency_s = np.mean(latencies) if latencies else float('nan')

    return {
        'duration_test_h': test_duration_h,
        'number_of_testing': n_testing,
        'seizures_number': seizures_number,
        'true_detection_sensitivity': true_detection_sensitivity,
        'FDR_per_h': FDR_per_h,
        'latency_s': latency_s,
    }


def evaluate_subject(subject_name: str, cfg: Config, experiments_dir: str = 'experiments') -> dict:
    subject_raw_dir = os.path.join(cfg.data_raw_dir, subject_name)
    ckpt_path = os.path.join(experiments_dir, subject_name, 'best_model.pth')

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    model = SeizureDetector(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    edf_files = sorted(f for f in os.listdir(subject_raw_dir) if f.endswith('.edf'))

    summary_files = [f for f in os.listdir(subject_raw_dir) if f.endswith('-summary.txt')]
    seizure_map = parse_summary(os.path.join(subject_raw_dir, summary_files[0]))

    _, _, test_files = _seizure_aware_split(
        edf_files, seizure_map, cfg.train_ratio, cfg.val_ratio
    )
    if not test_files:
        print(f"  [{subject_name}] No test files.")
        return {}

    stride_sec = cfg.infer_stride / cfg.fs

    all_probs_raw, all_gt_raw = [], []
    all_final, all_gt_post = [], []
    test_duration_sec = 0.0  # total recording seconds (across all test files)

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
        test_duration_sec += T / cfg.fs

        intervals_sec = seizure_map.get(edf_name, [])
        intervals_samples = [
            (max(0, int(s * cfg.fs)), min(T, int(e * cfg.fs)))
            for s, e in intervals_sec
        ]

        probs = infer_uniform_stride(model, filtered, cfg.n_timesteps, cfg.infer_stride, device)
        gt = _build_gt_sequence(T, intervals_samples, cfg.infer_stride, cfg.n_timesteps)

        min_len = min(len(probs), len(gt))
        probs, gt = probs[:min_len], gt[:min_len]

        all_probs_raw.extend(probs.tolist())
        all_gt_raw.extend(gt.tolist())

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

    # ── Segment-based metrics ──────────────────────────────────────────────
    cm = confusion_matrix(gt_post_arr, final_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    segment_metrics = {
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
    }

    # ── Event-based metrics ────────────────────────────────────────────────
    detected_events = _extract_events(final_arr, stride_sec)
    # Ground-truth events from summary (in seconds)
    gt_events_sec = []
    for edf_name in test_files:
        for s, e in seizure_map.get(edf_name, []):
            gt_events_sec.append((float(s), float(e)))

    event_metrics = _compute_event_metrics(detected_events, gt_events_sec, test_duration_sec)

    # ── Per-subject print ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"Subject : {subject_name}  |  test files : {len(test_files)}")
    print(f"Segment  | Sens={segment_metrics['sensitivity']:.4f}  "
          f"Spec={segment_metrics['specificity']:.4f}  "
          f"Acc={segment_metrics['accuracy']:.4f}")
    lat_str = f"{event_metrics['latency_s']:.2f}s" if not np.isnan(event_metrics['latency_s']) else "N/A"
    detsens_val = event_metrics['true_detection_sensitivity']
    detsens_str = f"{detsens_val:.4f}" if not np.isnan(detsens_val) else "N/A"
    print(f"Event    | Dur={event_metrics['duration_test_h']:.2f}h  "
          f"nTest={event_metrics['number_of_testing']}  "
          f"nSz={event_metrics['seizures_number']}  "
          f"DetSens={detsens_str}  "
          f"FDR={event_metrics['FDR_per_h']:.2f}/h  "
          f"Lat={lat_str}")
    print(sep)

    return {'segment': segment_metrics, 'event': event_metrics}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate seizure detector on a CHB-MIT subject')
    parser.add_argument('--subject', type=str, required=True, help='e.g. chb01')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    cfg = Config(device=args.device)
    evaluate_subject(args.subject, cfg)
