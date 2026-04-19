import os
import json
import numpy as np
from typing import List

from .edf_reader import parse_summary, read_edf
from .preprocessing import filter_all_channels, sliding_window
from ..utils.config import Config


def process_subject(subject_dir: str, out_dir: str, cfg: Config) -> bool:
    """
    Pre-process all EDF files for one subject and save train/val windows.

    Directory layout expected:
        subject_dir/
            chb01_01.edf
            chb01-summary.txt
            ...

    Output saved to out_dir/:
        train_X.npy, train_y.npy
        val_X.npy,   val_y.npy
        stats.json  (window counts, focal_alpha, split file lists)

    Returns True on success, False if no usable data was found.
    """
    subject_name = os.path.basename(subject_dir)

    # Locate summary file
    summary_files = [f for f in os.listdir(subject_dir) if f.endswith('-summary.txt')]
    if not summary_files:
        print(f"  [{subject_name}] No summary file, skipping.")
        return False
    seizure_map = parse_summary(os.path.join(subject_dir, summary_files[0]))

    # List EDF files in chronological order (filename order is time order in CHB-MIT)
    edf_files: List[str] = sorted(f for f in os.listdir(subject_dir) if f.endswith('.edf'))
    if not edf_files:
        print(f"  [{subject_name}] No EDF files, skipping.")
        return False

    # Split by file count (70 / 15 / 15)
    n = len(edf_files)
    n_train = max(1, int(n * cfg.train_ratio))
    n_val = max(1, int(n * cfg.val_ratio))
    train_files = edf_files[:n_train]
    val_files = edf_files[n_train:n_train + n_val]
    test_files = edf_files[n_train + n_val:]

    os.makedirs(out_dir, exist_ok=True)

    n_seizure_train = 0
    n_normal_train = 0

    for split_name, file_list in [('train', train_files), ('val', val_files)]:
        X_list, y_list = [], []
        for edf_name in file_list:
            edf_path = os.path.join(subject_dir, edf_name)
            if not os.path.isfile(edf_path):
                continue
            data, ok = read_edf(edf_path)
            if not ok:
                print(f"  [{subject_name}] Skipping {edf_name} (missing channels or read error)")
                continue

            filtered = filter_all_channels(data, cfg)
            T = filtered.shape[1]

            intervals_sec = seizure_map.get(edf_name, [])
            intervals_samples = [
                (max(0, int(s * cfg.fs)), min(T, int(e * cfg.fs)))
                for s, e in intervals_sec
            ]

            X, y = sliding_window(filtered, intervals_samples, cfg)
            if len(X) == 0:
                continue
            X_list.append(X)
            y_list.append(y)

        if not X_list:
            print(f"  [{subject_name}] No valid windows for {split_name} split, skipping split.")
            continue

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        np.save(os.path.join(out_dir, f'{split_name}_X.npy'), X_all)
        np.save(os.path.join(out_dir, f'{split_name}_y.npy'), y_all)

        n_sz = int((y_all == 1).sum())
        n_no = int((y_all == 0).sum())
        print(f"  [{subject_name}] {split_name}: {len(X_all)} windows  (seizure={n_sz}, normal={n_no})")

        if split_name == 'train':
            n_seizure_train = n_sz
            n_normal_train = n_no

    # Compute focal_alpha from training class distribution
    n_total = n_seizure_train + n_normal_train
    focal_alpha = round(1.0 - n_seizure_train / n_total, 4) if n_total > 0 else 0.9

    stats = {
        'n_seizure_train': n_seizure_train,
        'n_normal_train': n_normal_train,
        'focal_alpha': focal_alpha,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
    }
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  [{subject_name}] focal_alpha = {focal_alpha:.4f}  |  test files = {len(test_files)}")
    return True
