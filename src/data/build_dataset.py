import os
import json
import numpy as np
from typing import List, Optional, Tuple

from .edf_reader import parse_summary, read_edf
from .preprocessing import filter_all_channels, sliding_window
from ..utils.config import Config


def _load_split(subject_dir: str, subject_name: str,
                file_list: List[str], seizure_map: dict,
                cfg: Config) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        return None, None
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def process_subject_to_memory(subject_dir: str, cfg: Config):
    """
    预处理一个受试者的所有 EDF 文件，返回内存中的数组，不写入磁盘。

    Returns:
        (train_X, train_y, val_X, val_y, focal_alpha) 或 None（失败时）
    """
    subject_name = os.path.basename(subject_dir)

    summary_files = [f for f in os.listdir(subject_dir) if f.endswith('-summary.txt')]
    if not summary_files:
        print(f"  [{subject_name}] No summary file, skipping.")
        return None
    seizure_map = parse_summary(os.path.join(subject_dir, summary_files[0]))

    edf_files: List[str] = sorted(f for f in os.listdir(subject_dir) if f.endswith('.edf'))
    if not edf_files:
        print(f"  [{subject_name}] No EDF files, skipping.")
        return None

    n = len(edf_files)
    n_train = max(1, int(n * cfg.train_ratio))
    n_val = max(1, int(n * cfg.val_ratio))
    train_files = edf_files[:n_train]
    val_files = edf_files[n_train:n_train + n_val]
    test_files = edf_files[n_train + n_val:]

    train_X, train_y = _load_split(subject_dir, subject_name, train_files, seizure_map, cfg)
    if train_X is None:
        print(f"  [{subject_name}] No valid windows for train split, skipping.")
        return None

    val_X, val_y = _load_split(subject_dir, subject_name, val_files, seizure_map, cfg)
    if val_X is None:
        # val 无数据时从 train 尾部划出 15%
        n_val_fb = max(1, int(len(train_X) * 0.15))
        val_X, val_y = train_X[-n_val_fb:], train_y[-n_val_fb:]
        train_X, train_y = train_X[:-n_val_fb], train_y[:-n_val_fb]
        print(f"  [{subject_name}] No val split data, using last {n_val_fb} train windows as val.")

    n_sz = int((train_y == 1).sum())
    n_no = int((train_y == 0).sum())
    n_total = n_sz + n_no
    focal_alpha = round(1.0 - n_sz / n_total, 4) if n_total > 0 else 0.9

    print(f"  [{subject_name}] train: {len(train_X)} windows  (seizure={n_sz}, normal={n_no})")
    print(f"  [{subject_name}] val:   {len(val_X)} windows")
    print(f"  [{subject_name}] focal_alpha = {focal_alpha:.4f}  |  test files = {len(test_files)}")

    return train_X, train_y, val_X, val_y, focal_alpha


def process_subject(subject_dir: str, out_dir: str, cfg: Config) -> bool:
    """
    预处理并将结果保存到磁盘（用于显式缓存，--mode preprocess）。

    Returns True on success, False if no usable data was found.
    """
    subject_name = os.path.basename(subject_dir)

    summary_files = [f for f in os.listdir(subject_dir) if f.endswith('-summary.txt')]
    if not summary_files:
        print(f"  [{subject_name}] No summary file, skipping.")
        return False
    seizure_map = parse_summary(os.path.join(subject_dir, summary_files[0]))

    edf_files: List[str] = sorted(f for f in os.listdir(subject_dir) if f.endswith('.edf'))
    if not edf_files:
        print(f"  [{subject_name}] No EDF files, skipping.")
        return False

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
        X_all, y_all = _load_split(subject_dir, subject_name, file_list, seizure_map, cfg)
        if X_all is None:
            print(f"  [{subject_name}] No valid windows for {split_name} split, skipping split.")
            continue
        np.save(os.path.join(out_dir, f'{split_name}_X.npy'), X_all)
        np.save(os.path.join(out_dir, f'{split_name}_y.npy'), y_all)
        n_sz = int((y_all == 1).sum())
        n_no = int((y_all == 0).sum())
        print(f"  [{subject_name}] {split_name}: {len(X_all)} windows  (seizure={n_sz}, normal={n_no})")
        if split_name == 'train':
            n_seizure_train = n_sz
            n_normal_train = n_no

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

    if not os.path.isfile(os.path.join(out_dir, 'train_X.npy')):
        print(f"  [{subject_name}] 未生成 train 数据，跳过该受试者。")
        return False

    return True

