import os
import json
import numpy as np
from typing import List, Optional, Tuple

from .edf_reader import parse_summary, read_edf
from .preprocessing import filter_all_channels, sliding_window
from ..utils.config import Config


def _seizure_aware_split(
    edf_files: List[str],
    seizure_map: dict,
    train_ratio: float,
    val_ratio: float,
):
    """按发作事件顺序分配文件，保证 val/test 含癫痫事件。

    1. 含发作文件按比例分到 train/val/test（保时间序）。
    2. 非发作文件按时间位置填充到相邻子集。
    3. 各子集内部重新排序，保持时间先后。
    """
    sz_files     = [f for f in edf_files if seizure_map.get(f)]
    non_sz_files = [f for f in edf_files if not seizure_map.get(f)]
    n_sz = len(sz_files)

    if n_sz == 0:
        n = len(edf_files)
        n_tr = max(1, int(n * train_ratio))
        n_v  = max(1, int(n * val_ratio))
        return edf_files[:n_tr], edf_files[n_tr:n_tr + n_v], edf_files[n_tr + n_v:]

    n_sz_train = max(1, round(n_sz * train_ratio))
    if n_sz >= 2:
        n_sz_val = max(1, min(round(n_sz * val_ratio), n_sz - n_sz_train))
    else:
        n_sz_val = 0
    # n_sz_test = n_sz - n_sz_train - n_sz_val (implicit)

    sz_train = sz_files[:n_sz_train]
    sz_val   = sz_files[n_sz_train:n_sz_train + n_sz_val]
    sz_test  = sz_files[n_sz_train + n_sz_val:]

    train_boundary = sz_train[-1]
    val_boundary   = sz_val[-1] if sz_val else train_boundary

    non_tr, non_v, non_te = [], [], []
    for f in non_sz_files:
        if f < train_boundary:
            non_tr.append(f)
        elif f < val_boundary:
            non_v.append(f)
        else:
            non_te.append(f)

    return (
        sorted(sz_train + non_tr),
        sorted(sz_val   + non_v),
        sorted(sz_test  + non_te),
    )


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

    train_files, val_files, test_files = _seizure_aware_split(
        edf_files, seizure_map, cfg.train_ratio, cfg.val_ratio
    )

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

    train_files, val_files, test_files = _seizure_aware_split(
        edf_files, seizure_map, cfg.train_ratio, cfg.val_ratio
    )

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

