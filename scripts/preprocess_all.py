#!/usr/bin/env python3
"""Batch-preprocess all CHB-MIT subjects into train/val windows."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.utils.config import Config
from src.data.build_dataset import process_subject


def main():
    cfg = Config()
    raw_dir = cfg.data_raw_dir
    processed_dir = cfg.data_processed_dir

    if not os.path.isdir(raw_dir):
        print(f"Raw data directory not found: {raw_dir}")
        print("Please place CHB-MIT data under data/raw/chb01/, data/raw/chb02/, etc.")
        sys.exit(1)

    subjects = sorted(
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d)) and d not in cfg.excluded_subjects
    )

    if not subjects:
        print(f"No subject directories found in {raw_dir}")
        sys.exit(1)

    print(f"Found {len(subjects)} subjects: {subjects}")

    ok, failed = 0, []
    for subject in tqdm(subjects, desc='Preprocessing'):
        subject_raw_dir = os.path.join(raw_dir, subject)
        subject_out_dir = os.path.join(processed_dir, subject)
        print(f"\n--- {subject} ---")
        success = process_subject(subject_raw_dir, subject_out_dir, cfg)
        if success:
            ok += 1
        else:
            failed.append(subject)

    print(f"\nDone. {ok}/{len(subjects)} subjects processed successfully.")
    if failed:
        print(f"Failed subjects: {failed}")


if __name__ == '__main__':
    main()
