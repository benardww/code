import os
import re
import numpy as np

import mne
from typing import Dict, List, Optional, Tuple

mne.set_log_level('WARNING')

STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]
_STANDARD_UPPER = [ch.upper() for ch in STANDARD_CHANNELS]


def _normalize_ch(name: str) -> str:
    """Normalize CHB-MIT channel name to uppercase without prefix/suffix artifacts."""
    name = name.strip()
    if name.upper().startswith('EEG '):
        name = name[4:].strip()
    # Remove -REF or -LE suffixes
    for suffix in ('-REF', '-LE'):
        if name.upper().endswith(suffix):
            name = name[:-len(suffix)]
    name = name.rstrip('.')
    name = re.sub(r'-\d+$', '', name)
    return name.upper()


def parse_summary(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse a CHB-MIT *-summary.txt file.

    Returns:
        dict mapping EDF filename → list of (start_sec, end_sec) tuples.
        Files with no seizures are mapped to an empty list.
    """
    seizures: Dict[str, List[Tuple[int, int]]] = {}
    current_file: Optional[str] = None
    starts: List[int] = []
    ends: List[int] = []

    with open(summary_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                if current_file is not None:
                    seizures[current_file] = list(zip(starts, ends))
                current_file = line.split(':', 1)[1].strip()
                starts, ends = [], []
            elif 'Start Time:' in line and 'seconds' in line:
                m = re.search(r'(\d+)\s*seconds', line)
                if m:
                    starts.append(int(m.group(1)))
            elif 'End Time:' in line and 'seconds' in line:
                m = re.search(r'(\d+)\s*seconds', line)
                if m:
                    ends.append(int(m.group(1)))

    if current_file is not None:
        seizures[current_file] = list(zip(starts, ends))

    return seizures


def read_edf(edf_path: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    Read an EDF file and extract the 18 standard channels.

    Returns:
        (data, success) where data is (18, T) float32 array in Volts,
        or (None, False) if any required channel is missing.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Cannot read {edf_path}: {e}")
        return None, False

    # Build normalized-name → original-name mapping
    ch_map: Dict[str, str] = {}
    for ch in raw.ch_names:
        normalized = _normalize_ch(ch)
        ch_map[normalized] = ch

    missing = [ch for ch in _STANDARD_UPPER if ch not in ch_map]
    if missing:
        return None, False

    selected = [ch_map[ch] for ch in _STANDARD_UPPER]
    raw.pick_channels(selected)
    raw.reorder_channels(selected)

    data, _ = raw[:]
    return data.astype(np.float32), True
