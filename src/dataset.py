"""
src/dataset.py — IRMAS, OpenMIC-2018, and CombinedDataset.
Identical logic to the Colab Cell 4 dataset classes.

Expected data layout:
    data/
    ├── IRMAS/
    │   └── IRMAS-TrainingData/
    │       ├── cel/  *.wav
    │       ├── cla/  *.wav
    │       └── ...
    └── OpenMIC/
        ├── audio/
        │   └── 000/ ... fff/  *.ogg
        └── openmic-2018-aggregated-labels.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, INSTRUMENT_CLASSES, NUM_CLASSES, IRMAS_MAP,
    TRAIN_SPLIT, VAL_SPLIT,
)
from src.preprocessor import random_crop, augment_spec


# ── IRMAS ─────────────────────────────────────────────────────────────────────

class IRMASDataset(Dataset):
    """
    IRMAS training dataset (single predominant instrument label).
    Download: https://zenodo.org/record/1290750
    """

    def __init__(self, split='train', augment=True):
        self.aug  = augment and split == 'train'
        root      = DATA_DIR / 'IRMAS' / 'IRMAS-TrainingData'
        rows      = []

        for code, inst in IRMAS_MAP.items():
            folder = root / code
            if not folder.exists():
                continue
            for wav in folder.glob('*.wav'):
                vec = [0.0] * NUM_CLASSES
                if inst in INSTRUMENT_CLASSES:
                    vec[INSTRUMENT_CLASSES.index(inst)] = 1.0
                rows.append((str(wav), vec))

        n   = len(rows)
        idx = np.random.RandomState(42).permutation(n)
        ntr = int(n * TRAIN_SPLIT)
        nva = int(n * VAL_SPLIT)

        sl = (slice(None, ntr)        if split == 'train'
              else slice(ntr, ntr+nva) if split == 'val'
              else slice(ntr+nva, None))

        self.samples = [rows[i] for i in idx[sl]]
        print(f'IRMAS [{split}]: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        spec = random_crop(path)
        if self.aug:
            spec = augment_spec(spec)
        return spec, torch.tensor(label, dtype=torch.float32)


# ── OpenMIC-2018 ──────────────────────────────────────────────────────────────

class OpenMICDataset(Dataset):
    """
    OpenMIC-2018 dataset (multi-label, 20 instruments).
    Download: https://zenodo.org/records/1432913
    """

    def __init__(self, split='train', augment=True):
        self.aug = augment and split == 'train'
        root     = DATA_DIR / 'OpenMIC'

        df   = pd.read_csv(root / 'openmic-2018-aggregated-labels.csv')
        wide = df.pivot_table(
            index='sample_key', columns='instrument',
            values='relevance', aggfunc='mean',
        ).fillna(0.0)

        for c in INSTRUMENT_CLASSES:
            if c not in wide.columns:
                wide[c] = 0.0

        self.labels = wide[INSTRUMENT_CLASSES].values.astype(np.float32)
        keys        = wide.index.tolist()
        self.paths  = [root / 'audio' / k[:3] / f'{k}.ogg' for k in keys]

        n   = len(keys)
        idx = np.random.RandomState(42).permutation(n)
        ntr = int(n * TRAIN_SPLIT)
        nva = int(n * VAL_SPLIT)

        self.idx = (idx[:ntr]          if split == 'train'
                    else idx[ntr:ntr+nva] if split == 'val'
                    else idx[ntr+nva:])

        print(f'OpenMIC [{split}]: {len(self.idx)} samples')

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j    = self.idx[i]
        spec = random_crop(self.paths[j])
        if self.aug:
            spec = augment_spec(spec)
        return spec, torch.tensor(self.labels[j], dtype=torch.float32)


# ── Combined ──────────────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    """Concatenates IRMAS + OpenMIC. Skips whichever is not downloaded."""

    def __init__(self, split='train'):
        self.ds, self.ls = [], []
        for Cls in [IRMASDataset, OpenMICDataset]:
            try:
                d = Cls(split)
                self.ds.append(d)
                self.ls.append(len(d))
            except Exception as e:
                print(f'Skipping {Cls.__name__}: {e}')
        if not self.ds:
            raise RuntimeError(
                'No datasets found. Download IRMAS and/or OpenMIC into data/.'
            )

    def __len__(self):
        return sum(self.ls)

    def __getitem__(self, i):
        for d, l in zip(self.ds, self.ls):
            if i < l:
                return d[i]
            i -= l
