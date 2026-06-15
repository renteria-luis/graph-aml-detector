# src/data/loader.py
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATA_RAW

LABEL_MAP = {'1': 'illicit', '2': 'licit', 'unknown': 'unknown'}


@dataclass
class EllipticData:
    features: pd.DataFrame  # txId, time_step, feat_1 .. feat_165
    labels: pd.DataFrame    # txId, label
    edges: pd.DataFrame     # txId1, txId2


def load_elliptic(raw_dir: Path = DATA_RAW) -> EllipticData:
    features = pd.read_csv(raw_dir / 'elliptic_txs_features.csv', header=None)
    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(1, features.shape[1] - 1)]

    labels = pd.read_csv(raw_dir / 'elliptic_txs_classes.csv')
    labels['label'] = labels['class'].astype(str).map(LABEL_MAP)
    labels = labels[['txId', 'label']]

    edges = pd.read_csv(raw_dir / 'elliptic_txs_edgelist.csv')

    return EllipticData(features=features, labels=labels, edges=edges)