# src/data/split.py
import numpy as np
import pandas as pd

from config import TRAIN_MAX_STEP
from data.loader import EllipticData


def temporal_split(
    data: EllipticData, train_max_step: int = TRAIN_MAX_STEP
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.merge(data.features, data.labels, on='txId')
    df = df[df['label'] != 'unknown'].copy()
    df['y'] = (df['label'] == 'illicit').astype(int)

    train = df[df['time_step'] <= train_max_step]
    test = df[df['time_step'] > train_max_step]
    return train, test


def to_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    return df[feature_cols], df['y'].to_numpy()