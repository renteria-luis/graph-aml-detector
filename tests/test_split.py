# tests/test.split.py

import pandas as pd

from data.loader import EllipticData
from data.split import temporal_split, to_xy

def make_data() -> EllipticData:
    features = pd.DataFrame({
        'txId': [1, 2, 3, 4, 5, 6, 7, 8],
        'time_step': [1, 34, 35, 2, 14, 15, 36, 37],
        'feat_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'feat_2': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    })
    labels = pd.DataFrame({
        'txId': [1, 2, 3, 4, 5, 6, 7, 8],
        'label': ['licit', 'illicit', 'unknown', 'illicit', 'licit', 'licit', 'unknown', 'licit']
    })
    edges = pd.DataFrame({'txId1': [10], 'txId2': [11]})
    return EllipticData(features=features, labels=labels, edges=edges)


def test_boundary_no_feature_in_train():
    train, test = temporal_split(make_data(), train_max_step=34)
    assert (train['time_step'] <= 34).all()
    assert (test['time_step'] > 34).all()


def test_drops_unknown_and_binary_labels():
    train, test = temporal_split(make_data())
    both = pd.concat([train, test])
    assert 'unknown' not in both['label'].values
    assert set(both['y'].unique()) <= {0, 1}
    assert both.loc[both['label'] == 'illicit', 'y'].eq(1).all()
    assert both.loc[both['label'] == 'licit', 'y'].eq(0).all()


def test_train_test_disjoint_and_cover_labeled():
    data = make_data()
    train, test = temporal_split(data)
    train_ids, test_ids = set(train['txId']), set(test['txId'])
    assert train_ids.isdisjoint(test_ids)
    labeled = set(data.labels.loc[data.labels['label'] != 'unknown', 'txId'])
    assert train_ids | test_ids == labeled


def test_to_xy_selects_features():
    train, _ = temporal_split(make_data())
    X, y = to_xy(train, ['feat_1'])
    assert list(X.columns) == ['feat_1']
    assert len(X) == len(y)