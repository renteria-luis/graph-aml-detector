# tests/test_metrics.py
import numpy as np

from evaluation.metrics import evaluate


def test_perfect_predictor():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.2, 0.8, 1.0])
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.pr_auc == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_predicting_all_negative_gives_zero_recall():
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.0])  # every score below 0.5
    m = evaluate(y_true, y_proba, threshold=0.5)
    assert m.recall == 0.0
    assert m.precision == 0.0
    assert m.f1 == 0.0


def test_pr_auc_stays_in_unit_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 100)
    y_proba = rng.random(100)
    m = evaluate(y_true, y_proba)
    assert 0.0 <= m.pr_auc <= 1.0