"""Regression checks for the evaluation boundaries in evaluate.py."""
import hashlib
import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as ev  # noqa: E402

FIT_LOG = []  # module level so it survives sklearn's clone()


def _row_hashes(X):
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in X}


class FitRecorder(BaseEstimator, TransformerMixin):
    """Pass-through step that records which rows each fit() received."""

    def fit(self, X, y=None):
        FIT_LOG.append(_row_hashes(X))
        return self

    def transform(self, X):
        return X


def test_label_encoding_malignant_is_positive():
    X, y, class_names, _ = ev.load_data()
    assert class_names == ["benign", "malignant"]
    assert int(y.sum()) == 212 and int((1 - y).sum()) == 357


def test_split_is_disjoint_stratified_and_deterministic():
    X, y, _, _ = ev.load_data()
    dev, test = ev.split(X, y)
    assert set(dev).isdisjoint(test)
    assert len(dev) + len(test) == len(y)
    assert abs(y[test].mean() - y.mean()) < 0.01
    dev2, test2 = ev.split(X, y)
    assert np.array_equal(test, test2) and np.array_equal(dev, dev2)


def test_preprocessing_never_fits_on_test_rows_or_inner_validation_rows():
    X, y, _, _ = ev.load_data()
    dev, test = ev.split(X, y)
    FIT_LOG.clear()
    pipe = Pipeline([("rec", FitRecorder()), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000))])
    ev.tune(pipe, {"clf__C": [0.1, 1.0]}, X[dev], y[dev], n_jobs=1)
    test_rows = _row_hashes(X[test])
    assert FIT_LOG, "recorder was never fitted"
    assert all(rows.isdisjoint(test_rows) for rows in FIT_LOG)
    # Inner CV fits see only ~4/5 of the development rows; the final refit sees all of them.
    n_dev = len(dev)
    inner = [len(r) for r in FIT_LOG[:-1]]
    assert inner and all(n < n_dev for n in inner)
    assert len(FIT_LOG[-1]) == len(_row_hashes(X[dev]))


def test_refit_scaler_statistics_come_from_development_data_only():
    X, y, _, _ = ev.load_data()
    dev, _ = ev.split(X, y)
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
    gs = ev.tune(pipe, {"clf__C": [1.0]}, X[dev], y[dev], n_jobs=1)
    scaler = gs.best_estimator_.named_steps["scale"]
    assert np.allclose(scaler.mean_, X[dev].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_every_candidate_puts_preprocessing_inside_the_pipeline():
    for name, (pipe, grid) in ev.candidates().items():
        assert [s for s, _ in pipe.steps] == ["scale", "reduce", "clf"], name
        assert "scale" in grid and "reduce" in grid, name
