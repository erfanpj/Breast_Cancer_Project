"""Leakage-free classifier comparison on the scikit-learn breast cancer dataset.

Run:  python evaluate.py            (writes everything to results/)

Procedure (fixed before the test set is used):
1. Split the raw samples into a development set (75%) and a held-out test set (25%),
   stratified, seed 42. Nothing below sees the test rows until step 5.
2. Every candidate is one Pipeline: scaler -> optional PCA -> classifier. Scaler type,
   PCA (off / 5 / 10 components) and classifier settings are tuned together with
   GridSearchCV on stratified 5-fold CV inside the development set.
3. Candidates are compared with nested cross-validation on the development set
   (outer: 5 folds x 3 repeats), so the comparison score is not the same score that
   picked the hyperparameters. Two baselines go through the same outer folds.
4. The candidate with the highest mean outer-CV F1 for the malignant class is selected,
   re-tuned on the full development set, and the protocol is written to disk.
   A descriptive PCA on/off comparison (development data only) is saved alongside;
   it does not influence the selection.
5. Only then is the test set scored, once, for the selected model and the two baselines.

Positive class: malignant. scikit-learn's raw target is 0 = malignant, 1 = benign, so this
script re-encodes y as 1 = malignant, 0 = benign and takes the class names from the dataset.
This is an educational benchmark, not a diagnostic tool.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TEST_SIZE = 0.25
SELECTION_METRIC = "f1"  # F1 of the positive (malignant) class
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
OUTER_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
N_BOOTSTRAP = 2000
POSITIVE_NAME = "malignant"
RESULTS = Path(__file__).resolve().parent / "results"

CV_SCORING = {
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}


def load_data():
    """Return X, y (1 = malignant) and class names ordered by encoded label."""
    data = load_breast_cancer()
    names = list(data.target_names)
    malignant_raw = names.index(POSITIVE_NAME)
    y = (data.target == malignant_raw).astype(int)
    benign_name = [n for n in names if n != POSITIVE_NAME][0]
    class_names = [str(benign_name), POSITIVE_NAME]  # index == encoded label
    return data.data, y, class_names, list(data.feature_names)


def split(X, y):
    """Stratified split of raw sample indices into development and test."""
    idx = np.arange(len(y))
    dev_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    return np.sort(dev_idx), np.sort(test_idx)


def _pipeline(clf):
    return Pipeline([("scale", StandardScaler()), ("reduce", "passthrough"), ("clf", clf)])


PREPROCESS_GRID = {
    "scale": [StandardScaler(), MinMaxScaler()],
    "reduce": ["passthrough", PCA(n_components=5, random_state=SEED), PCA(n_components=10, random_state=SEED)],
}


def candidates():
    """Tuned candidates: (pipeline, grid). Scaling and PCA are searched inside CV."""
    specs = {
        "logistic_regression": (LogisticRegression(max_iter=5000, random_state=SEED), {"clf__C": [0.1, 1, 10]}),
        "svm": (SVC(random_state=SEED), {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]}),
        "random_forest": (RandomForestClassifier(random_state=SEED, n_jobs=1), {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5]}),
        "decision_tree": (DecisionTreeClassifier(random_state=SEED), {"clf__max_depth": [3, 5, 10]}),
        "knn": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7, 11]}),
        "gaussian_nb": (GaussianNB(), {}),
        "mlp": (MLPClassifier(max_iter=2000, random_state=SEED), {"clf__hidden_layer_sizes": [(50,), (100,)], "clf__alpha": [1e-4, 1e-2]}),
    }
    out = {}
    for name, (clf, grid) in specs.items():
        out[name] = (_pipeline(clf), {**PREPROCESS_GRID, **grid})
    return out


def baselines():
    """Untuned reference models evaluated on exactly the same folds and test set."""
    return {
        "baseline_majority_class": Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
        "baseline_logreg_default": Pipeline([("scale", StandardScaler()),
                                             ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
    }


def tune(pipe, grid, X, y, n_jobs=1):
    """GridSearchCV on the given (development) data only; refits the best setting on it."""
    gs = GridSearchCV(pipe, grid, scoring=SELECTION_METRIC, cv=INNER_CV, n_jobs=n_jobs, refit=True)
    return gs.fit(X, y)


def nested_cv(X_dev, y_dev, n_jobs=-1):
    rows = []
    models = {name: GridSearchCV(p, g, scoring=SELECTION_METRIC, cv=INNER_CV, n_jobs=1)
              for name, (p, g) in candidates().items()}
    models.update(baselines())
    for name, est in models.items():
        t0 = time.perf_counter()
        res = cross_validate(est, X_dev, y_dev, cv=OUTER_CV, scoring=CV_SCORING, n_jobs=n_jobs,
                             error_score="raise")
        print(f"  nested CV {name:26s} F1={res['test_f1'].mean():.4f}±{res['test_f1'].std(ddof=1):.4f} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        for fold in range(len(res["test_f1"])):
            row = {"candidate": name, "outer_fold": fold}
            row.update({m: round(float(res[f"test_{m}"][fold]), 6) for m in CV_SCORING})
            rows.append(row)
    return pd.DataFrame(rows)


def pca_ablation(X_dev, y_dev, families, n_jobs=-1):
    """Descriptive only: nested-CV F1 with PCA forced off / 5 / 10, on development data.

    Runs after selection has been decided and is never used to choose the model."""
    out = {}
    options = {"no_pca": "passthrough", "pca_5": PCA(n_components=5, random_state=SEED),
               "pca_10": PCA(n_components=10, random_state=SEED)}
    for fam in families:
        pipe, grid = candidates()[fam]
        out[fam] = {}
        for label, reduce in options.items():
            g = {**grid, "reduce": [reduce]}
            res = cross_validate(GridSearchCV(pipe, g, scoring=SELECTION_METRIC, cv=INNER_CV, n_jobs=1),
                                 X_dev, y_dev, cv=OUTER_CV, scoring="f1", n_jobs=n_jobs)
            out[fam][label] = {"f1_mean": round(float(res["test_score"].mean()), 6),
                               "f1_std": round(float(res["test_score"].std(ddof=1)), 6)}
            print(f"  PCA ablation {fam:20s} {label:7s} F1={out[fam][label]['f1_mean']:.4f}"
                  f"±{out[fam][label]['f1_std']:.4f}", flush=True)
    return out


def summarize_cv(cv_df):
    agg = cv_df.groupby("candidate", sort=False)[list(CV_SCORING)].agg(["mean", "std"])
    summary = {}
    for name in agg.index:
        summary[name] = {f"{m}_{s}": round(float(agg.loc[name, (m, s)]), 6)
                         for m in CV_SCORING for s in ("mean", "std")}
    return summary


def select(summary):
    eligible = {k: v for k, v in summary.items() if k != "baseline_majority_class"}
    # Highest mean outer-CV F1; ties broken by lower std, then name (deterministic).
    return sorted(eligible, key=lambda k: (-eligible[k]["f1_mean"], eligible[k]["f1_std"], k))[0]


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def test_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = {
        "f1_malignant": f1_score(y_true, y_pred, zero_division=0),
        "recall_malignant_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "precision_malignant": precision_score(y_true, y_pred, zero_division=0),
        "specificity_benign_recall": tn / (tn + fp),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_score)) > 1 else 0.5,
    }
    m = {k: round(float(v), 6) for k, v in m.items()}
    m["confusion_matrix"] = {"tn_benign_as_benign": int(tn), "fp_benign_as_malignant": int(fp),
                             "fn_malignant_as_benign": int(fn), "tp_malignant_as_malignant": int(tp)}
    return m


def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    f1s, recs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        f1s.append(f1_score(y_true[i], y_pred[i], zero_division=0))
        recs.append(recall_score(y_true[i], y_pred[i], zero_division=0))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 6), round(float(np.percentile(a, 97.5)), 6)]
    return {"method": f"percentile bootstrap over test samples, {n} resamples, seed {seed}",
            "f1_malignant_95ci": q(f1s), "recall_malignant_95ci": q(recs)}


def _jsonable(params):
    return {k: (repr(v) if not isinstance(v, (int, float, str, type(None))) else v) for k, v in params.items()}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def save_confusion_png(cm_dict, class_names, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = np.array([[cm_dict["tn_benign_as_benign"], cm_dict["fp_benign_as_malignant"]],
                   [cm_dict["fn_malignant_as_benign"], cm_dict["tp_malignant_as_malignant"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, metadata={"Software": None})
    plt.close(fig)


def main():
    t_start = time.perf_counter()
    RESULTS.mkdir(exist_ok=True)
    X, y, class_names, _ = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features; encoded labels {dict(enumerate(class_names))}; "
          f"malignant={int(y.sum())}, benign={int((1 - y).sum())}")

    dev_idx, test_idx = split(X, y)
    X_dev, y_dev, X_test, y_test = X[dev_idx], y[dev_idx], X[test_idx], y[test_idx]
    split_info = {"seed": SEED, "test_size": TEST_SIZE, "stratified": True,
                  "dev_indices": dev_idx.tolist(), "test_indices": test_idx.tolist(),
                  "dev_counts": {"benign": int((y_dev == 0).sum()), "malignant": int(y_dev.sum())},
                  "test_counts": {"benign": int((y_test == 0).sum()), "malignant": int(y_test.sum())},
                  "test_indices_sha256": hashlib.sha256(",".join(map(str, test_idx)).encode()).hexdigest()}
    _write_json(RESULTS / "split.json", split_info)
    print(f"Split: dev={len(dev_idx)} {split_info['dev_counts']}, test={len(test_idx)} {split_info['test_counts']}")

    # ---- Development data only from here until the protocol is frozen ----
    print("Nested CV on development data (outer 5x3, inner 5-fold GridSearchCV):")
    cv_df = nested_cv(X_dev, y_dev)
    cv_df.to_csv(RESULTS / "cv_outer_folds.csv", index=False, float_format="%.6f")
    summary = summarize_cv(cv_df)
    _write_json(RESULTS / "cv_summary.json", summary)

    chosen = select(summary)
    families = sorted({"logistic_regression", chosen} & set(candidates()))
    print("PCA ablation on development data (descriptive, not used for selection):")
    _write_json(RESULTS / "pca_ablation_dev_cv.json", pca_ablation(X_dev, y_dev, families))
    if chosen in candidates():
        pipe, grid = candidates()[chosen]
        gs = tune(pipe, grid, X_dev, y_dev, n_jobs=-1)
        final_model, best_params, inner_best = gs.best_estimator_, _jsonable(gs.best_params_), round(float(gs.best_score_), 6)
    else:
        final_model = baselines()[chosen].fit(X_dev, y_dev)
        best_params, inner_best = {}, None
    reference = {name: est.fit(X_dev, y_dev) for name, est in baselines().items()}

    protocol = {
        "positive_class": POSITIVE_NAME, "label_encoding": dict(enumerate(class_names)),
        "selection_metric": "F1 of the malignant class (outer nested-CV mean on development data)",
        "inner_cv": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "outer_cv": "RepeatedStratifiedKFold(5 splits x 3 repeats, random_state=42)",
        "preprocessing_search": {"scale": ["StandardScaler", "MinMaxScaler"], "reduce": ["none", "PCA(5)", "PCA(10)"]},
        "candidates": sorted(candidates()), "baselines": sorted(baselines()),
        "selected_candidate": chosen, "selected_params": best_params,
        "selected_inner_cv_f1_on_full_dev": inner_best,
        "selected_outer_cv": summary[chosen],
        "test_indices_sha256": split_info["test_indices_sha256"],
        "seed": SEED,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    _write_json(RESULTS / "protocol.json", protocol)
    protocol_sha = hashlib.sha256((RESULTS / "protocol.json").read_bytes()).hexdigest()
    print(f"Selected on CV only: {chosen} {best_params}; protocol frozen (sha256 {protocol_sha[:12]})")

    # ---- Single held-out test evaluation ----
    test = {"protocol_sha256": protocol_sha, "n_test": int(len(y_test)), "models": {}}
    for name, model in [(chosen, final_model)] + [(n, m) for n, m in reference.items() if n != chosen]:
        pred = model.predict(X_test)
        test["models"][name] = test_metrics(y_test, pred, _scores(model, X_test))
        test["models"][name]["role"] = "selected" if name == chosen else "baseline"
        if name == chosen:
            test["models"][name]["bootstrap"] = bootstrap_ci(y_test, pred)
    _write_json(RESULTS / "test_metrics.json", test)
    save_confusion_png(test["models"][chosen]["confusion_matrix"], class_names,
                       f"Held-out test (n={len(y_test)}): {chosen}", RESULTS / "confusion_matrix_test.png")
    sel = test["models"][chosen]
    print(f"Test (once): {chosen} F1={sel['f1_malignant']:.4f} recall={sel['recall_malignant_sensitivity']:.4f} "
          f"precision={sel['precision_malignant']:.4f} CM={sel['confusion_matrix']}")
    for n in reference:
        if n != chosen:
            print(f"  baseline {n}: F1={test['models'][n]['f1_malignant']:.4f} acc={test['models'][n]['accuracy']:.4f}")
    print(f"Done in {time.perf_counter() - t_start:.1f}s; outputs in {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
