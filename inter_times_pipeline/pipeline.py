from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from core.data import Example
from core.splits import Split
from core.eval import EvalResult, evaluate_classifier


def _parse_window(text: str) -> np.ndarray:
    s = (text or "").strip()
    if not s:
        return np.zeros((0,), dtype=np.float32)
    parts = s.split()
    return np.asarray([float(p) for p in parts], dtype=np.float32)


def _build_matrix(examples: List[Example], idx: np.ndarray, *, expected_len: Optional[int] = None) -> np.ndarray:
    rows: List[np.ndarray] = []
    for i in idx:
        v = _parse_window(examples[int(i)].text)
        rows.append(v)

    if not rows:
        return np.zeros((0, expected_len or 0), dtype=np.float32)

    # Determine window length
    L = expected_len if expected_len is not None else max(len(r) for r in rows)
    X = np.zeros((len(rows), L), dtype=np.float32)
    for r_i, r in enumerate(rows):
        # pad/truncate to L
        m = min(L, len(r))
        if m > 0:
            X[r_i, :m] = r[:m]
    return X


def build_model(model_name: str, params: Optional[Dict[str, Any]] = None, *, random_state: int = 42) -> BaseEstimator:
    params = params or {}

    if model_name == "dummy_most_frequent":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "dummy_stratified":
        base = {"strategy": "stratified", "random_state": random_state}
        return DummyClassifier(**{**base, **params})

    if model_name == "logreg":
        base = {"max_iter": 5000, "solver": "lbfgs", "random_state": random_state}
        return LogisticRegression(**{**base, **params})

    if model_name == "svm":
        # LinearSVC does not accept random_state consistently across versions; safe to omit.
        return LinearSVC(**{**params})

    if model_name == "sgd_hinge":
        base = {"loss": "hinge", "random_state": random_state}
        return SGDClassifier(**{**base, **params})

    if model_name == "sgd_log":
        base = {"loss": "log_loss", "random_state": random_state}
        return SGDClassifier(**{**base, **params})

    if model_name == "ridge":
        # fast linear baseline; often strong on numeric features
        return RidgeClassifier(**{**params})

    if model_name == "knn":
        return KNeighborsClassifier(**{**params})

    if model_name == "gnb":
        # Gaussian Naive Bayes (works on dense numeric)
        return GaussianNB(**{**params})

    if model_name == "rf":
        base = {"random_state": random_state, "n_jobs": -1}
        return RandomForestClassifier(**{**base, **params})

    raise ValueError(f"Unknown model_name={model_name!r}")


@dataclass(frozen=True)
class Candidate:
    model_name: str
    model_params: Dict[str, Any]
    use_scaler: bool = True

def search(
    examples: List[Example],
    split: Split,
    candidates: Iterable[Candidate],
    *,
    metric: str = "f1_macro",
    random_state: int = 42,
    evaluate_test_for_all: bool = False,   # NEW
    verbose: bool = True,
) -> Tuple[Candidate, EvalResult, EvalResult, List[Tuple[Candidate, EvalResult]]]:

    if metric not in {"f1_macro", "f1_weighted", "accuracy", "balanced_accuracy"}:
        raise ValueError("metric must be one of: f1_macro, f1_weighted, accuracy, balanced_accuracy")

    def score(res: EvalResult) -> float:
        return float(getattr(res, metric))

    candidates = list(candidates)

    # labels (strings)
    y = np.array([ex.label for ex in examples], dtype=object)
    labels_sorted = sorted(set(y.tolist()))

    # Determine fixed window length from TRAIN (avoids peeking)
    train_rows = [_parse_window(examples[int(i)].text) for i in split.train_idx]
    if not train_rows or max(len(r) for r in train_rows) == 0:
        raise ValueError("No inter-time windows found in TRAIN. Check loader config/window sizes.")
    L = max(len(r) for r in train_rows)

    # Build matrices
    X_train = _build_matrix(examples, split.train_idx, expected_len=L)
    X_val   = _build_matrix(examples, split.val_idx, expected_len=L)
    X_test  = _build_matrix(examples, split.test_idx, expected_len=L)

    y_train, y_val, y_test = y[split.train_idx], y[split.val_idx], y[split.test_idx]

    best: Optional[Candidate] = None
    best_val: Optional[EvalResult] = None
    best_test: Optional[EvalResult] = None
    all_val: List[Tuple[Candidate, EvalResult]] = []

    for cand in candidates:
        steps = []
        if cand.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", build_model(cand.model_name, cand.model_params, random_state=random_state)))

        model = Pipeline(steps)
        model.fit(X_train, y_train)

        # --- VAL ---
        y_val_pred = model.predict(X_val)
        val_res = evaluate_classifier(y_val, y_val_pred, labels=labels_sorted)
        all_val.append((cand, val_res))

        # --- Optional TEST for all ---
        test_res: Optional[EvalResult] = None
        if evaluate_test_for_all:
            y_test_pred = model.predict(X_test)
            test_res = evaluate_classifier(y_test, y_test_pred, labels=labels_sorted)

        # --- Track best by VAL ---
        if best_val is None or score(val_res) > score(best_val):
            best = cand
            best_val = val_res
            # If we computed test for all, keep the best's test too
            best_test = test_res if evaluate_test_for_all else None

        if verbose:
            msg = f"[INTER] {cand.model_name} {cand.model_params} | val {metric}={score(val_res):.4f}"
            if evaluate_test_for_all and test_res is not None:
                msg += f" | test {metric}={score(test_res):.4f}"
            print(msg)

    assert best is not None and best_val is not None

    # If we did NOT evaluate test for all, evaluate test once for best candidate
    if best_test is None:
        steps = []
        if best.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", build_model(best.model_name, best.model_params, random_state=random_state)))

        model = Pipeline(steps)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        best_test = evaluate_classifier(y_test, y_test_pred, labels=labels_sorted)

    return best, best_val, best_test, all_val
