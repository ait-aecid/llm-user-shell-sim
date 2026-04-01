#!/usr/bin/env python3
"""
log_token_attribution_linear_models.py

Train 3 models on log lines with TF-IDF features and, for each log line,
mark the SINGLE most relevant token/ngram present (highest contribution toward
the predicted class).

Models:
  - LinearSVC
  - LogisticRegression
  - ComplementNB (naive bayes variant good for text)

Install:
  pip install scikit-learn numpy
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB


@dataclass
class AttributionResult:
    model: str
    pred: int
    decision: float
    top_token: str
    top_contribution: float
    marked_text: str


def mark_first_occurrence(text: str, token: str) -> str:
    """Mark first occurrence of token in text; fallback to appending if not found."""
    idx = text.find(token)
    if idx == -1:
        return text + f"  [TOP_TOKEN={token}]"
    return text[:idx] + "[[" + token + "]]" + text[idx + len(token):]


def get_weights_for_attribution(clf) -> np.ndarray:
    """
    Return a weight vector w (n_features,) such that token contribution can be
    approximated by w_j * x_j (toward class 1).

    Supported:
      - LinearSVC, LogisticRegression (coef_)
      - ComplementNB / MultinomialNB (feature_log_prob_)
    """
    if hasattr(clf, "coef_"):
        # LinearSVC, LogisticRegression, SGDClassifier, RidgeClassifier, etc.
        w = clf.coef_.ravel()
        return w

    if hasattr(clf, "feature_log_prob_"):
        # Naive Bayes: use log-prob difference as "direction toward class 1"
        logp = clf.feature_log_prob_
        if logp.shape[0] != 2:
            raise ValueError("This script assumes binary classification (2 classes).")
        w = (logp[1] - logp[0]).ravel()
        return w

    raise TypeError(f"Unsupported model type for attribution: {type(clf).__name__}")


def decision_value(clf, X_row) -> float:
    """
    Get a scalar decision value:
      - for LinearSVC/LogisticRegression: decision_function
      - for NB: use log-prob difference (approx decision) if predict_log_proba exists
    """
    if hasattr(clf, "decision_function"):
        return float(clf.decision_function(X_row)[0])

    # Naive Bayes: use log-prob difference as a "decision"
    if hasattr(clf, "predict_log_proba"):
        lp = clf.predict_log_proba(X_row)[0]
        return float(lp[1] - lp[0])

    # Fallback: not ideal, but keeps code robust
    return float(clf.predict_proba(X_row)[0, 1] - 0.5)


def top_token_for_text(text: str, vectorizer: TfidfVectorizer, clf, model_name: str) -> AttributionResult:
    """
    Identify the single most relevant present token/ngram for the predicted class.

    Steps:
      1) Vectorize text -> sparse row X
      2) Compute per-feature contributions toward class 1: contrib = w * x
      3) If predicted class is 1: choose max(contrib)
         If predicted class is 0: choose max(-contrib)
    """
    X = vectorizer.transform([text])  # (1, n_features) CSR
    pred = int(clf.predict(X)[0])
    decision = decision_value(clf, X)

    w = get_weights_for_attribution(clf)
    feature_names = vectorizer.get_feature_names_out()

    nz = X.nonzero()[1]
    if nz.size == 0:
        return AttributionResult(
            model=model_name,
            pred=pred,
            decision=decision,
            top_token="<NO_FEATURES>",
            top_contribution=0.0,
            marked_text=text,
        )

    # For single-row CSR, X.data aligns with X.indices (non-zero columns)
    x_vals = X.data
    contrib_toward_1 = w[nz] * x_vals

    if pred == 1:
        k = int(np.argmax(contrib_toward_1))
        best_contrib = float(contrib_toward_1[k])
    else:
        k = int(np.argmax(-contrib_toward_1))
        best_contrib = float((-contrib_toward_1)[k])  # toward class 0

    best_feat_idx = int(nz[k])
    best_token = str(feature_names[best_feat_idx])

    marked = mark_first_occurrence(text, best_token)

    return AttributionResult(
        model=model_name,
        pred=pred,
        decision=decision,
        top_token=best_token,
        top_contribution=best_contrib,
        marked_text=marked,
    )


def main():
    # -----------------------------
    # 1) Sample data (your logs)
    # -----------------------------
    texts = [
        # SYSCALL → label 1
        "type=SYSCALL success=yes exit=0 uid=33 comm=apache2",
        "type=SYSCALL success=yes exit=0 uid=0 comm=cron",
        "type=SYSCALL success=no exit=13 uid=33 comm=apache2",
        "type=SYSCALL success=no exit=1 uid=0 comm=sudo",
        "type=SYSCALL success=yes exit=0 uid=1000 comm=systemd",
        "type=SYSCALL success=no exit=5 uid=1000 comm=ssh",
        "type=SYSCALL success=yes exit=0 uid=33 comm=nginx",
        "type=SYSCALL success=no exit=13 uid=33 comm=php-fpm",
        "type=SYSCALL success=yes exit=0 uid=0 comm=rsyslogd",
        "type=SYSCALL success=no exit=1 uid=1000 comm=bash",
        "type=SYSCALL success=yes exit=0 uid=33 comm=mysql",
        "type=SYSCALL success=no exit=2 uid=0 comm=systemctl",
        "type=SYSCALL success=yes exit=0 uid=1000 comm=python",
        "type=SYSCALL success=no exit=13 uid=33 comm=node",

        # EXECVE → label 0
        "type=EXECVE user=root cmd=/usr/bin/apt update",
        "type=EXECVE user=root cmd=/usr/bin/systemctl restart apache2",
        "type=EXECVE user=www-data cmd=/usr/bin/php -v",
        "type=EXECVE user=root cmd=/bin/sh -c ls /root",
        "type=EXECVE user=alice cmd=/usr/bin/vim /etc/hosts",
        "type=EXECVE user=root cmd=/usr/bin/journalctl -xe",
        "type=EXECVE user=bob cmd=/usr/bin/python script.py",
        "type=EXECVE user=root cmd=/bin/sh -c rm -rf /tmp/test",
        "type=EXECVE user=www-data cmd=/usr/bin/curl http://localhost",
        "type=EXECVE user=root cmd=/usr/bin/docker ps",
        "type=EXECVE user=alice cmd=/usr/bin/ssh user@server",
        "type=EXECVE user=root cmd=/usr/bin/find /var/log",
        "type=EXECVE user=bob cmd=/usr/bin/ls -la",
        "type=EXECVE user=root cmd=/usr/bin/tail -f /var/log/syslog",
        "type=EXECVE user=unknown cmd=/bin/sh -c nc 10.0.0.5 4444",
    ]
    y = [1] * 14 + [0] * 15

    # -----------------------------
    # 2) Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, random_state=42, stratify=y
    )

    # -----------------------------
    # 3) Vectorizer
    # NOTE:
    # - If you set ngram_range=(1,5), the "token" might be a bigram/trigram/etc.
    # - If you want only single tokens, use (1,1).
    # -----------------------------
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 5),
        lowercase=False,
        token_pattern=r"(?u)\b\w[\w\-/=:.]*\b",
    )

    Xtr = vectorizer.fit_transform(X_train)

    # -----------------------------
    # 4) Models
    # -----------------------------
    models: Dict[str, object] = {
        "LinearSVC": LinearSVC(C=1.0),
        "LogReg": LogisticRegression(max_iter=2000, C=1.0),
        "ComplementNB": ComplementNB(alpha=0.1),
    }

    # Train
    for name, clf in models.items():
        clf.fit(Xtr, y_train)

    # -----------------------------
    # 5) Attribution per log line (test set)
    # -----------------------------
    print("=== TOP TOKEN/NGGRAM ATTRIBUTION (TEST SET) ===")
    print("NOTE: With ngram_range=(1,5), the 'top token' may be a multi-word n-gram.\n")

    for line in X_test:
        print(f"LOG: {line}")
        for name, clf in models.items():
            res = top_token_for_text(line, vectorizer, clf, name)
            print(
                f"  [{res.model:11s}] pred={res.pred}  decision={res.decision:+.4f}  "
                f"top={res.top_token!r}  contrib={res.top_contribution:+.4f}"
            )
            print(f"    {res.marked_text}")
        print("-" * 100)


if __name__ == "__main__":
    main()
