#!/usr/bin/env python3
"""
tfidf_compare_models.py

Compare multiple classic text classifiers on the same TF-IDF features:
- Linear SVM (LinearSVC)
- Logistic Regression
- Complement Naive Bayes

Uses a single reproducible train/test split (no overlap).
Prints accuracy + confusion matrix + classification report + sample predictions for each model.
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier



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
    # 2) Reproducible train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # -----------------------------
    # 3) Define models to compare
    # -----------------------------
    models = {
        "svm": LinearSVC(C=1.0),
        "logreg": LogisticRegression(max_iter=2000),
        "sgd_hinge": SGDClassifier(loss="hinge"),
        "sgd_log": SGDClassifier(loss="log_loss"),
        "passive": PassiveAggressiveClassifier(),
        "ridge": RidgeClassifier(),
        "mnb": MultinomialNB(alpha=0.1),
        "cnb": ComplementNB(alpha=0.1),
        "bnb": BernoulliNB(alpha=0.1),
        "knn": KNeighborsClassifier(metric="cosine"),
    }

    # Shared TF-IDF settings (word n-grams)
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        lowercase=True,
    )

    # -----------------------------
    # 4) Train + Evaluate each model
    # -----------------------------
    for name, model in models.items():
        clf = Pipeline([
            ("tfidf", tfidf),
            ("clf", model),
        ])

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        acc = accuracy_score(y_test, pred)

        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)
        print(f"Accuracy: {acc:.3f}")
        print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
        print("\nReport:\n", classification_report(y_test, pred, digits=3))

        print("\n--- SAMPLE PREDICTIONS ---")
        for t, yt, yp in zip(X_test, y_test, pred):
            print(f"true={yt} pred={yp} | {t}")


if __name__ == "__main__":
    main()
