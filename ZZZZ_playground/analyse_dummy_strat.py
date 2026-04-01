import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

# -----------------------------
# Settings
# -----------------------------
N_SAMPLES = 500
N_RUNS = 50
TEST_SIZE = 0.2
RANDOM_SEED = 42

HUMAN_RATIO = 0.7   # <-- configure dataset imbalance here

OUTPUT_FILE = "svm_vs_dummy_hist.png"

rng = np.random.default_rng(RANDOM_SEED)

svm_scores = []
dummy_scores = []

# -----------------------------
# Repeat experiment
# -----------------------------
for run in range(N_RUNS):

    n_human = int(N_SAMPLES * HUMAN_RATIO)
    n_ai = N_SAMPLES - n_human

    # generate feature values
    X_human = rng.uniform(0, 1, size=(n_human, 1))
    X_ai = rng.uniform(-1, 0, size=(n_ai, 1))

    y_human = np.array(["human"] * n_human)
    y_ai = np.array(["ai"] * n_ai)

    X = np.vstack([X_human, X_ai])
    y = np.concatenate([y_human, y_ai])

    # shuffle dataset
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        #stratify=y,
        random_state=run
    )

    # -----------------------------
    # SVM
    # -----------------------------
    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)

    y_pred_svm = svm.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, average="macro")
    svm_scores.append(f1_svm)

    # -----------------------------
    # Dummy
    # -----------------------------
    dummy = DummyClassifier(strategy="stratified", random_state=run)
    dummy.fit(X_train, y_train)

    y_pred_dummy = dummy.predict(X_test)
    f1_dummy = f1_score(y_test, y_pred_dummy, average="macro")
    dummy_scores.append(f1_dummy)

# -----------------------------
# Print summary
# -----------------------------
print(f"SVM mean F1_macro:   {np.mean(svm_scores):.4f}")
print(f"Dummy mean F1_macro: {np.mean(dummy_scores):.4f}")

# -----------------------------
# Plot histogram
# -----------------------------
svm_mean = np.mean(svm_scores)
dummy_mean = np.mean(dummy_scores)

plt.figure(figsize=(10, 6))

plt.hist(svm_scores, bins=20, alpha=0.6, label="SVM")
plt.hist(dummy_scores, bins=20, alpha=0.6, label="Dummy Stratified")

# vertical mean lines
plt.axvline(svm_mean, linestyle="--", linewidth=2,
            label=f"SVM mean = {svm_mean:.3f}")

plt.axvline(dummy_mean, linestyle="--", linewidth=2,
            label=f"Dummy mean = {dummy_mean:.3f}")

plt.xlim(0, 1)
plt.xlabel("F1_macro")
plt.ylabel("Frequency")
plt.title("Distribution of F1_macro over 50 runs")

plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()