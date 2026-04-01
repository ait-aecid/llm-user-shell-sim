# experiments/tfidf_search.py

from __future__ import annotations

import numpy as np

from core.loader import load_examples as load_real_examples, LoadConfig
from core.splits import make_splits
from core.reporting import print_split_stats
from tfidf.pipeline import Candidate, VectorizerConfig, search


def main():
    # ---- Load your real dataset (same loader you use elsewhere) ----
    examples = load_real_examples(
        LoadConfig(
            # pick what you want here:
            log_files=("audit.log",),          # e.g. only audit
            prefix_with_log_type=False,
            preprocess_mode="soft",
            window_mode="none",
            window_size=None,
            window_stride=None,
            cid_prefix="CID",
            max_lines_per_file=200,

            # or keep defaults by omitting fields
        )
    )

    y = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    # ---- Split (you can keep random group split, or use predefined groups) ----

    split = make_splits(
        y,
        groups=groups,
        val_groups=["Armin", "GPT4.1"],
        test_groups=["Benni", "GPT4.1_V2"],
    )


    print_split_stats(examples, split)

    # ---- Define search space ----
    candidates = []

    # -------------------------
    # 1) Character TF-IDF (usually best for logs)
    # -------------------------
    for ngram in [(3, 5), (4, 6), (5, 7)]:          # common sweet spots for char ngrams
        for min_df in [1, 2, 5]:
            vec_cfg = VectorizerConfig(
                analyzer="char",
                ngram_range=ngram,
                min_df=min_df,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=False,                    # logs are often case-informative
                max_features=None,
            )

            # Linear SVM
            for C in [0.1, 1.0, 3.0, 10.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="svm",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            # Logistic Regression
            for C in [0.1, 1.0, 3.0, 10.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="logreg",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            # SGD hinge
            for alpha in [1e-6, 1e-5, 1e-4]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="sgd_hinge",
                    model_params={
                        "alpha": alpha,
                        "class_weight": "balanced",
                        "max_iter": 2000,
                        "tol": 1e-3,
                    },
                ))

            # Ridge classifier
            for alpha in [0.1, 1.0, 10.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="ridge",
                    model_params={"alpha": alpha},
                ))

    # -------------------------
    # 2) Word TF-IDF (can help when keywords matter)
    # -------------------------
    for ngram in [(1, 1), (1, 2)]:
        for min_df in [1, 2]:
            vec_cfg = VectorizerConfig(
                analyzer="word",
                ngram_range=ngram,
                min_df=min_df,
                max_df=0.95,
                sublinear_tf=True,
                lowercase=True,
                max_features=200_000,               # cap vocab for word model
            )

            for C in [0.3, 1.0, 3.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="logreg",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            for C in [0.3, 1.0, 3.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="svm",
                    model_params={"C": C, "class_weight": "balanced"},
                ))

            # ComplementNB often decent on text + imbalance
            for alpha in [0.01, 0.1, 1.0]:
                candidates.append(Candidate(
                    vectorizer=vec_cfg,
                    model_name="cnb",
                    model_params={"alpha": alpha},
                ))


    # ---- Run search ----
    best, best_val, best_test, all_val = search(
        examples,
        split,
        candidates,
        metric="f1_macro",
    )

    print("\nBest configuration:")
    print(best)
    print("Best VAL F1:", best_val.f1_macro)
    print("Best TEST F1:", best_test.f1_macro)


if __name__ == "__main__":
    main()
