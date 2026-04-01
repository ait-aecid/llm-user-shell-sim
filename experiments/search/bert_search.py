# experiments/bert_search.py
# Run from project root:
#   python -m experiments.bert_search

from __future__ import annotations

import numpy as np

from core.loader import load_examples, LoadConfig
from core.splits import make_splits
from core.reporting import print_split_stats

from bert_pipeline.pipeline import Candidate, TransformerConfig, search


def main():
    # ---- Load real dataset via core.loader ----
    examples = load_examples(
        LoadConfig(
            log_files=("audit.log",),         # ONLY audit (adjust if you want more)
            prefix_with_log_type=False,
            preprocess_mode="soft",           # BERT usually benefits from keeping more surface form than aggressive
            # window_mode="none",               # one line = one example (switch to "lines" for speed)
            # window_size=50,
            # window_stride=25,
            # join_token=" <EOL> ",
            # If you want Drain3 CID windows with BERT (works, but it's basically token sequences like "CID12"):
            window_mode="cids",
            window_size=30,
            window_stride=15,
            cid_prefix="CID",
            max_lines_per_file=200,
        )
    )

    y = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    # ---- Use your predefined group split (recommended for your setup) ----
    split = make_splits(
        y,
        groups=groups,
        val_groups=["Armin", "GPT4.1"],
        test_groups=["Benni", "GPT4.1_V2"],
    )

    print_split_stats(examples, split)

    # ---- Candidate grid (keep small; BERT tuning is expensive) ----
    candidates = []

    for lr in [1e-5, 2e-5, 5e-5]:
        for max_len in [128, 256]:
            for wd in [0.0, 0.01]:
                candidates.append(
                    Candidate(
                        cfg=TransformerConfig(
                            model_name="bert-base-uncased",
                            max_length=max_len,
                            batch_size=8,     # increase if GPU allows
                            lr=lr,
                            epochs=4,         # early stopping will cut earlier
                            weight_decay=wd,
                            warmup_ratio=0.1,
                            seed=42,
                            patience=1
                        )
                    )
                )

    best, best_val, best_test, all_val = search(
        examples,
        split,
        candidates,
        metric="f1_macro",
        evaluate_test_for_all=False,  # recommended
        verbose=True,                 # shows progress bars + status prints
    )

    print("\nBest config:")
    print(best.cfg)
    print(f"Best VAL f1_macro:  {best_val.f1_macro:.4f}")
    print(f"Best TEST f1_macro: {best_test.f1_macro:.4f}")

    # Optional: show full per-class report for the best config
    print("\n=== Best VAL classification report ===")
    print(best_val.per_class_report)
    print("Confusion matrix (val):")
    print(best_val.confusion)


if __name__ == "__main__":
    main()
