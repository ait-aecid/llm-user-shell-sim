# experiments/cnn_search.py
# Run: python -m experiments.cnn_search

from __future__ import annotations

import numpy as np

from core.loader import load_examples, LoadConfig
from core.splits import make_splits
from core.reporting import print_split_stats

from cnn_pipeline.pipeline import Candidate, CNNConfig, search


def main():
    # ---- Load real dataset via core.loader ----
    examples = load_examples(
        LoadConfig(
            log_files=("audit.log",),         # ONLY audit (adjust if you want more)
            prefix_with_log_type=False,
            preprocess_mode="soft",           # CNN usually fine with soft; try aggressive too
            window_mode="cids",               # one line = one example
            window_size=50,
            window_stride=25,
            cid_prefix="CID",
            # For speed / more context, consider windows:
            # window_mode="lines",
            # window_size=50,
            # window_stride=25,
            # join_token=" <EOL> ",
            max_lines_per_file=200,
        )
    )

    y = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    # ---- Use predefined group split (matches your other experiments) ----
    split = make_splits(
        y,
        groups=groups,
        val_groups=["Marvin", "GPT4.1"],
        test_groups=["Hotti", "GPT4.1_V2"],
    )

    print_split_stats(examples, split)

    candidates = []
    for lr in [1e-3, 3e-4, 1e-4]:
        for num_filters in [64, 128]:
            for dropout in [0.3, 0.5]:
                candidates.append(
                    Candidate(
                        cfg=CNNConfig(
                            lr=lr,
                            weight_decay=1e-4,
                            embed_dim=64,
                            num_filters=num_filters,
                            fc_dim=128,
                            dropout=dropout,
                            kernel_sizes=(3, 5, 7),
                            batch_size=32,
                            epochs=12,
                            early_stopping=True,
                            patience=3,
                            eval_every=1,
                            max_len_cap=512,
                            len_percentile=95.0,
                            seed=42,
                        )
                    )
                )


    best, best_val, best_test, all_val = search(
        examples,
        split,
        candidates,
        metric="f1_macro",
        evaluate_test_for_all=False,
        verbose=True,   # progress bars + status prints (from your updated cnn pipeline)
    )

    print("\nBest config:")
    print(best.cfg)
    print(f"VAL f1_macro:  {best_val.f1_macro:.4f}")
    print(f"TEST f1_macro: {best_test.f1_macro:.4f}")


if __name__ == "__main__":
    main()
