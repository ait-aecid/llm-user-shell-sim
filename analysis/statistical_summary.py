#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import pandas as pd


def _bootstrap_mean_ci(
    x: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = x[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate one nested-CV CSV for a selected metric."
    )
    p.add_argument("csv_file", help="Path to results CSV.")
    p.add_argument(
        "--metric",
        default="test_f1_macro",
        help="Metric column to summarize (default: test_f1_macro).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for confidence interval (default: 0.05 -> 95%% CI).",
    )
    p.add_argument(
        "--n_boot",
        type=int,
        default=10000,
        help="Bootstrap iterations for CI (default: 10000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv_file)
    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not found in CSV.")

    x = df[args.metric].dropna().astype(float).to_numpy()
    if x.size == 0:
        raise ValueError(f"Metric '{args.metric}' has no non-NaN values.")

    mean_v = float(np.mean(x))
    median_v = float(np.median(x))
    std_v = float(np.std(x))
    p05_v = float(np.quantile(x, 0.05))
    ci_lo, ci_hi = _bootstrap_mean_ci(
        x,
        n_boot=int(args.n_boot),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )

    print(f"CSV              : {args.csv_file}")
    print(f"Metric           : {args.metric}")
    print(f"N (non-NaN)      : {x.size}")
    print(f"Mean             : {mean_v:.6f}")
    print(f"Median           : {median_v:.6f}")
    print(f"Std              : {std_v:.6f}")
    print(f"5th percentile   : {p05_v:.6f}")
    print(f"{int((1.0 - args.alpha) * 100)}% CI (mean)  : [{ci_lo:.6f}, {ci_hi:.6f}]")


if __name__ == "__main__":
    main()

