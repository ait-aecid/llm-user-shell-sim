#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from typing import List, Sequence, Tuple

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


def _parse_key_cols(raw: str) -> List[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("At least one key column is required.")
    return cols


def _sign_test_two_sided(nonzero_deltas: np.ndarray) -> Tuple[int, int, int, float]:
    pos = int(np.sum(nonzero_deltas > 0))
    neg = int(np.sum(nonzero_deltas < 0))
    n = pos + neg
    if n == 0:
        return pos, neg, n, float("nan")

    k = min(pos, neg)
    # exact two-sided p-value under Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p_two = min(1.0, 2.0 * tail)
    return pos, neg, n, float(p_two)


def _optional_wilcoxon(nonzero_deltas: np.ndarray) -> Tuple[float, float] | None:
    if nonzero_deltas.size == 0:
        return None
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception:
        return None

    stat, p = wilcoxon(nonzero_deltas, zero_method="wilcox", alternative="two-sided")
    return float(stat), float(p)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Paired significance test between two nested-CV CSVs."
    )
    p.add_argument("csv_a", help="Results CSV A (reference).")
    p.add_argument("csv_b", help="Results CSV B (compared against A).")
    p.add_argument(
        "--metric",
        default="test_f1_macro",
        help="Metric column to compare (default: test_f1_macro).",
    )
    p.add_argument(
        "--key_cols",
        default="outer_i",
        help="Comma-separated key columns for pairing rows (default: outer_i).",
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
    p.add_argument(
        "--tsv",
        action="store_true",
        help=(
            "Print a single TSV line for easy sorting: "
            "model_csv mean_delta median_delta std_delta ci_lo ci_hi wins losses ties pairs p_sign p_wilcoxon"
        ),
    )
    args = p.parse_args()

    key_cols: Sequence[str] = _parse_key_cols(args.key_cols)
    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)

    for c in key_cols:
        if c not in df_a.columns or c not in df_b.columns:
            raise ValueError(f"Key column '{c}' must exist in both CSVs.")
    if args.metric not in df_a.columns or args.metric not in df_b.columns:
        raise ValueError(f"Metric '{args.metric}' must exist in both CSVs.")

    a = df_a[list(key_cols) + [args.metric]].rename(columns={args.metric: "metric_a"})
    b = df_b[list(key_cols) + [args.metric]].rename(columns={args.metric: "metric_b"})

    if a.duplicated(list(key_cols)).any():
        raise ValueError("CSV A has duplicate key rows; choose stricter key columns.")
    if b.duplicated(list(key_cols)).any():
        raise ValueError("CSV B has duplicate key rows; choose stricter key columns.")

    merged = a.merge(b, on=list(key_cols), how="inner")
    if merged.empty:
        raise ValueError("No paired rows after merge. Check key columns.")

    merged = merged.dropna(subset=["metric_a", "metric_b"]).copy()
    if merged.empty:
        raise ValueError("No paired non-NaN metric values after merge.")

    # Delta = A - B
    merged["delta"] = merged["metric_a"].astype(float) - merged["metric_b"].astype(float)
    d = merged["delta"].to_numpy(dtype=float)
    d_nonzero = d[d != 0.0]

    mean_delta = float(np.mean(d))
    median_delta = float(np.median(d))
    std_delta = float(np.std(d))

    ci_lo, ci_hi = _bootstrap_mean_ci(
        d,
        n_boot=int(args.n_boot),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )

    pos, neg, n_sign, p_sign = _sign_test_two_sided(d_nonzero)
    wilcoxon_res = _optional_wilcoxon(d_nonzero)

    wins = int(np.sum(d > 0))
    losses = int(np.sum(d < 0))
    ties = int(np.sum(d == 0))
    pairs = int(d.size)

    p_w = float("nan")
    if wilcoxon_res is not None:
        _, p_w = wilcoxon_res

    if args.tsv:
        # One machine-readable line (TSV) so bash can sort by mean_delta easily.
        model_name = os.path.basename(args.csv_a)
        print(
            "\t".join(
                [
                    model_name,
                    f"{mean_delta:.6f}",
                    f"{median_delta:.6f}",
                    f"{std_delta:.6f}",
                    f"{ci_lo:.6f}",
                    f"{ci_hi:.6f}",
                    str(wins),
                    str(losses),
                    str(ties),
                    str(pairs),
                    "nan" if math.isnan(p_sign) else f"{p_sign:.6g}",
                    "nan" if math.isnan(p_w) else f"{p_w:.6g}",
                ]
            )
        )
        return

    # Human-readable output (original style)
    print(f"CSV A               : {args.csv_a}")
    print(f"CSV B               : {args.csv_b}")
    print(f"Metric              : {args.metric}")
    print(f"Pair keys           : {', '.join(key_cols)}")
    print(f"Pairs used          : {pairs}")
    print(f"Mean delta (A-B)    : {mean_delta:.6f}")
    print(f"Median delta (A-B)  : {median_delta:.6f}")
    print(f"Std delta           : {std_delta:.6f}")
    print(
        f"{int((1.0 - args.alpha) * 100)}% CI (mean delta): [{ci_lo:.6f}, {ci_hi:.6f}]"
    )
    print(f"Wins / Losses / Ties: {wins} / {losses} / {ties}")
    print(f"Sign test + / - / n : {pos} / {neg} / {n_sign}")
    if math.isnan(p_sign):
        print("Sign test p-value    : nan (all ties)")
    else:
        print(f"Sign test p-value    : {p_sign:.6g}")

    if wilcoxon_res is None:
        print("Wilcoxon             : unavailable (scipy missing) or all ties")
    else:
        stat, p_w2 = wilcoxon_res
        print(f"Wilcoxon W, p-value  : {stat:.6f}, {p_w2:.6g}")

    print("\nInterpretation:")
    print("- If CI excludes 0 and p-value is small, difference is likely real.")
    print("- If CI includes 0, evidence for a meaningful difference is weak.")


if __name__ == "__main__":
    main()