# analysis/plot_nested_results.py
#
# Run:
#   python -m analysis.plot_nested_results tfidf_360_nested_results.csv
#
# or:
#   python -m analysis.plot_nested_results bert_360_nested_results.csv
#
# Custom output:
#   python -m analysis.plot_nested_results results.csv --output myplot.png
#
# Plots:
#   - distribution of TEST f1_macro
#   - mean
#   - median
#   - standard deviation band
#

from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "csv_file",
        help="CSV file from nested 360 experiment"
    )

    parser.add_argument(
        "--metric",
        default="test_f1_macro",
        help="Metric column (default=test_f1_macro)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output plot filename (default: derived from CSV name)"
    )

    args = parser.parse_args()

    print("Reading:", args.csv_file)

    # Auto-generate output filename if not provided
    if args.output is None:
        base = os.path.splitext(args.csv_file)[0]
        args.output = base + ".png"

    df = pd.read_csv(args.csv_file)

    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not in CSV columns")

    scores = df[args.metric].dropna().values

    print("\nNumber of splits:", len(scores))

    mean_val = np.mean(scores)
    median_val = np.median(scores)
    std_val = np.std(scores)
    min_val = np.min(scores)
    max_val = np.max(scores)

    print("\nStatistics:")
    print("Mean   :", round(mean_val, 4))
    print("Median :", round(median_val, 4))
    print("Std    :", round(std_val, 4))
    print("Min    :", round(min_val, 4))
    print("Max    :", round(max_val, 4))

    plt.figure(figsize=(10, 6))

    plt.hist(scores, bins=20)
    plt.xlim(0, 1)

    # Mean line
    plt.axvline(
        mean_val,
        linestyle="--",
        label=f"Mean = {mean_val:.3f}"
    )

    # Median line
    plt.axvline(
        median_val,
        linestyle="-",
        label=f"Median = {median_val:.3f}"
    )

    # Std band
    plt.axvspan(
        mean_val - std_val,
        mean_val + std_val,
        alpha=0.2,
        label=f"±1 std = {std_val:.3f}"
    )

    plt.xlabel(args.metric)
    plt.ylabel("Number of splits")

    plt.title("Distribution of Test Scores (Nested CV)")

    plt.legend()

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=300)

    print("\nSaved plot:", args.output)


if __name__ == "__main__":
    main()
