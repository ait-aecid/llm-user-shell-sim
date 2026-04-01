#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _load_json_dict(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _is_finite(value: float) -> bool:
    return math.isfinite(value)


def _compute_scores(row: dict[str, str]) -> dict[str, float]:
    ai_ai_mean = _to_float(row.get("ai_ai_mean"))
    ai_human_mean = _to_float(row.get("ai_human_mean"))
    human_human_mean = _to_float(row.get("human_human_mean"))
    silhouette_overall = _to_float(row.get("silhouette_overall_mean"))
    cliffs_ai_ai = _to_float(row.get("mw_ai_human_vs_ai_ai_cliffs_delta"))
    cliffs_human_human = _to_float(row.get("mw_ai_human_vs_human_human_cliffs_delta"))

    mean_separation = float("nan")
    if all(_is_finite(v) for v in (ai_ai_mean, ai_human_mean, human_human_mean)):
        mean_separation = ai_human_mean - max(ai_ai_mean, human_human_mean)

    cliffs_delta_mean = float("nan")
    if all(_is_finite(v) for v in (cliffs_ai_ai, cliffs_human_human)):
        cliffs_delta_mean = (cliffs_ai_ai + cliffs_human_human) / 2.0

    return {
        "mean_separation": mean_separation,
        "silhouette_overall": silhouette_overall,
        "cliffs_delta_mean": cliffs_delta_mean,
    }


def _load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            scores = _compute_scores(raw_row)
            hyperparameters = _load_json_dict(raw_row.get("hyperparameters_json"))
            rows.append(
                {
                    "timestamp_utc": raw_row.get("timestamp_utc", ""),
                    "approach": raw_row.get("approach", ""),
                    "distance_name": raw_row.get("distance_name", ""),
                    "hyperparameters": hyperparameters,
                    "scores": scores,
                    "ai_ai_mean": _to_float(raw_row.get("ai_ai_mean")),
                    "ai_human_mean": _to_float(raw_row.get("ai_human_mean")),
                    "human_human_mean": _to_float(raw_row.get("human_human_mean")),
                    "silhouette_overall_mean": _to_float(raw_row.get("silhouette_overall_mean")),
                    "mw_ai_human_vs_ai_ai_cliffs_delta": _to_float(
                        raw_row.get("mw_ai_human_vs_ai_ai_cliffs_delta")
                    ),
                    "mw_ai_human_vs_human_human_cliffs_delta": _to_float(
                        raw_row.get("mw_ai_human_vs_human_human_cliffs_delta")
                    ),
                    "mw_ai_human_vs_ai_ai_p": _to_float(
                        raw_row.get("mw_ai_human_vs_ai_ai_p")
                    ),
                    "mw_ai_human_vs_human_human_p": _to_float(
                        raw_row.get("mw_ai_human_vs_human_human_p")
                    ),
                }
            )

    return rows


def _format_hyperparameters(hyperparameters: dict[str, Any]) -> str:
    if not hyperparameters:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in sorted(hyperparameters.items()))


def _format_float_or_na(value: float, fmt: str = ".6f") -> str:
    if not _is_finite(value):
        return "nan"
    return format(value, fmt)


def _rank_rows(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = [row for row in rows if _is_finite(row["scores"][metric_name])]
    ranked.sort(key=lambda row: row["scores"][metric_name], reverse=True)
    return ranked[:top_k]


def _print_metric_block(
    rows: list[dict[str, Any]],
    *,
    metric_name: str,
    label: str,
    description: str,
    top_k: int,
) -> None:
    print(f"\n=== {label} ===")
    print(description)

    ranked = _rank_rows(rows, metric_name=metric_name, top_k=top_k)
    if not ranked:
        print("No finite rows for this metric.")
        return

    for index, row in enumerate(ranked, start=1):
        print(
            f"{index}. score={row['scores'][metric_name]:.6f} | "
            f"distance={row['distance_name']} | "
            f"hyperparameters: {_format_hyperparameters(row['hyperparameters'])}"
        )
        print(
            "   "
            f"means: ai_ai={_format_float_or_na(row['ai_ai_mean'])}, "
            f"ai_human={_format_float_or_na(row['ai_human_mean'])}, "
            f"human_human={_format_float_or_na(row['human_human_mean'])}"
        )
        print(
            "   "
            f"silhouette_overall={_format_float_or_na(row['silhouette_overall_mean'])}, "
            f"cliffs=({_format_float_or_na(row['mw_ai_human_vs_ai_ai_cliffs_delta'])}, "
            f"{_format_float_or_na(row['mw_ai_human_vs_human_human_cliffs_delta'])})"
        )
        print(
            "   "
            f"mw_p=({_format_float_or_na(row['mw_ai_human_vs_ai_ai_p'])}, "
            f"{_format_float_or_na(row['mw_ai_human_vs_human_human_p'])})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank hyperparameter configurations from one statistic CSV using "
            "mean separation, silhouette, and Mann-Whitney effect size."
        )
    )
    parser.add_argument("csv_file", help="Path to one statistic CSV file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top configurations to print per metric (default: 5).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}.")

    approach_names = sorted({row["approach"] for row in rows if row["approach"]})

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(rows)}")
    print(f"Approach: {', '.join(approach_names) if approach_names else '-'}")
    print(f"Top k: {args.top_k}")

    _print_metric_block(
        rows,
        metric_name="mean_separation",
        label="Mean Separation",
        description="score = ai_human_mean - (ai_ai_mean + human_human_mean) / 2",
        top_k=args.top_k,
    )
    _print_metric_block(
        rows,
        metric_name="silhouette_overall",
        label="Silhouette",
        description="score = silhouette_overall_mean",
        top_k=args.top_k,
    )
    _print_metric_block(
        rows,
        metric_name="cliffs_delta_mean",
        label="Mann-Whitney Effect Size",
        description=(
            "score = mean of "
            "(mw_ai_human_vs_ai_ai_cliffs_delta, "
            "mw_ai_human_vs_human_human_cliffs_delta)"
        ),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()