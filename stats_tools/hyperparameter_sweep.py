from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np

from stats_tools.audit_advanced_time import (
    analyze_file as analyze_advanced_time_file,
    compute_pairwise_time_distances,
)
from stats_tools.audit_field_analyzer import (
    QUOTED_OR_BARE_VALUE_PATTERN,
    analyze_file_pairs as analyze_audit_field_pairs,
    get_mode_config,
)
from stats_tools.complexity_metrics import METRIC_KEYS, score_file_pairs as score_complexity_pairs
from stats_tools.data_catalog import get_log_path
from stats_tools.inter_event_time import score_file_pairs as score_inter_event_pairs
from stats_tools.one_gram import score_file_pairs as score_one_gram_pairs
from stats_tools.pairwise_group_stats import (
    infer_binary_groups,
    mannwhitney_with_effect_size,
    silhouette_summary,
    summarize_vector,
)
from stats_tools.pairwise_viz import build_symmetric_distance_matrix, group_labels_humans_first
from stats_tools.statistic_evaluation_csv import append_statistic_evaluation_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "hyperparameter_sweep_statistic_evaluation.csv"


def audit_file_pairs() -> list[tuple[str, str]]:
    return [
        ("GPT4.1", str(get_log_path("GPT4.1", "audit"))),
        ("Hotti", str(get_log_path("Hotti", "audit"))),
        ("GPT4.1_V2", str(get_log_path("GPT4.1_V2", "audit"))),
        ("GPT4o", str(get_log_path("GPT4o", "audit"))),
        ("Armin", str(get_log_path("Armin", "audit"))),
        ("Marvin", str(get_log_path("Marvin", "audit"))),
        ("Benni", str(get_log_path("Benni", "audit"))),
        ("Nico", str(get_log_path("Nico", "audit"))),
        ("Torina", str(get_log_path("Torina", "audit"))),
        ("GPT5", str(get_log_path("GPT5", "audit"))),
    ]


def syslog_file_pairs() -> list[tuple[str, str]]:
    return [
        ("GPT4.1", str(get_log_path("GPT4.1", "syslog"))),
        ("Hotti", str(get_log_path("Hotti", "syslog"))),
        ("GPT4.1_V2", str(get_log_path("GPT4.1_V2", "syslog"))),
        ("GPT4o", str(get_log_path("GPT4o", "syslog"))),
        ("Armin", str(get_log_path("Armin", "syslog"))),
        ("Marvin", str(get_log_path("Marvin", "syslog"))),
        ("Benni", str(get_log_path("Benni", "syslog"))),
        ("Nico", str(get_log_path("Nico", "syslog"))),
        ("Torina", str(get_log_path("Torina", "syslog"))),
        ("GPT5", str(get_log_path("GPT5", "syslog"))),
    ]


def build_group_stats(
    ordered_labels: list[str],
    pairwise_results: list[Any],
    extract_pair: Callable[[Any], tuple[str, str, float]],
) -> dict[str, Any]:
    groups = infer_binary_groups(ordered_labels)
    vectors = {
        "ai_ai": [],
        "human_human": [],
        "ai_human": [],
    }
    finite_results: list[Any] = []

    for item in pairwise_results:
        label_1, label_2, distance = extract_pair(item)
        if distance != distance:
            continue

        finite_results.append(item)
        group_1 = groups[label_1]
        group_2 = groups[label_2]
        if group_1 == "ai" and group_2 == "ai":
            vectors["ai_ai"].append(distance)
        elif group_1 == "human" and group_2 == "human":
            vectors["human_human"].append(distance)
        else:
            vectors["ai_human"].append(distance)

    arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in vectors.items()
    }
    summaries = {
        name: summarize_vector(values)
        for name, values in arrays.items()
    }
    tests = {
        "ai_human_vs_ai_ai": mannwhitney_with_effect_size(arrays["ai_human"], arrays["ai_ai"]),
        "ai_human_vs_human_human": mannwhitney_with_effect_size(arrays["ai_human"], arrays["human_human"]),
    }

    silhouette = {
        "overall_mean": float("nan"),
        "ai_mean": float("nan"),
        "human_mean": float("nan"),
    }
    if len(finite_results) == len(pairwise_results):
        matrix = build_symmetric_distance_matrix(
            ordered_labels,
            finite_results,
            extract_pair=extract_pair,
        )
        silhouette = silhouette_summary(matrix, ordered_labels)

    return {
        "group_summaries": summaries,
        "silhouette": silhouette,
        "mannwhitney": tests,
    }


def append_result(
    *,
    output_path: Path,
    approach: str,
    distance_name: str,
    ordered_labels: list[str],
    hyperparameters: dict[str, Any],
    group_stats: dict[str, Any],
) -> None:
    append_statistic_evaluation_row(
        approach=approach,
        distance_name=distance_name,
        ordered_labels=ordered_labels,
        hyperparameters=hyperparameters,
        group_stats=group_stats,
        output_path=output_path,
    )


def run_inter_event_time(output_path: Path) -> int:
    file_pairs = audit_file_pairs()
    labels = [label for label, _ in file_pairs]
    files = [path for _, path in file_pairs]
    label_by_path = {path: label for label, path in file_pairs}
    ordered_labels = group_labels_humans_first(labels)
    count = 0

    for bins, log_bins in product([10, 25, 50], [True, False]):
        results = score_inter_event_pairs(files, bins=bins, log_bins=log_bins)
        for sort_by in ["l1", "ks"]:
            metric_index = 2 if sort_by == "l1" else 3
            group_stats = build_group_stats(
                ordered_labels,
                results,
                extract_pair=lambda item, idx=metric_index: (
                    label_by_path[item[0]],
                    label_by_path[item[1]],
                    item[idx],
                ),
            )
            append_result(
                output_path=output_path,
                approach="inter_event_time",
                distance_name=sort_by,
                ordered_labels=ordered_labels,
                hyperparameters={
                    "bins": bins,
                    "log_bins": log_bins,
                },
                group_stats=group_stats,
            )
            count += 1

    return count


def run_one_gram(output_path: Path) -> int:
    file_pairs = audit_file_pairs()
    labels = [label for label, _ in file_pairs]
    files = [path for _, path in file_pairs]
    label_by_path = {path: label for label, path in file_pairs}
    ordered_labels = group_labels_humans_first(labels)
    count = 0

    for mode in ["word", "char"]:
        results = score_one_gram_pairs(files, mode=mode, min_count=1)
        for sort_by in ["l1", "js"]:
            metric_index = 2 if sort_by == "l1" else 3
            group_stats = build_group_stats(
                ordered_labels,
                results,
                extract_pair=lambda item, idx=metric_index: (
                    label_by_path[item[0]],
                    label_by_path[item[1]],
                    item[idx],
                ),
            )
            append_result(
                output_path=output_path,
                approach="one_gram",
                distance_name=sort_by,
                ordered_labels=ordered_labels,
                hyperparameters={"mode": mode},
                group_stats=group_stats,
            )
            count += 1

    return count


def run_complexity_metrics(output_path: Path) -> int:
    file_pairs = syslog_file_pairs()
    ordered_labels = group_labels_humans_first([label for label, _ in file_pairs])
    count = 0

    for window_size in [5, 10, 20, 30]:
        stride_candidates = []
        for stride in [1, 2, 5, window_size // 2]:
            if stride > 0 and stride not in stride_candidates:
                stride_candidates.append(stride)

        for stride in stride_candidates:
            results = score_complexity_pairs(
                file_pairs,
                window_size=window_size,
                stride=stride,
                preprocess_mode="soft",
                drain_ini_path=None,
            )
            for sort_by in METRIC_KEYS:
                group_stats = build_group_stats(
                    ordered_labels,
                    results,
                    extract_pair=lambda item, key=sort_by: (
                        item["label_1"],
                        item["label_2"],
                        item["distances"][key],
                    ),
                )
                append_result(
                    output_path=output_path,
                    approach="complexity_metrics",
                    distance_name=sort_by,
                    ordered_labels=ordered_labels,
                    hyperparameters={
                        "window_size": window_size,
                        "stride": stride,
                    },
                    group_stats=group_stats,
                )
                count += 1

    return count


def run_audit_field_analyzer(output_path: Path) -> int:
    file_pairs = audit_file_pairs()
    labels = [label for label, _ in file_pairs]
    ordered_labels = group_labels_humans_first(labels)
    count = 0

    for mode in ["execve", "path", "syscall", "sockaddr"]:
        prefix, keys = get_mode_config(mode)
        results = analyze_audit_field_pairs(
            file_pairs,
            prefix_regex=prefix,
            keys=keys,
            value_pattern=QUOTED_OR_BARE_VALUE_PATTERN,
            require_all_keys=False,
            ignore_case=False,
            strip_quoted_values=True,
            top_k=30,
        )
        for sort_by in ["average_js_distance", *keys]:
            group_stats = build_group_stats(
                ordered_labels,
                results,
                extract_pair=lambda item, key=sort_by: (
                    item["label_1"],
                    item["label_2"],
                    item["average_js_distance"]
                    if key == "average_js_distance"
                    else item["js_distances_per_key"][key],
                ),
            )
            append_result(
                output_path=output_path,
                approach="audit_field_analyzer",
                distance_name=sort_by,
                ordered_labels=ordered_labels,
                hyperparameters={"mode": mode},
                group_stats=group_stats,
            )
            count += 1

    return count


def build_advanced_time_pairwise_results(
    file_pairs: list[tuple[str, str]],
    *,
    cluster_window: float,
    cmd: str,
    bins_all: int,
    bins_cmd: int,
    log_bins_all: bool,
    log_bins_cmd: bool,
    min_n_cmd: int,
) -> list[dict[str, Any]]:
    labels = [label for label, _ in file_pairs]
    analyzed = {
        label: analyze_advanced_time_file(path, cluster_window=cluster_window, cmd=cmd)
        for label, path in file_pairs
    }

    results: list[dict[str, Any]] = []
    for left_idx, label_1 in enumerate(labels):
        for label_2 in labels[left_idx + 1:]:
            distances = compute_pairwise_time_distances(
                analyzed[label_1],
                analyzed[label_2],
                bins_all=bins_all,
                bins_cmd=bins_cmd,
                log_bins_all=log_bins_all,
                log_bins_cmd=log_bins_cmd,
                min_n_cmd=min_n_cmd,
            )
            results.append(
                {
                    "label_1": label_1,
                    "label_2": label_2,
                    "distances": distances,
                }
            )

    return results


def run_audit_advanced_time(output_path: Path) -> int:
    file_pairs = audit_file_pairs()
    labels = [label for label, _ in file_pairs]
    ordered_labels = group_labels_humans_first(labels)
    count = 0
    sort_mapping = {
        "all_l1": lambda item: item["distances"]["all"]["l1"],
        "all_ks": lambda item: item["distances"]["all"]["ks"],
        "cmd_l1": lambda item: item["distances"]["cmd"]["l1"],
        "cmd_ks": lambda item: item["distances"]["cmd"]["ks"],
    }

    for cluster_window, cmd in product([0.5, 1, 2, 5], ["ls", "curl", "cat", "tail", "grep", "chmod"]):
        for bins_all, log_bins_all, log_bins_cmd in product([10, 25, 50], [True, False], [True, False]):
            results = build_advanced_time_pairwise_results(
                file_pairs,
                cluster_window=cluster_window,
                cmd=cmd,
                bins_all=bins_all,
                bins_cmd=bins_all,
                log_bins_all=log_bins_all,
                log_bins_cmd=log_bins_cmd,
                min_n_cmd=10,
            )
            for sort_by in ["all_l1", "all_ks", "cmd_l1", "cmd_ks"]:
                group_stats = build_group_stats(
                    ordered_labels,
                    results,
                    extract_pair=lambda item, key=sort_by: (
                        item["label_1"],
                        item["label_2"],
                        sort_mapping[key](item),
                    ),
                )
                append_result(
                    output_path=output_path,
                    approach="audit_advanced_time",
                    distance_name=sort_by,
                    ordered_labels=ordered_labels,
                    hyperparameters={
                        "cluster_window": cluster_window,
                        "cmd": cmd,
                        "bins_all": bins_all,
                        "bins_cmd": bins_all,
                        "log_bins_all": log_bins_all,
                        "log_bins_cmd": log_bins_cmd,
                    },
                    group_stats=group_stats,
                )
                count += 1

    return count


def main() -> None:
    output_path = DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    counts = {
        "inter_event_time": run_inter_event_time(output_path),
        "one_gram": run_one_gram(output_path),
        "complexity_metrics": run_complexity_metrics(output_path),
        "audit_field_analyzer": run_audit_field_analyzer(output_path),
        "audit_advanced_time": run_audit_advanced_time(output_path),
    }
    total = sum(counts.values())

    print(f"Wrote {total} rows to {output_path}")
    for approach, count in counts.items():
        print(f"  {approach}: {count}")


if __name__ == "__main__":
    main()
