from pathlib import Path
from itertools import combinations, product

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from stats_tools.data_catalog import analysis_actors, get_log_path
from stats_tools.pairwise_group_stats import analyze_binary_group_structure, print_binary_group_structure_report
from stats_tools.statistic_evaluation_csv import append_statistic_evaluation_row
from stats_tools.pairwise_viz import (
    build_symmetric_distance_matrix,
    group_labels_humans_first,
    plot_distance_heatmap,
    plot_mds_embedding,
)

from core.loader import (
    _infer_log_type,
    _read_lines,
    _extract_timestamps,
    _inter_event_diffs_seconds,
)


def get_inter_event_times(file_path: str) -> np.ndarray:
    path = Path(file_path)
    log_type = _infer_log_type(path.name)

    raw_lines = [line for _, line in _read_lines(
        path,
        encoding="utf-8",
        errors="replace",
        max_lines=None,
    )]

    timestamps = _extract_timestamps(raw_lines, assumed_type=log_type)
    diffs = _inter_event_diffs_seconds(timestamps)
    return diffs


def make_shared_bins(a: np.ndarray, b: np.ndarray, bins: int = 30, log_bins: bool = False) -> np.ndarray:
    combined = np.concatenate([a, b])

    if len(combined) == 0:
        raise ValueError("No inter-event times found.")

    if log_bins:
        combined = combined[combined > 0]
        if len(combined) == 0:
            raise ValueError("No positive inter-event times for log bins.")
        return np.logspace(np.log10(combined.min()), np.log10(combined.max()), bins + 1)

    return np.linspace(combined.min(), combined.max(), bins + 1)


def histogram_probs(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def l1_bin_score(a: np.ndarray, b: np.ndarray, bins: int = 30, log_bins: bool = False):
    edges = make_shared_bins(a, b, bins=bins, log_bins=log_bins)
    p_a = histogram_probs(a, edges)
    p_b = histogram_probs(b, edges)
    diff = p_a - p_b
    score = np.sum(np.abs(diff))
    return edges, diff, score


def ks_score(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(ks_2samp(a, b).statistic)


def plot_bin_diff(file1: str, file2: str, bins: int = 30, log_bins: bool = False):
    a = get_inter_event_times(file1)
    b = get_inter_event_times(file2)

    if len(a) == 0 or len(b) == 0:
        raise RuntimeError("One file has no inter-event times.")

    edges, diff, l1 = l1_bin_score(a, b, bins=bins, log_bins=log_bins)
    ks = ks_score(a, b)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    plt.figure(figsize=(10, 5))
    plt.bar(centers, diff, width=widths, align="center")
    plt.axhline(0)

    if log_bins:
        plt.xscale("log")

    plt.xlabel("Inter-event time (s)")
    plt.ylabel("Bin probability difference (file1 - file2)")
    plt.title(f"Inter-event histogram difference\nL1={l1:.4f}, KS={ks:.4f}")
    plt.tight_layout()
    plt.show()

    return l1, ks


def score_file_pairs(files: list[str], bins: int = 30, log_bins: bool = False):
    cached = {f: get_inter_event_times(f) for f in files}
    results = []

    for f1, f2 in combinations(files, 2):
        a = cached[f1]
        b = cached[f2]

        if len(a) == 0 or len(b) == 0:
            results.append((f1, f2, float("nan"), float("nan")))
            continue

        _, _, l1 = l1_bin_score(a, b, bins=bins, log_bins=log_bins)
        ks = ks_score(a, b)
        results.append((f1, f2, l1, ks))

    return results


def plot_selected_inter_event_probabilities(
    file_pairs: list[tuple[str, str]],
    selected_actors: list[str],
    bins: int = 30,
    log_bins: bool = False,
):
    selected_file_pairs = [(label, path) for label, path in file_pairs if label in selected_actors]

    data = []
    for label, path in selected_file_pairs:
        diffs = get_inter_event_times(path)
        if log_bins:
            diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            data.append((label, diffs))

    if not data:
        raise ValueError("No usable inter-event times for selected actors.")

    combined = np.concatenate([x for _, x in data])
    if log_bins:
        edges = np.logspace(np.log10(combined.min()), np.log10(combined.max()), bins + 1)
    else:
        edges = np.linspace(combined.min(), combined.max(), bins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.figure(figsize=(10, 5))
    for label, diffs in data:
        probs = histogram_probs(diffs, edges)
        plt.plot(centers, probs, marker="o", label=label)

    if log_bins:
        plt.xscale("log")

    plt.xlabel("Inter-event time (s)")
    plt.ylabel("Bin probability")
    plt.title("Inter-event time distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"

    # 1. score=4.100000 | distance=l1 | hyperparameters: bins=10, log_bins=True, log_type=nextcloud

    # 2. score=4.600000 | distance=ks | hyperparameters: bins=10, log_bins=True, log_type=syslog

    # 3. score=4.700000 | distance=l1 | hyperparameters: bins=10, log_bins=True, log_type=syslog

    # 5. score=5.700000 | distance=l1 | hyperparameters: bins=25, log_bins=True, log_type=syslog

    ### Hyperparameter
    bins = 25
    log_bins = True
    sort_by = "ks"
    log_type = "audit"

    # choose actors here
    selected_actors = ["Armin", "Marvin", "Nico", "Hotti", "GPT4.1", "GPT4.1_V2", "GPT5", "GPT4o"]

    file_pairs = [
        (actor, str(get_log_path(actor, log_type, dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]
    labels = [label for label, _ in file_pairs]
    files = [path for _, path in file_pairs]

    # plot only the selected actors
    plot_selected_inter_event_probabilities(
        file_pairs=file_pairs,
        selected_actors=selected_actors,
        bins=bins,
        log_bins=log_bins,
    )

    label_by_path = {path: label for label, path in file_pairs}
    results = score_file_pairs(files, bins=bins, log_bins=log_bins)

    print("\nPairwise scores:")
    sort_index = 2 if sort_by == "l1" else 3
    for f1, f2, l1, ks in sorted(results, key=lambda x: x[sort_index], reverse=True):
        print(f"{label_by_path[f1]} vs {label_by_path[f2]} | L1={l1:.6f} | KS={ks:.6f}")

    ordered_labels = group_labels_humans_first(labels)
    metric_index = 2 if sort_by == "l1" else 3
    matrix = build_symmetric_distance_matrix(
        ordered_labels,
        results,
        extract_pair=lambda item: (label_by_path[item[0]], label_by_path[item[1]], item[metric_index]),
    )
    group_stats = analyze_binary_group_structure(matrix, ordered_labels)
    print_binary_group_structure_report(group_stats)

    metric_label = sort_by.upper()
    plot_distance_heatmap(matrix, ordered_labels, title=f"Inter-event {metric_label} heatmap")
    plot_mds_embedding(matrix, ordered_labels, title=f"Inter-event {metric_label} MDS")
    

'''
if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"
    output_path = f"results/statistic_inter_event_time{'_wp' if use_data_wp else ''}.csv"

    for bins, log_bins, sort_by, log_type in product([10, 25, 50], [True, False], ["ks", "l1"], ["audit", "syslog"]): # removed "nextcloud" for wp

        print(
            f"\n===== log_type={log_type} "
            f"bins={bins} "
            f"log_bins={log_bins} "
            f"metric={sort_by} ====="
        )

        file_pairs = [
            (actor, str(get_log_path(actor, log_type, dataset=dataset)))
            for actor in analysis_actors(dataset)
        ]

        labels = [label for label, _ in file_pairs]
        files = [path for _, path in file_pairs]


        label_by_path = {path: label for label, path in file_pairs}
        results = score_file_pairs(files, bins=bins, log_bins=log_bins)

        print("\nPairwise scores:")
        sort_index = 2 if sort_by == "l1" else 3
        for f1, f2, l1, ks in sorted(results, key=lambda x: x[sort_index], reverse=True):
            print(f"{label_by_path[f1]} vs {label_by_path[f2]} | L1={l1:.6f} | KS={ks:.6f}")

        ordered_labels = group_labels_humans_first(labels)
        metric_index = 2 if sort_by == "l1" else 3
        matrix = build_symmetric_distance_matrix(
            ordered_labels,
            results,
            extract_pair=lambda item: (label_by_path[item[0]], label_by_path[item[1]], item[metric_index]),
        )
        group_stats = analyze_binary_group_structure(matrix, ordered_labels)
        print_binary_group_structure_report(group_stats)
        append_statistic_evaluation_row(
            approach="inter_event_time",
            distance_name=sort_by,
            ordered_labels=ordered_labels,
            hyperparameters={
                "bins": bins,
                "log_bins": log_bins,
                "log_type": log_type,
            },
            group_stats=group_stats,
            output_path=output_path,
        )
'''
