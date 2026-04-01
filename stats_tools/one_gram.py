from pathlib import Path
from itertools import combinations, product
from collections import Counter
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

from stats_tools.data_catalog import analysis_actors, get_log_path
from stats_tools.pairwise_group_stats import (
    analyze_binary_group_structure,
    print_binary_group_structure_report,
)
from stats_tools.statistic_evaluation_csv import append_statistic_evaluation_row
from stats_tools.pairwise_viz import (
    build_symmetric_distance_matrix,
    group_labels_humans_first,
    plot_distance_heatmap,
    plot_mds_embedding,
)


def load_lines(file_path: str) -> list[str]:
    return Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()


def tokenize(lines: list[str], mode: str = "word") -> list[str]:
    if mode == "char":
        return list("\n".join(lines))
    elif mode == "word":
        tokens = []
        for line in lines:
            tokens.extend(line.split())
        return tokens
    else:
        raise ValueError("mode must be 'word' or 'char'")


def get_token_counts(file_path: str, mode: str = "word") -> Counter:
    lines = load_lines(file_path)
    toks = tokenize(lines, mode=mode)
    return Counter(toks)


def counter_to_probs(cnt: Counter, vocab: list[str]) -> np.ndarray:
    arr = np.array([cnt[tok] for tok in vocab], dtype=float)
    total = arr.sum()
    if total == 0:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def js_score_from_counters(cnt1: Counter, cnt2: Counter, vocab: list[str]) -> float:
    p = counter_to_probs(cnt1, vocab)
    q = counter_to_probs(cnt2, vocab)

    if p.sum() == 0 or q.sum() == 0:
        return float("nan")

    return float(jensenshannon(p, q))


def compute_diff_from_counters(
    cnt1: Counter,
    cnt2: Counter,
    *,
    top_k: int = 20,
    min_count: int = 1,
):
    total1 = sum(cnt1.values())
    total2 = sum(cnt2.values())

    if total1 == 0 or total2 == 0:
        raise RuntimeError("One file has no tokens.")

    pooled = cnt1 + cnt2
    vocab = [tok for tok, c in pooled.items() if c >= min_count]

    p = counter_to_probs(cnt1, vocab)
    q = counter_to_probs(cnt2, vocab)
    diff = p - q

    rows = []
    for tok, p1, p2, delta in zip(vocab, p, q, diff):
        rows.append((tok, cnt1[tok], cnt2[tok], p1, p2, delta))

    l1_score = float(np.sum(np.abs(diff)))
    js_score = js_score_from_counters(cnt1, cnt2, vocab)

    rows.sort(key=lambda x: abs(x[5]), reverse=True)
    return rows[:top_k], l1_score, js_score


def compute_diff(
    file1: str,
    file2: str,
    mode: str = "word",
    top_k: int = 20,
    min_count: int = 1,
):
    cnt1 = get_token_counts(file1, mode=mode)
    cnt2 = get_token_counts(file2, mode=mode)
    return compute_diff_from_counters(cnt1, cnt2, top_k=top_k, min_count=min_count)


def plot_diff(rows, title: str = "Top 1-gram frequency differences"):
    if not rows:
        print("No differences to plot.")
        return

    labels = [r[0] for r in rows][::-1]
    values = [r[5] for r in rows][::-1]

    plt.figure(figsize=(10, max(4, 0.4 * len(labels))))
    bars = plt.barh(labels, values)
    plt.axvline(0)

    plt.xlabel("Relative frequency difference (file1 - file2)")
    plt.title(title)

    max_abs = max(abs(v) for v in values) if values else 0.001
    offset = max(0.001, 0.02 * max_abs)

    for bar, v in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if x >= 0:
            plt.text(x + offset, y, f"{v:+.4f}", va="center", ha="left")
        else:
            plt.text(x - offset, y, f"{v:+.4f}", va="center", ha="right")

    plt.tight_layout()
    plt.show()


def plot_two_histograms(
    cnt1: Counter,
    cnt2: Counter,
    *,
    label1: str = "file1",
    label2: str = "file2",
    top_k: int = 20,
    min_count: int = 1,
    sort_by: str = "absdiff",
    title: str = "Top 1-gram distributions",
):
    pooled = cnt1 + cnt2
    vocab = [tok for tok, c in pooled.items() if c >= min_count]

    if not vocab:
        print("No tokens satisfy min_count; nothing to plot.")
        return

    p = counter_to_probs(cnt1, vocab)
    q = counter_to_probs(cnt2, vocab)
    diff = p - q

    rows = list(zip(vocab, p, q, diff))

    if sort_by == "absdiff":
        rows.sort(key=lambda x: abs(x[3]), reverse=True)
    elif sort_by == "file1":
        rows.sort(key=lambda x: x[1], reverse=True)
    elif sort_by == "file2":
        rows.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == "sum":
        rows.sort(key=lambda x: x[1] + x[2], reverse=True)
    else:
        raise ValueError("sort_by must be one of: 'absdiff', 'file1', 'file2', 'sum'")

    rows = rows[:top_k]

    labels = [r[0] for r in rows][::-1]
    vals1 = [r[1] for r in rows][::-1]
    vals2 = [r[2] for r in rows][::-1]

    y = np.arange(len(labels))
    height = 0.38

    plt.figure(figsize=(12, max(4, 0.45 * len(labels))))
    plt.barh(y - height / 2, vals1, height=height, label=label1)
    plt.barh(y + height / 2, vals2, height=height, label=label2)

    plt.yticks(y, labels)
    plt.xlabel("Relative frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pair_comparison(
    file1: str,
    file2: str,
    *,
    mode: str = "word",
    top_k: int = 20,
    min_count: int = 1,
    label1: str | None = None,
    label2: str | None = None,
    show_diff: bool = True,
    show_histograms: bool = True,
    histogram_sort_by: str = "absdiff",
):
    cnt1 = get_token_counts(file1, mode=mode)
    cnt2 = get_token_counts(file2, mode=mode)

    rows, l1_score, js_score = compute_diff_from_counters(
        cnt1,
        cnt2,
        top_k=top_k,
        min_count=min_count,
    )

    if label1 is None:
        label1 = Path(file1).name
    if label2 is None:
        label2 = Path(file2).name

    print(f"{label1} vs {label2} | L1={l1_score:.6f} | JS={js_score:.6f}")

    if show_diff:
        plot_diff(
            rows,
            title=f"Top 1-gram differences\n{label1} vs {label2} | L1={l1_score:.4f}, JS={js_score:.4f}",
        )

    if show_histograms:
        plot_two_histograms(
            cnt1,
            cnt2,
            label1=label1,
            label2=label2,
            top_k=top_k,
            min_count=min_count,
            sort_by=histogram_sort_by,
            title=f"Top 1-gram distributions\n{label1} vs {label2}",
        )

    return rows, l1_score, js_score


def score_file_pairs(
    files: list[str],
    *,
    mode: str = "word",
    min_count: int = 1,
):
    cached = {f: get_token_counts(f, mode=mode) for f in files}
    results = []

    for f1, f2 in combinations(files, 2):
        cnt1 = cached[f1]
        cnt2 = cached[f2]

        total1 = sum(cnt1.values())
        total2 = sum(cnt2.values())

        if total1 == 0 or total2 == 0:
            results.append((f1, f2, float("nan"), float("nan")))
            continue

        _, l1_score, js_score = compute_diff_from_counters(
            cnt1,
            cnt2,
            top_k=20,  # irrelevant for pair scoring
            min_count=min_count,
        )
        results.append((f1, f2, l1_score, js_score))

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="1-gram pairwise analysis")

    parser.add_argument(
        "--use_data_wp",
        action="store_true",
        help="Use Data_WP instead of Data",
    )

    parser.add_argument(
        "--log_type",
        type=str,
        default=None,
        choices=["audit", "syslog", "nextcloud"],
        help="Restrict to a single log type. If omitted, use the default grid.",
    )

    parser.add_argument(
        "--assignment_mode",
        type=str,
        default="true",
        choices=["true", "random_stratified", "indexed_stratified"],
        help="How actors are assigned to ai/human groups",
    )

    parser.add_argument(
        "--assignment_idx",
        type=int,
        default=None,
        help="Assignment index for assignment_mode='indexed_stratified'",
    )

    return parser.parse_args()

def build_assignment_suffix(assignment_mode: str, assignment_idx: int | None) -> str:
    idx_part = f"{assignment_idx}_idx" if assignment_idx is not None else "null_idx"
    return f"{assignment_mode}_{idx_part}"

'''
if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"

    # 1. score=1.000000 | distance=l1 | hyperparameters: log_type=syslog, min_count=1, mode=char
    # 2. score=2.200000 | distance=js | hyperparameters: log_type=syslog, min_count=1, mode=char
    # 3. score=4.700000 | distance=l1 | hyperparameters: log_type=audit, min_count=1, mode=char
    # 4. score=6.100000 | distance=l1 | hyperparameters: log_type=nextcloud, min_count=1, mode=word
    # 5. score=6.200000 | distance=l1 | hyperparameters: log_type=audit, min_count=1, mode=word

    # Hyperparameters
    mode = "char"          # "word" for lexical content, "char" for surface style/noise patterns
    sort_by = "l1"         # "l1" or "js"
    log_type = "syslog"
    min_count = 1          # do not change unless you explicitly want to filter rare grams

    # Optional pairwise plotting controls
    enable_pair_plot = True
    pair_plot_labels = ("Armin", "GPT4.1_V3")   # actor labels to compare visually; set to None to disable lookup
    pair_plot_top_k = 100
    pair_plot_show_diff = True
    pair_plot_show_histograms = True
    pair_plot_hist_sort_by = "absdiff"   # absdiff, file1, file2, sum

    file_pairs = [
        (actor, str(get_log_path(actor, log_type, dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]
    labels = [label for label, _ in file_pairs]
    files = [path for _, path in file_pairs]
    label_by_path = {path: label for label, path in file_pairs}
    path_by_label = {label: path for label, path in file_pairs}

    results = score_file_pairs(files, mode=mode, min_count=min_count)

    print("\nPairwise scores:")
    sort_index = 2 if sort_by == "l1" else 3
    for f1, f2, l1, js in sorted(results, key=lambda x: x[sort_index], reverse=True):
        print(f"{label_by_path[f1]} vs {label_by_path[f2]} | L1={l1:.6f} | JS={js:.6f}")

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
    plot_distance_heatmap(matrix, ordered_labels, title="", anonymize_humans=False)
    plot_mds_embedding(matrix, ordered_labels, title="")

    if enable_pair_plot and pair_plot_labels is not None:
        actor1, actor2 = pair_plot_labels
        if actor1 not in path_by_label or actor2 not in path_by_label:
            raise ValueError(
                f"Could not find selected actors {pair_plot_labels}. "
                f"Available labels: {sorted(path_by_label.keys())}"
            )

        plot_pair_comparison(
            path_by_label[actor1],
            path_by_label[actor2],
            mode=mode,
            top_k=pair_plot_top_k,
            min_count=min_count,
            label1=actor1,
            label2=actor2,
            show_diff=pair_plot_show_diff,
            show_histograms=pair_plot_show_histograms,
            histogram_sort_by=pair_plot_hist_sort_by,
        )
'''


if __name__ == "__main__":
    args = parse_args()

    use_data_wp = args.use_data_wp
    dataset = "Data_WP" if use_data_wp else "Data"

    assignment_mode = args.assignment_mode
    assignment_idx = args.assignment_idx
    selected_log_type = args.log_type

    if assignment_mode == "indexed_stratified" and assignment_idx is None:
        raise ValueError(
            "assignment_mode='indexed_stratified' requires --assignment_idx"
        )

    assignment_suffix = build_assignment_suffix(assignment_mode, assignment_idx)
    output_path = (
        f"results/statistic_one_gram"
        f"{'_wp' if use_data_wp else ''}"
        f"_{selected_log_type}"
        f"_{assignment_suffix}.csv"
    )

    min_count = 1

    # hyperparameter grid
    if selected_log_type is not None:
        log_types = [selected_log_type]
    else:
        log_types = ["audit", "syslog"] if use_data_wp else ["audit", "syslog", "nextcloud"]

    configs = list(product(
        log_types,
        ["word", "char"],
        ["l1", "js"],
    ))

    # runtime config print
    print(
        f"\nRunning with: "
        f"use_data_wp={use_data_wp}, "
        f"log_type={selected_log_type}, "
        f"assignment_mode={assignment_mode}, "
        f"assignment_idx={assignment_idx}, "
        f"output_path={output_path}"
    )

    for i, (log_type, mode, sort_by) in enumerate(configs, 1):
        print(
            f"\n[{i}/{len(configs)}] "
            f"log_type={log_type} "
            f"mode={mode} "
            f"metric={sort_by}"
        )

        file_pairs = [
            (actor, str(get_log_path(actor, log_type, dataset=dataset)))
            for actor in analysis_actors(dataset)
        ]

        labels = [label for label, _ in file_pairs]
        files = [path for _, path in file_pairs]
        label_by_path = {path: label for label, path in file_pairs}

        results = score_file_pairs(
            files,
            mode=mode,
            min_count=min_count,
        )

        print("\nPairwise scores:")
        sort_index = 2 if sort_by == "l1" else 3
        for f1, f2, l1, js in sorted(results, key=lambda x: x[sort_index], reverse=True):
            print(f"{label_by_path[f1]} vs {label_by_path[f2]} | L1={l1:.6f} | JS={js:.6f}")

        ordered_labels = group_labels_humans_first(labels)
        metric_index = 2 if sort_by == "l1" else 3

        matrix = build_symmetric_distance_matrix(
            ordered_labels,
            results,
            extract_pair=lambda item: (
                label_by_path[item[0]],
                label_by_path[item[1]],
                item[metric_index],
            ),
        )

        group_stats = analyze_binary_group_structure(
            matrix,
            ordered_labels,
            assignment_mode=assignment_mode,
            assignment_idx=assignment_idx,
        )
        print_binary_group_structure_report(group_stats)

        append_statistic_evaluation_row(
            approach="one_gram",
            distance_name=sort_by,
            ordered_labels=ordered_labels,
            hyperparameters={
                "log_type": log_type,
                "mode": mode,
                "min_count": min_count,
                "assignment_mode": assignment_mode,
                "assignment_idx": assignment_idx,
            },
            group_stats=group_stats,
            output_path=output_path,
        )
