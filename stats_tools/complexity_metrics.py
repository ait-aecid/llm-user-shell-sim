from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from textwrap import shorten
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

from core.loader import (
    LoadConfig,
    _assign_templates_and_cids_global,
    _create_template_miner,
    _get_drain_ini,
    _infer_log_type,
    _preprocess_line,
    _read_lines,
)
from stats_tools.data_catalog import analysis_actors, get_log_path
from stats_tools.pairwise_group_stats import (
    analyze_binary_group_structure,
    print_binary_group_structure_report,
)
from stats_tools.pairwise_viz import (
    build_symmetric_distance_matrix,
    group_labels_humans_first,
    plot_distance_heatmap,
    plot_mds_embedding,
)
from stats_tools.statistic_evaluation_csv import append_statistic_evaluation_row


METRIC_KEYS = (
    "gini",
    "kurtosis",
    "entropy",
    "mad",
    "gini_seq",
    "kurtosis_seq",
    "entropy_seq",
    "mad_seq",
    "l1",
    "l1_seq",
    "js",
    "js_seq",
)


def load_preprocessed_lines(
    file_path: str | Path,
    *,
    preprocess_mode: str = "template",
) -> list[str]:
    path = Path(file_path)
    log_type = _infer_log_type(path.name)

    return [
        _preprocess_line(
            line.rstrip("\n"),
            mode=preprocess_mode,
            assumed_type=log_type,
        )
        for _, line in _read_lines(
            path,
            encoding="utf-8",
            errors="replace",
            max_lines=None,
        )
        if line.rstrip("\n")
    ]


def sliding_windows(ids: list[int], window_size: int, stride: int) -> list[tuple[int, ...]]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    return [
        tuple(ids[start:start + window_size])
        for start in range(0, len(ids) - window_size + 1, stride)
    ]


def gini_from_counts(counter: Counter) -> float:
    if not counter:
        return float("nan")
    x = np.array(sorted(counter.values()), dtype=float)
    n = x.size
    s = x.sum()
    if s <= 0 or n == 0:
        return float("nan")
    return float(2.0 * (np.arange(1, n + 1) * x).sum() / (n * s) - (n + 1) / n)


def kurtosis_from_counts(counter: Counter, *, convexify: bool = True) -> float:
    freq = list(counter.values())
    if len(freq) == 0:
        return float("nan")

    if convexify:
        asc = sorted(freq)
        desc = sorted(freq, reverse=True)
        x = np.asarray(asc + desc, dtype=float)
    else:
        x = np.asarray(freq, dtype=float)

    n = x.size
    if n < 4:
        return float("nan")

    mean = x.mean()
    d = x - mean
    s2 = (d @ d) / (n - 1)
    if s2 == 0:
        return float("nan")
    s4 = s2 ** 2
    m4 = np.sum(d ** 4)

    g2 = (
        (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * (m4 / s4)
        - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    )
    return float(g2)


def entropy_from_counts(counter: Counter, *, base: float = 2.0) -> float:
    if not counter:
        return float("nan")
    x = np.array(list(counter.values()), dtype=float)
    total = x.sum()
    if total <= 0:
        return float("nan")
    p = x / total
    p = p[p > 0]
    if p.size == 0:
        return float("nan")

    if base == 2.0:
        return float(-np.sum(p * np.log2(p)))
    if base == np.e:
        return float(-np.sum(p * np.log(p)))
    return float(-np.sum(p * (np.log(p) / np.log(base))))


def mad_from_counts(counter: Counter) -> float:
    if not counter:
        return float("nan")
    x = np.array(list(counter.values()), dtype=float)
    if x.size == 0:
        return float("nan")
    mean = x.mean()
    return float(np.mean(np.abs(x - mean)))


def counter_to_probs(counter: Counter, vocab: list[Any]) -> np.ndarray:
    arr = np.array([counter[item] for item in vocab], dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def l1_distance_from_counters(counter1: Counter, counter2: Counter) -> float:
    vocab = sorted(set(counter1) | set(counter2), key=str)
    if not vocab:
        return float("nan")
    p1 = counter_to_probs(counter1, vocab)
    p2 = counter_to_probs(counter2, vocab)
    return float(np.sum(np.abs(p1 - p2)))


def js_distance_from_counters(counter1: Counter, counter2: Counter) -> float:
    vocab = sorted(set(counter1) | set(counter2), key=str)
    if not vocab:
        return float("nan")
    p1 = counter_to_probs(counter1, vocab)
    p2 = counter_to_probs(counter2, vocab)
    if p1.sum() == 0 or p2.sum() == 0:
        return float("nan")
    return float(jensenshannon(p1, p2))


def stats_from_ids(
    ids: list[int],
    *,
    min_ids: int = 20,
    min_unique_ids: int = 3,
) -> dict[str, float]:
    if len(ids) < min_ids:
        return {
            "gini": float("nan"),
            "kurtosis": float("nan"),
            "entropy": float("nan"),
            "mad": float("nan"),
        }

    cnt = Counter(ids)

    if len(cnt) < min_unique_ids:
        kurtosis_val = float("nan")
    else:
        kurtosis_val = kurtosis_from_counts(cnt, convexify=True)

    return {
        "gini": gini_from_counts(cnt),
        "kurtosis": kurtosis_val,
        "entropy": entropy_from_counts(cnt, base=2.0),
        "mad": mad_from_counts(cnt),
    }


def stats_from_windows(
    ids: list[int],
    window_size: int,
    stride: int,
    *,
    min_windows: int = 20,
    min_unique_windows: int = 4,
) -> dict[str, float]:
    wins = sliding_windows(ids, window_size, stride)

    if len(wins) < min_windows:
        return {
            "gini_seq": float("nan"),
            "kurtosis_seq": float("nan"),
            "entropy_seq": float("nan"),
            "mad_seq": float("nan"),
        }

    cnt = Counter(wins)

    if len(cnt) < min_unique_windows:
        kurtosis_seq = float("nan")
    else:
        kurtosis_seq = kurtosis_from_counts(cnt, convexify=True)

    return {
        "gini_seq": gini_from_counts(cnt),
        "kurtosis_seq": kurtosis_seq,
        "entropy_seq": entropy_from_counts(cnt, base=2.0),
        "mad_seq": mad_from_counts(cnt),
    }


def complexity_metrics_from_lines(
    lines1: list[str],
    lines2: list[str],
    *,
    window_size: int,
    stride: int,
    drain_ini_path: Optional[str] = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[int],
    list[int],
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    dict[int, str],
]:
    cfg = LoadConfig(drain_ini_path=drain_ini_path)
    miner = _create_template_miner(ini_path=_get_drain_ini(cfg))

    assigned_templates, cluster_ids = _assign_templates_and_cids_global(miner, lines1 + lines2)

    cid_to_template: dict[int, str] = {}
    for template, cid in zip(assigned_templates, cluster_ids):
        if cid not in cid_to_template:
            cid_to_template[cid] = str(template)

    n1 = len(lines1)
    cids1 = cluster_ids[:n1]
    cids2 = cluster_ids[n1:]

    seqs1 = sliding_windows(cids1, window_size, stride) if len(cids1) >= window_size else []
    seqs2 = sliding_windows(cids2, window_size, stride) if len(cids2) >= window_size else []

    m1 = stats_from_ids(cids1)
    m2 = stats_from_ids(cids2)

    if seqs1:
        m1.update(stats_from_windows(cids1, window_size, stride))
    else:
        m1.update({k: float("nan") for k in ("gini_seq", "kurtosis_seq", "entropy_seq", "mad_seq")})

    if seqs2:
        m2.update(stats_from_windows(cids2, window_size, stride))
    else:
        m2.update({k: float("nan") for k in ("gini_seq", "kurtosis_seq", "entropy_seq", "mad_seq")})

    return m1, m2, cids1, cids2, seqs1, seqs2, cid_to_template


def compute_pairwise_metric_differences(
    file_1: str | Path,
    file_2: str | Path,
    *,
    window_size: int,
    stride: int,
    preprocess_mode: str = "template",
    drain_ini_path: Optional[str] = None,
) -> dict[str, Any]:
    lines1 = load_preprocessed_lines(file_1, preprocess_mode=preprocess_mode)
    lines2 = load_preprocessed_lines(file_2, preprocess_mode=preprocess_mode)

    metrics1, metrics2, cids1, cids2, seqs1, seqs2, cid_to_template = complexity_metrics_from_lines(
        lines1,
        lines2,
        window_size=window_size,
        stride=stride,
        drain_ini_path=drain_ini_path,
    )

    cid_counter_1 = Counter(cids1)
    cid_counter_2 = Counter(cids2)
    seq_counter_1 = Counter(seqs1)
    seq_counter_2 = Counter(seqs2)

    distances = {
        key: abs(metrics1[key] - metrics2[key])
        for key in (
            "gini",
            "kurtosis",
            "entropy",
            "mad",
            "gini_seq",
            "kurtosis_seq",
            "entropy_seq",
            "mad_seq",
        )
    }

    distances["l1"] = l1_distance_from_counters(cid_counter_1, cid_counter_2)
    distances["js"] = js_distance_from_counters(cid_counter_1, cid_counter_2)

    if seqs1 and seqs2:
        distances["l1_seq"] = l1_distance_from_counters(seq_counter_1, seq_counter_2)
        distances["js_seq"] = js_distance_from_counters(seq_counter_1, seq_counter_2)
    else:
        distances["l1_seq"] = float("nan")
        distances["js_seq"] = float("nan")

    return {
        "file_1": str(file_1),
        "file_2": str(file_2),
        "metrics_1": metrics1,
        "metrics_2": metrics2,
        "distances": distances,
        "n_cids_1": len(cids1),
        "n_cids_2": len(cids2),
        "cids_1": cids1,
        "cids_2": cids2,
        "seqs_1": seqs1,
        "seqs_2": seqs2,
        "cid_counter_1": cid_counter_1,
        "cid_counter_2": cid_counter_2,
        "seq_counter_1": seq_counter_1,
        "seq_counter_2": seq_counter_2,
        "cid_to_template": cid_to_template,
    }


def score_file_pairs(
    file_pairs: list[tuple[str, str | Path]],
    *,
    window_size: int,
    stride: int,
    preprocess_mode: str = "template",
    drain_ini_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for (label_1, file_1), (label_2, file_2) in combinations(file_pairs, 2):
        result = compute_pairwise_metric_differences(
            file_1,
            file_2,
            window_size=window_size,
            stride=stride,
            preprocess_mode=preprocess_mode,
            drain_ini_path=drain_ini_path,
        )
        result["label_1"] = label_1
        result["label_2"] = label_2
        results.append(result)

    return results


def print_pairwise_distance_report(
    pairwise_results: list[dict[str, Any]],
    *,
    sort_by: str,
) -> None:
    if sort_by not in METRIC_KEYS:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    print(f"\nPairwise complexity distances sorted by {sort_by}:")
    for item in sorted(pairwise_results, key=lambda x: x["distances"][sort_by], reverse=True):
        distances = ", ".join(
            f"{key}={item['distances'][key]:.6f}"
            for key in METRIC_KEYS
        )
        print(f"{item['label_1']} vs {item['label_2']} | {distances}")


def format_seq_label(seq: tuple[int, ...]) -> str:
    return "(" + ",".join(map(str, seq)) + ")"


def format_template_description(cid: int, template: str, max_len: int = 120) -> str:
    short_template = shorten(template.replace("\n", " "), width=max_len, placeholder=" ...")
    return f"CID {cid}: {short_template}"


def plot_counter_distributions(
    counter1: Counter,
    counter2: Counter,
    *,
    label1: str,
    label2: str,
    title: str,
    top_k: int = 20,
    min_count: int = 1,
    sort_by: str = "absdiff",
    value_mode: str = "count",
    item_formatter: Optional[callable] = None,
    item_descriptions: Optional[dict[Any, str]] = None,
    show_descriptions_below: bool = False,
) -> list[Any]:
    pooled = counter1 + counter2
    vocab = [item for item, c in pooled.items() if c >= min_count]

    if not vocab:
        print("No items satisfy min_count; nothing to plot.")
        return []

    if value_mode == "count":
        v1 = np.array([counter1[item] for item in vocab], dtype=float)
        v2 = np.array([counter2[item] for item in vocab], dtype=float)
    elif value_mode == "relative":
        v1 = counter_to_probs(counter1, vocab)
        v2 = counter_to_probs(counter2, vocab)
    else:
        raise ValueError("value_mode must be 'count' or 'relative'")

    diff = v1 - v2
    rows = list(zip(vocab, v1, v2, diff))

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

    if item_formatter is None:
        formatted_labels = [str(x) for x in labels]
    else:
        formatted_labels = [item_formatter(x) for x in labels]

    y = np.arange(len(formatted_labels))
    height = 0.38

    extra_bottom = 0.0
    if show_descriptions_below and item_descriptions:
        extra_bottom = min(0.45, 0.02 * len(labels) + 0.08)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(formatted_labels) + 3 * extra_bottom)))

    ax.barh(y - height / 2, vals1, height=height, label=label1)
    ax.barh(y + height / 2, vals2, height=height, label=label2)

    ax.set_yticks(y)
    ax.set_yticklabels(formatted_labels)
    ax.set_xlabel("Count" if value_mode == "count" else "Relative frequency")
    ax.set_title(title)
    ax.legend()

    if show_descriptions_below and item_descriptions:
        description_lines = [
            item_descriptions[item]
            for item in labels[::-1]
            if item in item_descriptions
        ]
        if description_lines:
            text_block = "\n".join(description_lines)
            fig.subplots_adjust(bottom=max(0.18, extra_bottom))
            fig.text(
                0.01,
                0.01,
                text_block,
                ha="left",
                va="bottom",
                fontsize=8,
                family="monospace",
            )
        else:
            fig.tight_layout()
    else:
        fig.tight_layout()

    plt.show()
    return labels[::-1]


def plot_pair_template_distributions(
    file_1: str | Path,
    file_2: str | Path,
    *,
    label_1: str,
    label_2: str,
    window_size: int,
    stride: int,
    preprocess_mode: str = "template",
    drain_ini_path: Optional[str] = None,
    top_k_ids: int = 20,
    top_k_seqs: int = 20,
    min_count: int = 1,
    sort_by: str = "absdiff",
    value_mode: str = "count",
    show_ids: bool = True,
    show_sequences: bool = True,
    show_templates_below_ids: bool = True,
    template_text_width: int = 120,
) -> dict[str, Any]:
    lines1 = load_preprocessed_lines(file_1, preprocess_mode=preprocess_mode)
    lines2 = load_preprocessed_lines(file_2, preprocess_mode=preprocess_mode)

    _, _, cids1, cids2, seqs1, seqs2, cid_to_template = complexity_metrics_from_lines(
        lines1,
        lines2,
        window_size=window_size,
        stride=stride,
        drain_ini_path=drain_ini_path,
    )

    result: dict[str, Any] = {
        "cids_1": cids1,
        "cids_2": cids2,
        "cid_counter_1": Counter(cids1),
        "cid_counter_2": Counter(cids2),
        "seq_counter_1": Counter(seqs1),
        "seq_counter_2": Counter(seqs2),
        "cid_to_template": cid_to_template,
    }

    if show_ids:
        cid_descriptions = {
            cid: format_template_description(cid, template, max_len=template_text_width)
            for cid, template in cid_to_template.items()
        }

        plot_counter_distributions(
            result["cid_counter_1"],
            result["cid_counter_2"],
            label1=label_1,
            label2=label_2,
            title=f"Template-ID distributions\n{label_1} vs {label_2}",
            top_k=top_k_ids,
            min_count=min_count,
            sort_by=sort_by,
            value_mode=value_mode,
            item_formatter=lambda cid: f"CID {cid}",
            item_descriptions=cid_descriptions,
            show_descriptions_below=show_templates_below_ids,
        )

    if show_sequences:
        if len(seqs1) == 0 or len(seqs2) == 0:
            print(
                f"Not enough CIDs to build sequence windows for {label_1} or {label_2} "
                f"(window_size={window_size}, stride={stride})."
            )
        else:
            plot_counter_distributions(
                result["seq_counter_1"],
                result["seq_counter_2"],
                label1=label_1,
                label2=label_2,
                title=(
                    f"Template-ID sequence distributions\n"
                    f"{label_1} vs {label_2} | window_size={window_size}, stride={stride}"
                ),
                top_k=top_k_seqs,
                min_count=min_count,
                sort_by=sort_by,
                value_mode=value_mode,
                item_formatter=format_seq_label,
                item_descriptions=None,
                show_descriptions_below=False,
            )

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Complexity metric evaluation")

    parser.add_argument(
        "--use_data_wp",
        action="store_true",
        help="Use Data_WP instead of Data",
    )

    parser.add_argument(
        "--assignment_mode",
        type=str,
        default="true",
        choices=["true", "random_stratified", "indexed_stratified"],
        help="How to assign actors to AI/human groups",
    )

    parser.add_argument(
        "--log_type",
        type=str,
        default=None,
        choices=["audit", "syslog", "nextcloud"],
        help="Restrict evaluation to a single log type. If omitted, use the default search space.",
    )

    parser.add_argument(
        "--assignment_idx",
        type=int,
        default=None,
        help="Index for indexed_stratified assignment mode",
    )

    return parser.parse_args()

def build_assignment_suffix(assignment_mode: str, assignment_idx: int | None) -> str:
    idx_part = f"{assignment_idx}_idx" if assignment_idx is not None else "null_idx"
    return f"{assignment_mode}_{idx_part}"

'''
if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"
    HEATMAP_title = "Complexity heatmap"
    MDS_title = "Complexity mds"

    # Example:
    # score=10.800000 | distance=kurtosis_seq | hyperparameters: log_type=audit, stride=1, window_size=10

    # Hyperparameters
    window_size = 5
    stride = 5
    sort_by = "gini_seq"
    log_type = "audit"

    # do not change unless intended
    preprocess_mode = "soft"
    drain_ini_path = None

    # Optional pair-distribution plotting
    enable_pair_distribution_plot = True
    #pair_plot_labels = ("GPT4.1", "GPT5")
    #pair_plot_labels = ("Armin", "GPT4.1_V3")
    pair_plot_labels = ("Marvin", "GPT4.1_V3")
    pair_plot_top_k_ids = 5
    pair_plot_top_k_seqs = 10
    pair_plot_min_count = 1
    pair_plot_sort_by = "sum"          # absdiff, file1, file2, sum
    pair_plot_value_mode = "relative"         # count, relative
    pair_plot_show_ids = True
    pair_plot_show_sequences = True
    pair_plot_show_templates_below_ids = True
    pair_plot_template_text_width = 120

    file_pairs = [
        (actor, str(get_log_path(actor, log_type, dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    path_by_label = {label: path for label, path in file_pairs}

    pairwise_results = score_file_pairs(
        file_pairs,
        window_size=window_size,
        stride=stride,
        preprocess_mode=preprocess_mode,
        drain_ini_path=drain_ini_path,
    )
    print_pairwise_distance_report(pairwise_results, sort_by=sort_by)

    ordered_labels = group_labels_humans_first([label for label, _ in file_pairs])
    matrix = build_symmetric_distance_matrix(
        ordered_labels,
        pairwise_results,
        extract_pair=lambda item: (
            item["label_1"],
            item["label_2"],
            item["distances"][sort_by],
        ),
    )

    # optionally:
    rng = np.random.default_rng(42)

    group_stats = analyze_binary_group_structure(
        matrix,
        ordered_labels,
        assignment_mode="random_stratified",
        rng=rng,
    )

    #group_stats = analyze_binary_group_structure(matrix, ordered_labels)
    print_binary_group_structure_report(group_stats)

    plot_distance_heatmap(matrix, ordered_labels, title="", anonymize_humans=False)
    plot_mds_embedding(matrix, ordered_labels, title="")

    if enable_pair_distribution_plot and pair_plot_labels is not None:
        label_1, label_2 = pair_plot_labels
        if label_1 not in path_by_label or label_2 not in path_by_label:
            raise ValueError(
                f"Could not find selected actors {pair_plot_labels}. "
                f"Available labels: {sorted(path_by_label.keys())}"
            )

        plot_pair_template_distributions(
            path_by_label[label_1],
            path_by_label[label_2],
            label_1=label_1,
            label_2=label_2,
            window_size=window_size,
            stride=stride,
            preprocess_mode=preprocess_mode,
            drain_ini_path=drain_ini_path,
            top_k_ids=pair_plot_top_k_ids,
            top_k_seqs=pair_plot_top_k_seqs,
            min_count=pair_plot_min_count,
            sort_by=pair_plot_sort_by,
            value_mode=pair_plot_value_mode,
            show_ids=pair_plot_show_ids,
            show_sequences=pair_plot_show_sequences,
            show_templates_below_ids=pair_plot_show_templates_below_ids,
            template_text_width=pair_plot_template_text_width,
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
        f"results/statistic_complexity_metrics"
        f"{'_wp' if use_data_wp else ''}"
        f"_{selected_log_type}"
        f"_{assignment_suffix}.csv"
    )

    preprocess_mode = "soft"
    drain_ini_path = None

    # hyperparameter search space
    log_types = [selected_log_type] if selected_log_type is not None else ["audit", "syslog"]

    configs = list(product(
        log_types,
        [5, 10, 25],
        [1, 2, 5],
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

    for i, (log_type, window_size, stride) in enumerate(configs, 1):
        print(
            f"\n[CONFIG {i}/{len(configs)}] "
            f"log_type={log_type} "
            f"window_size={window_size} "
            f"stride={stride}"
        )

        file_pairs = [
            (user, str(get_log_path(user, log_type, dataset=dataset)))
            for user in analysis_actors(dataset)
        ]

        pairwise_results = score_file_pairs(
            file_pairs,
            window_size=window_size,
            stride=stride,
            preprocess_mode=preprocess_mode,
            drain_ini_path=drain_ini_path,
        )

        ordered_labels = group_labels_humans_first(
            [label for label, _ in file_pairs]
        )

        for metric_name in METRIC_KEYS:
            print(f"\n--- Evaluating metric={metric_name} ---")

            print_pairwise_distance_report(pairwise_results, sort_by=metric_name)

            invalid_pairs = [
                item for item in pairwise_results
                if not np.isfinite(item["distances"][metric_name])
            ]

            if invalid_pairs:
                invalid_labels = sorted({
                    item["label_1"] for item in invalid_pairs
                } | {
                    item["label_2"] for item in invalid_pairs
                })

                print(
                    f"skipped metric={metric_name} for "
                    f"log_type={log_type}, window_size={window_size}, stride={stride} "
                    f"(non-finite distances; affected labels={invalid_labels})"
                )
                continue

            matrix = build_symmetric_distance_matrix(
                ordered_labels,
                pairwise_results,
                extract_pair=lambda item, metric_name=metric_name: (
                    item["label_1"],
                    item["label_2"],
                    item["distances"][metric_name],
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
                approach="complexity_metrics",
                distance_name=metric_name,
                ordered_labels=ordered_labels,
                hyperparameters={
                    "log_type": log_type,
                    "window_size": window_size,
                    "stride": stride,
                },
                group_stats=group_stats,
                output_path=output_path,
            )
