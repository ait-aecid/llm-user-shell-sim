import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Pattern, Any
from collections import defaultdict, Counter
from itertools import combinations, product

import matplotlib.pyplot as plt

from stats_tools.data_catalog import analysis_actors, get_log_path
from stats_tools.pairwise_group_stats import analyze_binary_group_structure, print_binary_group_structure_report
from stats_tools.statistic_evaluation_csv import append_statistic_evaluation_row
from stats_tools.pairwise_viz import (
    build_symmetric_distance_matrix,
    group_labels_humans_first,
    plot_distance_heatmap,
    plot_mds_embedding,
)


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------

def extract_kv_from_file(
    file_path: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = r"\S+",
    require_all_keys: bool = True,
    ignore_case: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract key=value values from lines matching prefix_regex.

    Returns:
        {
            "key1": [v1, v2, ...],
            "key2": [v1, v2, ...],
            ...
        }
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    flags = re.IGNORECASE if ignore_case else 0

    key_regexes: Dict[str, Pattern[str]] = {
        k: re.compile(rf"\b{re.escape(k)}=(?P<val>{value_pattern})(?=\s|$)", flags=flags)
        for k in keys
    }

    out: Dict[str, List[str]] = defaultdict(list)

    for line in lines:
        if not prefix_regex.search(line):
            continue

        found: Dict[str, str] = {}

        for k, rx in key_regexes.items():
            m = rx.search(line)
            if m:
                found[k] = m.group("val")

        if require_all_keys and any(k not in found for k in keys):
            continue

        for k in keys:
            if k in found:
                out[k].append(found[k])

    return dict(out)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

QUOTED_OR_BARE_VALUE_PATTERN = r'(?:\"[^"]*\"|\S+)'


def strip_quotes(values: Dict[str, List[str]]) -> Dict[str, List[str]]:
    def _strip(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            return v[1:-1]
        return v

    return {k: [_strip(x) for x in vs] for k, vs in values.items()}


def value_frequencies(d: Dict[str, List[str]]) -> Dict[str, Counter]:
    return {k: Counter(v) for k, v in d.items()}


# ----------------------------------------------------------------------
# Jensen-Shannon distance helpers
# ----------------------------------------------------------------------

def counter_to_probability_vector(
    counter: Counter,
    vocabulary: Sequence[str],
) -> List[float]:
    """
    Convert a Counter into a probability vector over a fixed shared vocabulary.
    """
    total = sum(counter.values())
    if total == 0:
        return [0.0] * len(vocabulary)

    return [counter.get(v, 0) / total for v in vocabulary]


def kl_divergence(
    p: Sequence[float],
    q: Sequence[float],
) -> float:
    """
    KL divergence KL(p || q), assuming q_i > 0 whenever p_i > 0.
    """
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            s += pi * math.log(pi / qi, 2)
    return s


def jensen_shannon_distance_from_counters(
    c1: Counter,
    c2: Counter,
) -> float:
    """
    Compute Jensen-Shannon distance between two categorical count distributions.

    Returns:
        0.0 for identical distributions
        larger values for more different distributions
    """
    vocabulary = sorted(set(c1) | set(c2))

    if not vocabulary:
        return 0.0

    p = counter_to_probability_vector(c1, vocabulary)
    q = counter_to_probability_vector(c2, vocabulary)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    js_dist = math.sqrt(js_div)

    return js_dist


def compute_js_distances_per_key(
    comparison: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute Jensen-Shannon distance for every analyzed key/field.
    """
    distances: Dict[str, float] = {}

    for key, result in comparison.items():
        c1 = result["counter_1"]
        c2 = result["counter_2"]
        distances[key] = jensen_shannon_distance_from_counters(c1, c2)

    return distances


def compute_average_js_distance(
    comparison: Dict[str, Dict[str, Any]],
) -> float:
    """
    Compute the final file-to-file distance as the mean JS distance across keys.
    """
    per_key = compute_js_distances_per_key(comparison)

    if not per_key:
        return 0.0

    return sum(per_key.values()) / len(per_key)


# ----------------------------------------------------------------------
# Comparison logic
# ----------------------------------------------------------------------

def compare_field_values(
    values_1: List[str],
    values_2: List[str],
    *,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Compare one extracted field across two files.
    """
    c1 = Counter(values_1)
    c2 = Counter(values_2)

    n1 = sum(c1.values())
    n2 = sum(c2.values())

    only_1 = set(c1) - set(c2)
    only_2 = set(c2) - set(c1)

    top_unique_1 = sorted(only_1, key=lambda v: c1[v], reverse=True)[:top_k]
    top_unique_2 = sorted(only_2, key=lambda v: c2[v], reverse=True)[:top_k]

    rate_diff = {}
    if n1 > 0 and n2 > 0:
        rate_diff = {
            v: (c1[v] / n1) - (c2[v] / n2)
            for v in (set(c1) | set(c2))
        }

    top_rate_diff = sorted(rate_diff.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

    return {
        "n1": n1,
        "n2": n2,
        "num_unique_1": len(c1),
        "num_unique_2": len(c2),
        "num_only_1": len(only_1),
        "num_only_2": len(only_2),
        "counter_1": c1,
        "counter_2": c2,
        "top_unique_1": [(v, c1[v]) for v in top_unique_1],
        "top_unique_2": [(v, c2[v]) for v in top_unique_2],
        "top_rate_diff": top_rate_diff,
    }


def compare_extracted_dicts(
    d1: Dict[str, List[str]],
    d2: Dict[str, List[str]],
    *,
    keys: Sequence[str],
    top_k: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple fields across two extracted dictionaries.
    """
    return {
        key: compare_field_values(d1.get(key, []), d2.get(key, []), top_k=top_k)
        for key in keys
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def print_field_report(
    field_name: str,
    result: Dict[str, Any],
    *,
    file1_label: str = "file 1",
    file2_label: str = "file 2",
    example_k: int = 10,
) -> None:
    print(f"\n=== FIELD: {field_name} ===")
    print(
        f"{file1_label}: total={result['n1']}, unique={result['num_unique_1']}, "
        f"unique-only={result['num_only_1']}"
    )
    print(
        f"{file2_label}: total={result['n2']}, unique={result['num_unique_2']}, "
        f"unique-only={result['num_only_2']}"
    )

    print(f"\nTop {example_k} unique values only in {file1_label}:")
    for v, c in result["top_unique_1"][:example_k]:
        print(f"  {v}, {c}")

    print(f"\nTop {example_k} unique values only in {file2_label}:")
    for v, c in result["top_unique_2"][:example_k]:
        print(f"  {v}, {c}")

    print(f"\nTop {example_k} rate differences ({file1_label} - {file2_label}):")
    for v, d in result["top_rate_diff"][:example_k]:
        print(f"  {v}: {d:+.4f}")


def print_comparison_report(
    comparison: Dict[str, Dict[str, Any]],
    *,
    file1_label: str = "file 1",
    file2_label: str = "file 2",
    example_k: int = 10,
) -> None:
    for field_name, result in comparison.items():
        print_field_report(
            field_name,
            result,
            file1_label=file1_label,
            file2_label=file2_label,
            example_k=example_k,
        )


def print_js_distance_report(
    comparison: Dict[str, Dict[str, Any]],
    *,
    file1_label: str = "file 1",
    file2_label: str = "file 2",
) -> None:
    """
    Print the per-key JS distances and the final averaged distance.
    """
    per_key = compute_js_distances_per_key(comparison)
    final_distance = compute_average_js_distance(comparison)

    print("\n=== JENSEN-SHANNON DISTANCES ===")
    print(f"Files: {file1_label} vs {file2_label}")

    for key, dist in per_key.items():
        print(f"  {key}: {dist:.6f}")

    print(f"\nFinal averaged distance: {final_distance:.6f}")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_rate_differences(
    field_name: str,
    result: Dict[str, Any],
    *,
    file1_label: str = "file 1",
    file2_label: str = "file 2",
    top_k: int = 20,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    rows = result["top_rate_diff"][:top_k]
    if not rows:
        print(f"No rate-difference data to plot for field: {field_name}")
        return

    labels = [val for val, _ in rows][::-1]
    values = [d for _, d in rows][::-1]

    plt.figure(figsize=figsize)
    bars = plt.barh(labels, values)
    plt.axvline(0)

    plt.xlabel(f"Rate difference ({file1_label} - {file2_label})")
    plt.title(f"Top {top_k} {field_name} frequency differences (normalized)")

    max_abs = max(abs(v) for v in values) if values else 0.001
    offset = max(0.0005, 0.02 * max_abs)

    for bar, v in zip(bars, values):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if x >= 0:
            text_x = x + offset
            ha = "left"
        else:
            text_x = x - offset
            ha = "right"

        plt.text(
            text_x,
            y,
            f"{v:+.3f}",
            va="center",
            ha=ha,
            fontsize=9,
        )

    plt.tight_layout()
    plt.show()


def plot_comparison(
    comparison: Dict[str, Dict[str, Any]],
    *,
    file1_label: str = "file 1",
    file2_label: str = "file 2",
    top_k: int = 20,
) -> None:
    for field_name, result in comparison.items():
        plot_rate_differences(
            field_name,
            result,
            file1_label=file1_label,
            file2_label=file2_label,
            top_k=top_k,
        )


# ----------------------------------------------------------------------
# High-level convenience
# ----------------------------------------------------------------------

def analyze_two_files(
    file_1: str | Path,
    file_2: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = r"\S+",
    require_all_keys: bool = False,
    ignore_case: bool = False,
    strip_quoted_values: bool = True,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Full analysis pipeline for two files.
    """
    d1 = extract_kv_from_file(
        file_1,
        prefix_regex=prefix_regex,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
    )

    d2 = extract_kv_from_file(
        file_2,
        prefix_regex=prefix_regex,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
    )

    if strip_quoted_values:
        d1 = strip_quotes(d1)
        d2 = strip_quotes(d2)

    comparison = compare_extracted_dicts(d1, d2, keys=keys, top_k=top_k)
    js_distances_per_key = compute_js_distances_per_key(comparison)
    average_js_distance = compute_average_js_distance(comparison)

    return {
        "file_1": str(file_1),
        "file_2": str(file_2),
        "keys": list(keys),
        "extracted_1": d1,
        "extracted_2": d2,
        "comparison": comparison,
        "js_distances_per_key": js_distances_per_key,
        "average_js_distance": average_js_distance,
    }


def analyze_print_and_plot(
    file_1: str | Path,
    file_2: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = r"\S+",
    require_all_keys: bool = False,
    ignore_case: bool = False,
    strip_quoted_values: bool = True,
    top_k: int = 20,
    print_report: bool = True,
    plot: bool = True,
    print_js_distance: bool = True,
    file1_label: str = "file 1",
    file2_label: str = "file 2",
) -> Dict[str, Any]:
    """
    Convenience wrapper: analyze, print report, optionally plot, and print JS distances.
    """
    result = analyze_two_files(
        file_1,
        file_2,
        prefix_regex=prefix_regex,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
        strip_quoted_values=strip_quoted_values,
        top_k=top_k,
    )

    comparison = result["comparison"]

    if print_report:
        print_comparison_report(
            comparison,
            file1_label=file1_label,
            file2_label=file2_label,
            example_k=min(10, top_k),
        )

    if print_js_distance:
        print_js_distance_report(
            comparison,
            file1_label=file1_label,
            file2_label=file2_label,
        )

    if plot:
        plot_comparison(
            comparison,
            file1_label=file1_label,
            file2_label=file2_label,
            top_k=top_k,
        )

    return result


def analyze_file_pairs(
    file_pairs: Sequence[tuple[str, str | Path]],
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = r"\S+",
    require_all_keys: bool = False,
    ignore_case: bool = False,
    strip_quoted_values: bool = True,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Analyze all file combinations and compute every JS-based distance exposed here.
    """
    extracted: Dict[str, Dict[str, List[str]]] = {}

    for label, file_path in file_pairs:
        values = extract_kv_from_file(
            file_path,
            prefix_regex=prefix_regex,
            keys=keys,
            value_pattern=value_pattern,
            require_all_keys=require_all_keys,
            ignore_case=ignore_case,
        )
        if strip_quoted_values:
            values = strip_quotes(values)
        extracted[label] = values

    results: List[Dict[str, Any]] = []

    for (label_1, file_1), (label_2, file_2) in combinations(file_pairs, 2):
        comparison = compare_extracted_dicts(
            extracted[label_1],
            extracted[label_2],
            keys=keys,
            top_k=top_k,
        )
        results.append({
            "file_1": str(file_1),
            "file_2": str(file_2),
            "label_1": label_1,
            "label_2": label_2,
            "comparison": comparison,
            "js_distances_per_key": compute_js_distances_per_key(comparison),
            "average_js_distance": compute_average_js_distance(comparison),
        })

    return results


def print_pairwise_js_report(
    pairwise_results: Sequence[Dict[str, Any]],
    *,
    sort_by: str = "average_js_distance",
) -> None:
    def extract_distance(result: Dict[str, Any]) -> float:
        if sort_by == "average_js_distance":
            return result["average_js_distance"]
        if sort_by in result["js_distances_per_key"]:
            return result["js_distances_per_key"][sort_by]
        raise ValueError(f"Unknown sort_by: {sort_by}")

    print(f"\nPairwise Jensen-Shannon distances sorted by {sort_by}:")
    for result in sorted(pairwise_results, key=extract_distance, reverse=True):
        per_key = ", ".join(
            f"{key}={distance:.6f}"
            for key, distance in sorted(result["js_distances_per_key"].items())
        )
        print(
            f"{result['label_1']} vs {result['label_2']} | "
            f"selected_distance={extract_distance(result):.6f} | "
            f"average_js_distance={result['average_js_distance']:.6f} | "
            f"per_key: {per_key}"
        )


def get_mode_config(mode: str) -> tuple[Pattern[str], List[str]]:
    if mode == "execve":
        return re.compile(r"^type=EXECVE\s+msg=audit\("), ["a0", "a1", "a2"]
    if mode == "path":
        return re.compile(r"^type=PATH\s+msg=audit\("), ["name", "nametype"]
    if mode == "syscall":
        return (
            re.compile(r"^type=SYSCALL\s+msg=audit\("),
            ["syscall", "success", "exit", "comm", "exe", "auid", "uid", "tty", "key"],
        )
    if mode == "sockaddr":
        return re.compile(r"^type=SOCKADDR\s+msg=audit\("), ["saddr"]
    raise ValueError(f"Unknown mode: {mode}")


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    use_data_wp = False
    dataset = "Data_WP" if use_data_wp else "Data"

    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    # 3. score=5.600000 | distance=a2 | hyperparameters: mode=execve

    ### Hyperparameter
    mode = "execve"  # worth changing: "execve", "path", "syscall", "sockaddr" depending on which audit fields you want to compare
    sort_by = "a2"  # worth changing: "average_js_distance" for an overall summary, or a specific key like "a0", "a1", "a2"
    # don't change this
    value_pattern = QUOTED_OR_BARE_VALUE_PATTERN
    require_all_keys = False
    ignore_case = False
    strip_quoted_values = True
    top_k = 30

    prefix, keys = get_mode_config(mode)

    pairwise_results = analyze_file_pairs(
        file_pairs,
        prefix_regex=prefix,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
        strip_quoted_values=strip_quoted_values,
        top_k=top_k,
    )
    print_pairwise_js_report(pairwise_results, sort_by=sort_by)

    labels = [label for label, _ in file_pairs]
    ordered_labels = group_labels_humans_first(labels)
    def extract_selected_distance(item: Dict[str, Any]) -> float:
        if sort_by == "average_js_distance":
            return item["average_js_distance"]
        if sort_by in item["js_distances_per_key"]:
            return item["js_distances_per_key"][sort_by]
        raise ValueError(f"Unknown sort_by: {sort_by}")

    matrix = build_symmetric_distance_matrix(
        ordered_labels,
        pairwise_results,
        extract_pair=lambda item: (
            item["label_1"],
            item["label_2"],
            extract_selected_distance(item),
        ),
    )
    group_stats = analyze_binary_group_structure(matrix, ordered_labels)
    print_binary_group_structure_report(group_stats)

    plot_distance_heatmap(matrix, ordered_labels, title=f"{mode} {sort_by} heatmap")
    plot_mds_embedding(matrix, ordered_labels, title=f"{mode} {sort_by} MDS")

'''
if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"
    output_path = f"results/statistic_audit_field_analyzer{'_wp' if use_data_wp else ''}.csv"
    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    # fixed settings
    value_pattern = QUOTED_OR_BARE_VALUE_PATTERN
    require_all_keys = False
    ignore_case = False
    strip_quoted_values = True
    top_k = 30

    labels = [label for label, _ in file_pairs]
    ordered_labels = group_labels_humans_first(labels)

    modes = ["execve", "path", "syscall", "sockaddr"]

    for mode_idx, mode in enumerate(modes, 1):
        prefix, keys = get_mode_config(mode)

        print(
            f"\n[MODE {mode_idx}/{len(modes)}] "
            f"mode={mode} "
            f"keys={keys}"
        )

        pairwise_results = analyze_file_pairs(
            file_pairs,
            prefix_regex=prefix,
            keys=keys,
            value_pattern=value_pattern,
            require_all_keys=require_all_keys,
            ignore_case=ignore_case,
            strip_quoted_values=strip_quoted_values,
            top_k=top_k,
        )

        # valid distance choices for this mode:
        # average_js_distance + every extracted key
        sort_by_options = ["average_js_distance", *keys]

        for dist_idx, sort_by in enumerate(sort_by_options, 1):
            print(
                f"\n  [DIST {dist_idx}/{len(sort_by_options)}] "
                f"mode={mode} "
                f"sort_by={sort_by}"
            )

            print_pairwise_js_report(pairwise_results, sort_by=sort_by)

            def extract_selected_distance(item: Dict[str, Any], sort_by: str = sort_by) -> float:
                if sort_by == "average_js_distance":
                    return item["average_js_distance"]
                if sort_by in item["js_distances_per_key"]:
                    return item["js_distances_per_key"][sort_by]
                raise ValueError(f"Unknown sort_by: {sort_by}")

            matrix = build_symmetric_distance_matrix(
                ordered_labels,
                pairwise_results,
                extract_pair=lambda item, sort_by=sort_by: (
                    item["label_1"],
                    item["label_2"],
                    extract_selected_distance(item, sort_by),
                ),
            )

            group_stats = analyze_binary_group_structure(matrix, ordered_labels)
            print_binary_group_structure_report(group_stats)

            append_statistic_evaluation_row(
                approach="audit_field_analyzer",
                distance_name=sort_by,
                ordered_labels=ordered_labels,
                hyperparameters={
                    "mode": mode,
                },
                group_stats=group_stats,
                output_path=output_path,
            )
'''
