import math
import re
from pathlib import Path
from typing import Pattern, Sequence, List, Dict, Any, Optional, Tuple
from collections import Counter
from itertools import combinations, product
import matplotlib.pyplot as plt

from scipy.spatial.distance import jensenshannon

from stats_tools.data_catalog import get_log_path, analysis_actors
from stats_tools.pairwise_group_stats import analyze_binary_group_structure, print_binary_group_structure_report
from stats_tools.statistic_evaluation_csv import append_statistic_evaluation_row
from stats_tools.pairwise_viz import (
    build_symmetric_distance_matrix,
    group_labels_humans_first,
    plot_distance_heatmap,
    plot_mds_embedding,
)


QUOTED_OR_BARE_VALUE_PATTERN = r'(?:\"[^"]*\"|\S+)'


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def extract_rows_from_file(
    file_path: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = QUOTED_OR_BARE_VALUE_PATTERN,
    require_all_keys: bool = False,
    ignore_case: bool = False,
    keep_line: bool = True,
    strip_quotes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return row-aligned extracted data: one dict per matched log line.

    Each row contains:
      - "_lineno": 1-based line number in the file
      - "_line": original line (if keep_line=True)
      - extracted keys found in that line: {key: value, ...}

    If require_all_keys=True, only rows containing ALL keys are kept.
    """
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    flags = re.IGNORECASE if ignore_case else 0

    key_regexes = {
        k: re.compile(rf"\b{re.escape(k)}=(?P<val>{value_pattern})(?=\s|$)", flags=flags)
        for k in keys
    }

    rows: List[Dict[str, Any]] = []

    for lineno, line in enumerate(lines, start=1):
        if not prefix_regex.search(line):
            continue

        found: Dict[str, str] = {}
        for k, rx in key_regexes.items():
            m = rx.search(line)
            if m:
                val = m.group("val")
                if strip_quotes:
                    val = strip_outer_quotes(val)
                found[k] = val

        if require_all_keys and any(k not in found for k in keys):
            continue

        row: Dict[str, Any] = {"_lineno": lineno}
        if keep_line:
            row["_line"] = line
        row.update(found)

        rows.append(row)

    return rows


# ----------------------------------------------------------------------
# Co-occurrence helpers
# ----------------------------------------------------------------------

def conditional_value_counts(
    rows: List[Dict[str, Any]],
    *,
    given_key: str,
    given_value: str,
    target_key: str,
) -> Counter:
    """
    Count target_key values among rows where given_key == given_value.

    Example:
        given_key="a0", given_value="curl", target_key="a1"
    """
    return Counter(
        row[target_key]
        for row in rows
        if row.get(given_key) == given_value and target_key in row
    )


def pair_counts(
    rows: List[Dict[str, Any]],
    *,
    key1: str,
    key2: str,
) -> Counter:
    """
    Count co-occurring pairs (row[key1], row[key2]).
    """
    return Counter(
        (row[key1], row[key2])
        for row in rows
        if key1 in row and key2 in row
    )


def top_conditional_values(
    rows: List[Dict[str, Any]],
    *,
    given_key: str,
    given_value: str,
    target_key: str,
    top_k: int = 10,
) -> List[Tuple[str, int]]:
    c = conditional_value_counts(
        rows,
        given_key=given_key,
        given_value=given_value,
        target_key=target_key,
    )
    return c.most_common(top_k)


# ----------------------------------------------------------------------
# Presence / missing-key-pattern helpers
# ----------------------------------------------------------------------

def row_presence_pattern(
    row: Dict[str, Any],
    keys: Sequence[str],
) -> Tuple[str, ...]:
    """
    Return tuple of keys present in this row, preserving key order.

    Example:
        keys=["a0","a1","a2"], row has only a0 and a2
        -> ("a0", "a2")
    """
    return tuple(k for k in keys if k in row)


def presence_pattern_counts(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
) -> Counter:
    """
    Count how often each key-presence pattern appears.

    Example output keys:
        ("a0",)
        ("a0","a1")
        ("a0","a1","a2")
    """
    return Counter(row_presence_pattern(row, keys) for row in rows)


def command_presence_patterns(
    rows: List[Dict[str, Any]],
    *,
    command_key: str,
    command_value: Optional[str],
    keys: Sequence[str],
) -> Counter:
    """
    Count presence patterns for rows of one specific command.

    Example:
        command_key="a0", command_value="curl", keys=["a0","a1","a2"]

    If command_value is None, uses all rows that contain command_key.
    """
    selected = []
    for row in rows:
        if command_key not in row:
            continue
        if command_value is not None and row[command_key] != command_value:
            continue
        selected.append(row)

    return Counter(row_presence_pattern(row, keys) for row in selected)


def rows_with_all_keys(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    return [row for row in rows if all(k in row for k in keys)]


def rows_with_missing_keys(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
) -> List[Dict[str, Any]]:
    return [row for row in rows if any(k not in row for k in keys)]


def command_key_completeness_summary(
    rows: List[Dict[str, Any]],
    *,
    command_key: str,
    command_value: Optional[str],
    keys: Sequence[str],
) -> Dict[str, Any]:
    """
    Summarize whether a command tends to appear with all keys or only subsets.
    """
    selected = []
    for row in rows:
        if command_key not in row:
            continue
        if command_value is not None and row[command_key] != command_value:
            continue
        selected.append(row)

    total = len(selected)
    full = sum(1 for row in selected if all(k in row for k in keys))
    partial = total - full

    patterns = Counter(row_presence_pattern(row, keys) for row in selected)

    return {
        "command_key": command_key,
        "command_value": command_value,
        "total_rows": total,
        "rows_with_all_keys": full,
        "rows_with_missing_keys": partial,
        "fraction_all_keys": (full / total) if total > 0 else 0.0,
        "pattern_counts": patterns,
    }


# ----------------------------------------------------------------------
# Distance helpers
# ----------------------------------------------------------------------

def counter_to_prob_vector(counter: Counter, vocabulary: Sequence[Any]) -> List[float]:
    total = sum(counter.values())
    if total == 0:
        return [0.0] * len(vocabulary)
    return [counter.get(v, 0) / total for v in vocabulary]


def js_distance_from_counters(c1: Counter, c2: Counter) -> float:
    """
    Jensen-Shannon distance in [0, 1] using scipy.

    - 0.0 -> identical distributions
    - larger -> more different
    """
    vocabulary = sorted(set(c1) | set(c2), key=str)

    if not vocabulary:
        return 0.0

    total1 = sum(c1.values())
    total2 = sum(c2.values())

    if total1 == 0 and total2 == 0:
        return 0.0
    if total1 == 0 or total2 == 0:
        return 1.0

    p = counter_to_prob_vector(c1, vocabulary)
    q = counter_to_prob_vector(c2, vocabulary)

    return float(jensenshannon(p, q, base=2.0))


def non_full_fraction(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
) -> float:
    """
    Fraction of rows that do NOT contain all keys.
    """
    total = len(rows)
    if total == 0:
        return 0.0

    non_full = sum(1 for row in rows if any(k not in row for k in keys))
    return non_full / total


def presence_pattern_distance(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
) -> Dict[str, Any]:
    """
    JS distance between presence-pattern distributions.

    Each row contributes a pattern describing which keys are present,
    e.g. ("a0","a1","a2") or ("a0","a1").
    """

    c1 = presence_pattern_counts(rows1, keys=keys)
    c2 = presence_pattern_counts(rows2, keys=keys)

    distance = js_distance_from_counters(c1, c2)

    return {
        "num_patterns_1": len(c1),
        "num_patterns_2": len(c2),
        "distance": distance,
        "counter_1": c1,
        "counter_2": c2,
    }


def pair_distribution_distance(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    *,
    key1: str,
    key2: str,
) -> Dict[str, Any]:
    """
    JS distance between pair distributions such as (a0, a1).
    """
    c1 = pair_counts(rows1, key1=key1, key2=key2)
    c2 = pair_counts(rows2, key1=key1, key2=key2)

    return {
        "key1": key1,
        "key2": key2,
        "num_unique_pairs_1": len(c1),
        "num_unique_pairs_2": len(c2),
        "distance": js_distance_from_counters(c1, c2),
        "counter_1": c1,
        "counter_2": c2,
    }


def conditional_distribution_distance(
    rows1: List[Dict[str, Any]],
    rows2: List[Dict[str, Any]],
    *,
    given_key: str,
    given_value: str,
    target_key: str,
) -> Dict[str, Any]:
    """
    JS distance between conditional target distributions:
        P(target_key | given_key = given_value)
    """
    c1 = conditional_value_counts(
        rows1,
        given_key=given_key,
        given_value=given_value,
        target_key=target_key,
    )
    c2 = conditional_value_counts(
        rows2,
        given_key=given_key,
        given_value=given_value,
        target_key=target_key,
    )

    return {
        "given_key": given_key,
        "given_value": given_value,
        "target_key": target_key,
        "num_unique_targets_1": len(c1),
        "num_unique_targets_2": len(c2),
        "distance": js_distance_from_counters(c1, c2),
        "counter_1": c1,
        "counter_2": c2,
    }


def compute_row_distances(
    file_1: str | Path,
    file_2: str | Path,
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = QUOTED_OR_BARE_VALUE_PATTERN,
    require_all_keys: bool = False,
    ignore_case: bool = False,
    keep_line: bool = False,
    strip_quotes: bool = True,
    pair_key1: str,
    pair_key2: str,
    conditional_given_key: str,
    conditional_given_value: str,
    conditional_target_key: str,
) -> Dict[str, Any]:
    """
    Extract rows from two files and compute the three distances:
      1) presence-pattern distance
      2) pair-distribution distance
      3) conditional-distribution distance
    """
    rows1 = extract_rows_from_file(
        file_1,
        prefix_regex=prefix_regex,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
        keep_line=keep_line,
        strip_quotes=strip_quotes,
    )

    rows2 = extract_rows_from_file(
        file_2,
        prefix_regex=prefix_regex,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
        keep_line=keep_line,
        strip_quotes=strip_quotes,
    )

    presence = presence_pattern_distance(rows1, rows2, keys=keys)
    pair = pair_distribution_distance(rows1, rows2, key1=pair_key1, key2=pair_key2)
    conditional = conditional_distribution_distance(
        rows1,
        rows2,
        given_key=conditional_given_key,
        given_value=conditional_given_value,
        target_key=conditional_target_key,
    )

    return {
        "file_1": str(file_1),
        "file_2": str(file_2),
        "rows_1": rows1,
        "rows_2": rows2,
        "presence_pattern_distance": presence,
        "pair_distribution_distance": pair,
        "conditional_distribution_distance": conditional,
    }

def _format_distribution_label(x: Any) -> str:
    if isinstance(x, tuple):
        return " | ".join(str(v) for v in x)
    return str(x)


def _counter_to_prob_dict(counter: Counter) -> Dict[Any, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def plot_distribution_differences(
    result: Dict[str, Any],
    *,
    distribution_name: str,
    top_k: int = 15,
) -> None:
    """
    Plot the biggest absolute probability differences between the two distributions
    stored inside one pairwise result.

    distribution_name:
        - "presence_pattern_distance"
        - "pair_distribution_distance"
        - "conditional_distribution_distance"
    """
    if distribution_name not in result:
        raise ValueError(f"Unknown distribution_name: {distribution_name}")

    dist_info = result[distribution_name]
    counter_1 = dist_info["counter_1"]
    counter_2 = dist_info["counter_2"]

    vocab = sorted(set(counter_1) | set(counter_2), key=str)
    if not vocab:
        print(f"No values to plot for {distribution_name}.")
        return

    prob_1 = _counter_to_prob_dict(counter_1)
    prob_2 = _counter_to_prob_dict(counter_2)

    rows = []
    for item in vocab:
        p1 = prob_1.get(item, 0.0)
        p2 = prob_2.get(item, 0.0)
        diff = abs(p1 - p2)
        rows.append((item, p1, p2, diff))

    rows.sort(key=lambda x: x[3], reverse=True)
    rows = rows[:top_k]

    labels = [_format_distribution_label(item) for item, _, _, _ in rows]
    vals_1 = [p1 for _, p1, _, _ in rows]
    vals_2 = [p2 for _, _, p2, _ in rows]

    x = range(len(rows))
    width = 0.4

    label_1 = "AI" if "gpt" in result["label_1"].lower() else "Human"
    label_2 = "AI" if "gpt" in result["label_2"].lower() else "Human"
    plt.figure(figsize=(max(10, len(rows) * 0.8), 6))
    plt.bar([i - width / 2 for i in x], vals_1, width=width, label=label_1)
    plt.bar([i + width / 2 for i in x], vals_2, width=width, label=label_2)

    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/audit_row_extract_armin_gpt4.1.pdf")
    plt.show()

# ----------------------------------------------------------------------
# Small print helpers
# ----------------------------------------------------------------------

def print_presence_pattern_report(
    rows: List[Dict[str, Any]],
    *,
    keys: Sequence[str],
    top_k: int = 10,
    title: str = "Presence patterns",
) -> None:
    counts = presence_pattern_counts(rows, keys=keys)

    print(f"\n{title}")
    print("-" * len(title))

    for pattern, n in counts.most_common(top_k):
        label = ", ".join(pattern) if pattern else "<none>"
        print(f"  {label}: {n}")


def print_command_completeness_report(
    rows: List[Dict[str, Any]],
    *,
    command_key: str,
    command_value: Optional[str],
    keys: Sequence[str],
    top_k_patterns: int = 10,
) -> None:
    rep = command_key_completeness_summary(
        rows,
        command_key=command_key,
        command_value=command_value,
        keys=keys,
    )

    cmd_label = (
        f"{command_key}={command_value}"
        if command_value is not None
        else f"all rows with {command_key}"
    )

    print(f"\nCompleteness report for {cmd_label}")
    print("-" * (24 + len(cmd_label)))
    print(f"  total rows:            {rep['total_rows']}")
    print(f"  rows with all keys:    {rep['rows_with_all_keys']}")
    print(f"  rows with missing key: {rep['rows_with_missing_keys']}")
    print(f"  fraction all keys:     {rep['fraction_all_keys']:.3f}")

    print("\n  top presence patterns:")
    for pattern, n in rep["pattern_counts"].most_common(top_k_patterns):
        label = ", ".join(pattern) if pattern else "<none>"
        print(f"    {label}: {n}")


def print_conditional_values_report(
    rows: List[Dict[str, Any]],
    *,
    given_key: str,
    given_value: str,
    target_key: str,
    top_k: int = 10,
) -> None:
    top = top_conditional_values(
        rows,
        given_key=given_key,
        given_value=given_value,
        target_key=target_key,
        top_k=top_k,
    )

    print(f"\nTop {top_k} values of {target_key} when {given_key}={given_value}")
    print("-" * (20 + len(target_key) + len(given_key) + len(given_value)))

    if not top:
        print("  <no matches>")
        return

    for value, n in top:
        print(f"  {value}: {n}")


def print_distance_report(result: Dict[str, Any]) -> None:
    p = result["presence_pattern_distance"]
    pair = result["pair_distribution_distance"]
    cond = result["conditional_distribution_distance"]

    print("\n=== DISTANCES ===")

    print("\n1) Presence-pattern distance")
    print(f"  non_full_fraction file 1: {p['non_full_fraction_1']:.6f}")
    print(f"  non_full_fraction file 2: {p['non_full_fraction_2']:.6f}")
    print(f"  distance:                 {p['distance']:.6f}")

    print("\n2) Pair-distribution distance")
    print(f"  pair: ({pair['key1']}, {pair['key2']})")
    print(f"  unique pairs file 1: {pair['num_unique_pairs_1']}")
    print(f"  unique pairs file 2: {pair['num_unique_pairs_2']}")
    print(f"  JS distance:         {pair['distance']:.6f}")

    print("\n3) Conditional-distribution distance")
    print(
        f"  condition: {cond['given_key']}={cond['given_value']} "
        f"-> target {cond['target_key']}"
    )
    print(f"  unique target values file 1: {cond['num_unique_targets_1']}")
    print(f"  unique target values file 2: {cond['num_unique_targets_2']}")
    print(f"  JS distance:                 {cond['distance']:.6f}")


def score_file_pairs(
    file_pairs: Sequence[tuple[str, str | Path]],
    *,
    prefix_regex: Pattern[str],
    keys: Sequence[str],
    value_pattern: str = QUOTED_OR_BARE_VALUE_PATTERN,
    require_all_keys: bool = False,
    ignore_case: bool = False,
    keep_line: bool = False,
    strip_quotes: bool = True,
    pair_key1: str,
    pair_key2: str,
    conditional_given_key: str,
    conditional_given_value: str,
    conditional_target_key: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for (label_1, file_1), (label_2, file_2) in combinations(file_pairs, 2):
        result = compute_row_distances(
            file_1,
            file_2,
            prefix_regex=prefix_regex,
            keys=keys,
            value_pattern=value_pattern,
            require_all_keys=require_all_keys,
            ignore_case=ignore_case,
            keep_line=keep_line,
            strip_quotes=strip_quotes,
            pair_key1=pair_key1,
            pair_key2=pair_key2,
            conditional_given_key=conditional_given_key,
            conditional_given_value=conditional_given_value,
            conditional_target_key=conditional_target_key,
        )
        result["label_1"] = label_1
        result["label_2"] = label_2
        results.append(result)

    return results


def print_pairwise_distance_report(
    pairwise_results: Sequence[Dict[str, Any]],
    *,
    sort_by: str = "pair_distribution_distance",
) -> None:
    valid_sort_keys = {
        "presence_pattern_distance": lambda item: item["presence_pattern_distance"]["distance"],
        "pair_distribution_distance": lambda item: item["pair_distribution_distance"]["distance"],
        "conditional_distribution_distance": lambda item: item["conditional_distribution_distance"]["distance"],
    }
    if sort_by not in valid_sort_keys:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    print(f"\nPairwise row distances sorted by {sort_by}:")
    for result in sorted(pairwise_results, key=valid_sort_keys[sort_by], reverse=True):
        presence_distance = result["presence_pattern_distance"]["distance"]
        pair_distance = result["pair_distribution_distance"]["distance"]
        conditional_distance = result["conditional_distribution_distance"]["distance"]
        print(
            f"{result['label_1']} vs {result['label_2']} | "
            f"presence={presence_distance:.6f} | "
            f"pair_js={pair_distance:.6f} | "
            f"conditional_js={conditional_distance:.6f}"
        )


def get_pair_result(
    pairwise_results: Sequence[Dict[str, Any]],
    *,
    label_1: str,
    label_2: str,
) -> Dict[str, Any]:
    wanted = {label_1, label_2}
    for item in pairwise_results:
        if {item["label_1"], item["label_2"]} == wanted:
            return item
    raise ValueError(f"Pair not found: {label_1} vs {label_2}")

def get_most_different_pair(
    pairwise_results: Sequence[Dict[str, Any]],
    *,
    sort_by: str,
) -> Dict[str, Any]:
    if not pairwise_results:
        raise ValueError("pairwise_results is empty")
    return max(pairwise_results, key=lambda item: item[sort_by]["distance"])

def get_mode_config(mode: str) -> tuple[Pattern[str], List[str]]:
    if mode == "execve":
        return re.compile(r"^type=EXECVE\s+msg=audit\("), ["a0", "a1", "a2", "a3", "a4", "a5", "argc"]
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

    # 1. score=1.000000 | distance=conditional_distribution_distance | hyperparameters: conditional_given_key=comm, conditional_given_value=tail, conditional_target_key=tty, mode=syscall, pair_key1=comm, pair_key2=tty

    # 2. score=2.000000 | distance=conditional_distribution_distance | hyperparameters: conditional_given_key=comm, conditional_given_value=tail, conditional_target_key=tty, mode=syscall, pair_key1=comm, pair_key2=exe

    # 4. score=4.000000 | distance=conditional_distribution_distance | hyperparameters: conditional_given_key=a0, conditional_given_value=grep, conditional_target_key=a1, mode=execve, pair_key1=a0, pair_key2=a1

    # 7. score=9.000000 | distance=conditional_distribution_distance | hyperparameters: conditional_given_key=a0, conditional_given_value=tail, conditional_target_key=a1, mode=execve, pair_key1=a0, pair_key2=a1

    ### Hyperparameter
    mode = "execve"  # worth changing: "execve", "path", "syscall", "sockaddr" depending on which audit record family you want
    pair_key1 = "a0"  # worth changing: e.g. "a0" vs "a1", or "name" vs "nametype", depending on mode
    pair_key2 = "a1"  # worth changing together with pair_key1 to define which co-occurrence structure is compared
    conditional_given_key = "a0"  # worth changing: the key you condition on, e.g. "a0" or "comm"
    conditional_given_value = "grep"  # worth changing: try commands like "curl", "ls", "cat", "tail"
    conditional_target_key = "a1"  # worth changing: the argument/field whose conditional distribution you want to compare
    sort_by = "conditional_distribution_distance"  # worth changing: "presence_pattern_distance", "pair_distribution_distance", or "conditional_distribution_distance"
    
    # optional difference plot
    plot_distribution_diff = True
    plot_top_k = 15
    # choose actors manually; set to None to use the most different pair automatically
    selected_human_actor = "GPT4.1"
    selected_ai_actor = "Torina"

    # don't change this
    require_all_keys = False
    ignore_case = False
    keep_line = False
    strip_quotes = True 
    value_pattern = QUOTED_OR_BARE_VALUE_PATTERN


    prefix, keys = get_mode_config(mode)

    pairwise_results = score_file_pairs(
        file_pairs,
        prefix_regex=prefix,
        keys=keys,
        value_pattern=value_pattern,
        require_all_keys=require_all_keys,
        ignore_case=ignore_case,
        keep_line=keep_line,
        strip_quotes=strip_quotes,
        pair_key1=pair_key1,
        pair_key2=pair_key2,
        conditional_given_key=conditional_given_key,
        conditional_given_value=conditional_given_value,
        conditional_target_key=conditional_target_key,
    )
    print_pairwise_distance_report(pairwise_results, sort_by=sort_by)

    ordered_labels = group_labels_humans_first([label for label, _ in file_pairs])
    matrix = build_symmetric_distance_matrix(
        ordered_labels,
        pairwise_results,
        extract_pair=lambda item: (
            item["label_1"],
            item["label_2"],
            item[sort_by]["distance"],
        ),
    )
    group_stats = analyze_binary_group_structure(matrix, ordered_labels)
    print_binary_group_structure_report(group_stats)

    plot_distance_heatmap(matrix, ordered_labels, title=f"{mode} {sort_by} heatmap", anonymize_humans=False)
    plot_mds_embedding(matrix, ordered_labels, title=f"{mode} {sort_by} MDS")

    if plot_distribution_diff:
        if selected_human_actor is not None and selected_ai_actor is not None:
            selected_result = get_pair_result(
                pairwise_results,
                label_1=selected_human_actor,
                label_2=selected_ai_actor,
            )
        else:
            selected_result = get_most_different_pair(pairwise_results, sort_by=sort_by)

        plot_distribution_differences(
            selected_result,
            distribution_name=sort_by,
            top_k=plot_top_k,
        )

'''

if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"
    output_path = f"results/statistic_audit_row_extractor{'_wp' if use_data_wp else ''}.csv"

    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    require_all_keys = False
    ignore_case = False
    keep_line = False
    strip_quotes = True
    value_pattern = QUOTED_OR_BARE_VALUE_PATTERN

    labels = [label for label, _ in file_pairs]
    ordered_labels = group_labels_humans_first(labels)

    # Per-mode valid parameter choices
    mode_configs = {
        "execve": {
            "pair_options": [("a0", "a1"), ("a1", "a2"), ("a0", "argc")],
            "conditional_options": [
                ("a0", "tail", "a1"),
                ("a0", "cat", "a1"),
                ("a0", "grep", "a1"),
                ("a0", "ls", "a1"),
            ],
        },
        "path": {
            "pair_options": [("name", "nametype")],
            "conditional_options": [
                ("nametype", "NORMAL", "name"),
                ("nametype", "PARENT", "name"),
            ],
        },
        "syscall": {
            "pair_options": [("comm", "tty"), ("comm", "exe"), ("success", "syscall")],
            "conditional_options": [
                ("comm", "tail", "tty"),
                ("comm", "cat", "tty"),
                ("comm", "grep", "tty"),
                ("comm", "chmod", "tty"),
            ],
        },
        "sockaddr": {
            # sockaddr has only "saddr", so pair/conditional structure is not meaningful here
            # skip it in this script unless you redesign the features
            "pair_options": [],
            "conditional_options": [],
        },
    }

    distance_names = [
        "presence_pattern_distance",
        "pair_distribution_distance",
        "conditional_distribution_distance",
    ]

    active_modes = ["execve", "path", "syscall"]

    total_configs = sum(
        len(mode_configs[mode]["pair_options"]) * len(mode_configs[mode]["conditional_options"])
        for mode in active_modes
    )

    config_counter = 0

    for mode in active_modes:
        prefix, keys = get_mode_config(mode)

        pair_options = mode_configs[mode]["pair_options"]
        conditional_options = mode_configs[mode]["conditional_options"]

        for (pair_key1, pair_key2), (
            conditional_given_key,
            conditional_given_value,
            conditional_target_key,
        ) in product(pair_options, conditional_options):
            config_counter += 1

            print(
                f"\n[CONFIG {config_counter}/{total_configs}] "
                f"mode={mode} "
                f"pair=({pair_key1},{pair_key2}) "
                f"conditional=({conditional_given_key}={conditional_given_value} -> {conditional_target_key})"
            )

            pairwise_results = score_file_pairs(
                file_pairs,
                prefix_regex=prefix,
                keys=keys,
                value_pattern=value_pattern,
                require_all_keys=require_all_keys,
                ignore_case=ignore_case,
                keep_line=keep_line,
                strip_quotes=strip_quotes,
                pair_key1=pair_key1,
                pair_key2=pair_key2,
                conditional_given_key=conditional_given_key,
                conditional_given_value=conditional_given_value,
                conditional_target_key=conditional_target_key,
            )

            for sort_by in distance_names:
                print(f"  [DIST] sort_by={sort_by}")

                print_pairwise_distance_report(pairwise_results, sort_by=sort_by)

                matrix = build_symmetric_distance_matrix(
                    ordered_labels,
                    pairwise_results,
                    extract_pair=lambda item, sort_by=sort_by: (
                        item["label_1"],
                        item["label_2"],
                        item[sort_by]["distance"],
                    ),
                )

                group_stats = analyze_binary_group_structure(matrix, ordered_labels)
                print_binary_group_structure_report(group_stats)

                append_statistic_evaluation_row(
                    approach="audit_row_extractor",
                    distance_name=sort_by,
                    ordered_labels=ordered_labels,
                    hyperparameters={
                        "mode": mode,
                        "pair_key1": pair_key1,
                        "pair_key2": pair_key2,
                        "conditional_given_key": conditional_given_key,
                        "conditional_given_value": conditional_given_value,
                        "conditional_target_key": conditional_target_key,
                    },
                    group_stats=group_stats,
                    output_path=output_path,
                )
'''