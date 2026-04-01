#!/usr/bin/env python3
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
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


# -----------------------------
# Global
# -----------------------------

Metric = Literal["median", "average"]
Series = Literal["all", "cmd"]

# Example: msg=audit(1765984141.131:2375):
AUDIT_ID_RE = re.compile(r"msg=audit\((?P<ts>\d+(?:\.\d+)?):(?P<serial>\d+)\)")

# field pattern: key=value where value is either "quoted string" or a bare token
FIELD_RE_TEMPLATE = r"{key}=(?P<val>\"[^\"]*\"|\S+)"


# -----------------------------
# Functions
# -----------------------------

def extract_audit_id(line: str) -> Optional[Tuple[float, int]]:
    """Return (timestamp, serial) if line contains msg=audit(...:serial), else None."""
    m = AUDIT_ID_RE.search(line)
    if not m:
        return None
    return float(m.group("ts")), int(m.group("serial"))


def extract_field_from_line(line: str, key: str) -> Optional[str]:
    """Extract key=value from a single audit line (handles quoted strings)."""
    pattern = re.compile(FIELD_RE_TEMPLATE.format(key=re.escape(key)))
    m = pattern.search(line)
    if not m:
        return None
    val = m.group("val")
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        val = val[1:-1]
    return val


# -----------------------------
# Bundle model
# -----------------------------

@dataclass
class Bundle:
    ts: float
    serial: int
    lines: List[str] = field(default_factory=list)


# -----------------------------
# Reading
# -----------------------------

def read_bundles(path: str) -> List[Bundle]:
    bundles: Dict[Tuple[float, int], Bundle] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            info = extract_audit_id(line)
            if info is None:
                continue
            ts, serial = info
            key = (ts, serial)
            if key not in bundles:
                bundles[key] = Bundle(ts=ts, serial=serial, lines=[])
            bundles[key].lines.append(line)

    return sorted(bundles.values(), key=lambda b: (b.ts, b.serial))


# -----------------------------
# Bundle inspection
# -----------------------------

def has_execve(bundle: Bundle) -> bool:
    """True if bundle contains an EXECVE record."""
    return any("type=EXECVE" in line for line in bundle.lines)


def syscall_line(bundle: Bundle) -> Optional[str]:
    """Return the first SYSCALL line in the bundle, if any."""
    for line in bundle.lines:
        if "type=SYSCALL" in line:
            return line
    return None


def get_tty_exe_comm(bundle: Bundle) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract tty, exe, comm from the bundle's SYSCALL line (if present)."""
    sl = syscall_line(bundle)
    if sl is None:
        return None, None, None
    tty = extract_field_from_line(sl, "tty")
    exe = extract_field_from_line(sl, "exe")
    comm = extract_field_from_line(sl, "comm")
    return tty, exe, comm


def bundle_mentions_cmd(bundle: Bundle, cmd: str) -> bool:
    """
    Detect whether this bundle corresponds to running `cmd`.

    Checks:
      - SYSCALL comm
      - SYSCALL exe basename
      - EXECVE a0
    """
    cmd = os.path.basename(cmd)

    _, exe, comm = get_tty_exe_comm(bundle)

    if comm == cmd:
        return True
    if exe and os.path.basename(exe) == cmd:
        return True

    for line in bundle.lines:
        if line.startswith("type=EXECVE"):
            a0 = extract_field_from_line(line, "a0")
            if a0 == cmd or (a0 and os.path.basename(a0) == cmd):
                return True

    return False


# -----------------------------
# Filtering logic
# -----------------------------

def filter_bundles(bundles: List[Bundle]) -> List[Bundle]:
    """
    Keep only bundles that:
      1) have EXECVE
      2) have tty != "(none)" (interactive)
    """
    kept: List[Bundle] = []
    for b in bundles:
        if not has_execve(b):
            continue

        tty, _, _ = get_tty_exe_comm(b)

        if tty is None:
            continue
        if tty == "(none)":
            continue

        kept.append(b)

    return kept


# -----------------------------
# Clustering
# -----------------------------

def cluster_bundles(bundles: List[Bundle], cluster_window: float = 0.5) -> List[List[Bundle]]:
    """Cluster bundles by time proximity. Each cluster is a list of bundles."""
    if not bundles:
        return []

    bundles = sorted(bundles, key=lambda b: (b.ts, b.serial))

    clusters: List[List[Bundle]] = []
    current: List[Bundle] = [bundles[0]]
    last_ts = bundles[0].ts

    for b in bundles[1:]:
        if (b.ts - last_ts) <= cluster_window:
            current.append(b)
            last_ts = b.ts
        else:
            clusters.append(current)
            current = [b]
            last_ts = b.ts

    clusters.append(current)
    return clusters


def inter_event_deltas(timestamps: List[float]) -> List[float]:
    """Compute successive differences between sorted timestamps."""
    return [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]


def cmd_to_next_cluster_deltas(clusters: List[List[Bundle]], cmd: str) -> List[float]:
    """
    For each cluster that contains `cmd`, return delta to the NEXT cluster start time.
    Delta is (next_cluster_start - this_cluster_start).
    """
    if len(clusters) < 2:
        return []

    starts = [c[0].ts for c in clusters]

    out: List[float] = []
    for i, c in enumerate(clusters[:-1]):
        if any(bundle_mentions_cmd(b, cmd) for b in c):
            out.append(starts[i + 1] - starts[i])

    return out


# -----------------------------
# Stats helpers
# -----------------------------

def basic_stat(values: List[float]) -> Dict[str, float]:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return {}

    v = np.array(sorted(values), dtype=float)

    stats = {
        "min": float(v[0]),
        "median": float(np.median(v)),
        "max": float(v[-1]),
        "average": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "count": float(len(v)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) > 1 else 0.0,
    }
    return stats


# -----------------------------
# Distance helpers
# -----------------------------

def make_shared_bins(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bins: int = 30,
    log_bins: bool = True,
) -> np.ndarray:
    combined = np.concatenate([a, b])

    if len(combined) == 0:
        raise ValueError("No values to bin.")

    if log_bins:
        combined = combined[combined > 0]
        if len(combined) == 0:
            raise ValueError("No positive values for log bins.")
        vmin = combined.min()
        vmax = combined.max()
        if vmin == vmax:
            vmin = max(vmin * 0.9, 1e-12)
            vmax = vmax * 1.1
        return np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)

    vmin = combined.min()
    vmax = combined.max()
    if vmin == vmax:
        vmin = vmin - 0.5 if vmin != 0 else -0.5
        vmax = vmax + 0.5
    return np.linspace(vmin, vmax, bins + 1)


def histogram_probs(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(x, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def l1_hist_distance(
    a: List[float],
    b: List[float],
    *,
    bins: int = 30,
    log_bins: bool = True,
) -> float:
    a_arr = np.array([x for x in a if np.isfinite(x)], dtype=float)
    b_arr = np.array([x for x in b if np.isfinite(x)], dtype=float)

    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan")

    edges = make_shared_bins(a_arr, b_arr, bins=bins, log_bins=log_bins)
    p_a = histogram_probs(a_arr, edges)
    p_b = histogram_probs(b_arr, edges)
    return float(np.sum(np.abs(p_a - p_b)))


def ks_distance(
    a: List[float],
    b: List[float],
) -> float:
    a_arr = np.array([x for x in a if np.isfinite(x)], dtype=float)
    b_arr = np.array([x for x in b if np.isfinite(x)], dtype=float)

    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan")

    return float(ks_2samp(a_arr, b_arr).statistic)


def compute_pairwise_time_distances(
    result_1: Dict[str, Any],
    result_2: Dict[str, Any],
    *,
    bins_all: int = 30,
    bins_cmd: int = 30,
    log_bins_all: bool = True,
    log_bins_cmd: bool = True,
    min_n_cmd: int = 10,
) -> Dict[str, Any]:
    """
    Compute L1 and KS distances between two analysis results for:
      - all cluster deltas
      - command-specific cluster->next deltas

    Command-specific distances are only computed if both files have n >= min_n_cmd.
    """
    all_vals_1 = result_1.get("all", {}).get("values", [])
    all_vals_2 = result_2.get("all", {}).get("values", [])

    cmd_block_1 = result_1.get("cmd") or {}
    cmd_block_2 = result_2.get("cmd") or {}

    cmd_vals_1 = cmd_block_1.get("values", [])
    cmd_vals_2 = cmd_block_2.get("values", [])
    cmd_name_1 = cmd_block_1.get("name")
    cmd_name_2 = cmd_block_2.get("name")

    n_all_1 = len([v for v in all_vals_1 if np.isfinite(v)])
    n_all_2 = len([v for v in all_vals_2 if np.isfinite(v)])
    n_cmd_1 = len([v for v in cmd_vals_1 if np.isfinite(v)])
    n_cmd_2 = len([v for v in cmd_vals_2 if np.isfinite(v)])

    out = {
        "all": {
            "n_1": n_all_1,
            "n_2": n_all_2,
            "l1": l1_hist_distance(all_vals_1, all_vals_2, bins=bins_all, log_bins=log_bins_all),
            "ks": ks_distance(all_vals_1, all_vals_2),
        },
        "cmd": {
            "name_1": cmd_name_1,
            "name_2": cmd_name_2,
            "n_1": n_cmd_1,
            "n_2": n_cmd_2,
            "min_n_required": min_n_cmd,
            "valid": (n_cmd_1 >= min_n_cmd and n_cmd_2 >= min_n_cmd),
            "l1": float("nan"),
            "ks": float("nan"),
        },
    }

    if out["cmd"]["valid"]:
        out["cmd"]["l1"] = l1_hist_distance(
            cmd_vals_1,
            cmd_vals_2,
            bins=bins_cmd,
            log_bins=log_bins_cmd,
        )
        out["cmd"]["ks"] = ks_distance(cmd_vals_1, cmd_vals_2)

    return out


def score_file_pairs(
    paths: List[str],
    labels: List[str],
    *,
    cluster_window: float = 0.5,
    cmd: Optional[str] = None,
    bins_all: int = 30,
    bins_cmd: int = 30,
    log_bins_all: bool = True,
    log_bins_cmd: bool = True,
    min_n_cmd: int = 10,
) -> List[Dict[str, Any]]:
    """
    Analyze all files once, then compute pairwise L1/KS distances for 'all' and 'cmd'.
    """
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have the same length.")

    analyzed = {
        label: analyze_file(path, cluster_window=cluster_window, cmd=cmd)
        for label, path in zip(labels, paths)
    }

    results: List[Dict[str, Any]] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_1 = labels[i]
            label_2 = labels[j]

            dist = compute_pairwise_time_distances(
                analyzed[label_1],
                analyzed[label_2],
                bins_all=bins_all,
                bins_cmd=bins_cmd,
                log_bins_all=log_bins_all,
                log_bins_cmd=log_bins_cmd,
                min_n_cmd=min_n_cmd,
            )

            results.append({
                "label_1": label_1,
                "label_2": label_2,
                "distances": dist,
            })

    return results


def print_pairwise_distance_report(
    pairwise_results: List[Dict[str, Any]],
    *,
    sort_by: str = "all_l1",
) -> None:
    """
    sort_by options:
      - all_l1
      - all_ks
      - cmd_l1
      - cmd_ks
    """
    def sort_key(item: Dict[str, Any]) -> float:
        d = item["distances"]
        if sort_by == "all_l1":
            return d["all"]["l1"]
        if sort_by == "all_ks":
            return d["all"]["ks"]
        if sort_by == "cmd_l1":
            return d["cmd"]["l1"]
        if sort_by == "cmd_ks":
            return d["cmd"]["ks"]
        raise ValueError(f"Unknown sort_by: {sort_by}")

    sorted_items = sorted(
        pairwise_results,
        key=lambda x: (-np.inf if np.isnan(sort_key(x)) else sort_key(x)),
        reverse=True,
    )

    print(f"\nPairwise distances sorted by {sort_by}:")
    for item in sorted_items:
        l1 = item["distances"]["all"]["l1"]
        ks = item["distances"]["all"]["ks"]
        cmd_l1 = item["distances"]["cmd"]["l1"]
        cmd_ks = item["distances"]["cmd"]["ks"]
        valid = item["distances"]["cmd"]["valid"]

        print(
            f"{item['label_1']} vs {item['label_2']} | "
            f"ALL: L1={l1:.6f}, KS={ks:.6f} | "
            f"CMD(valid={valid}): L1={cmd_l1:.6f}, KS={cmd_ks:.6f}"
        )


# -----------------------------
# Analysis only
# -----------------------------

def analyze_file(path: str, cluster_window: float = 0.5, cmd: Optional[str] = None) -> Dict[str, Any]:
    """
    Pure analysis function: no plotting, no printing.
    """
    bundles = read_bundles(path)
    kept = filter_bundles(bundles)
    clusters = cluster_bundles(kept, cluster_window=cluster_window)

    cluster_starts = [c[0].ts for c in clusters]
    deltas = inter_event_deltas(cluster_starts)
    all_stats = basic_stat(deltas)

    cmd_deltas: List[float] = []
    cmd_stats: Dict[str, float] = {}

    if cmd is not None:
        cmd_deltas = cmd_to_next_cluster_deltas(clusters, cmd)
        cmd_stats = basic_stat(cmd_deltas)

    return {
        "meta": {
            "path": path,
            "cluster_window": cluster_window,
            "total_bundles": len(bundles),
            "kept_bundles": len(kept),
            "num_clusters": len(clusters),
        },
        "all": {
            "values": deltas,
            "n": len([v for v in deltas if np.isfinite(v)]),
            "stats": all_stats,
        },
        "cmd": {
            "name": cmd,
            "values": cmd_deltas,
            "n": len([v for v in cmd_deltas if np.isfinite(v)]),
            "stats": cmd_stats,
        } if cmd is not None else None,
    }


# -----------------------------
# Plot helpers
# -----------------------------

def compute_global_log_xlim(
    results: List[Dict[str, Any]],
    *,
    series: Series = "all",
) -> Optional[Tuple[float, float]]:
    """
    Compute one shared positive finite x-range for log-scale histograms
    across multiple analysis results.
    """
    vals: List[float] = []

    for res in results:
        if series == "all":
            values = res.get("all", {}).get("values", [])
        else:
            cmd_block = res.get("cmd")
            values = cmd_block.get("values", []) if cmd_block is not None else []

        vals.extend(v for v in values if np.isfinite(v) and v > 0)

    if not vals:
        return None

    vmin = min(vals)
    vmax = max(vals)

    if vmin == vmax:
        vmin = max(vmin * 0.9, 1e-12)
        vmax = vmax * 1.1

    return (vmin, vmax)


def plot_log_hist(
    values: List[float],
    title: str,
    xlabel: str,
    bins_n: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    values = [v for v in values if np.isfinite(v) and v > 0]
    if not values:
        print(f"[WARN] No positive finite values to plot for: {title}")
        return

    if xlim is not None:
        vmin, vmax = xlim
    else:
        vmin = min(values)
        vmax = max(values)
        if vmin == vmax:
            vmin = max(vmin * 0.9, 1e-12)
            vmax = vmax * 1.1

    bins = np.logspace(np.log10(vmin), np.log10(vmax), bins_n)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True)
    plt.xscale("log")
    plt.xlim(vmin, vmax)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    #rand = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    #plt.savefig(f"results/advanced_time_unique_{rand}.pdf")
    plt.show()
    


def print_analysis_summary(result: Dict[str, Any]) -> None:
    meta = result["meta"]
    all_block = result["all"]
    cmd_block = result.get("cmd")

    print(f"File: {meta['path']}")
    print(f"Total bundles parsed: {meta['total_bundles']}")
    print(f"Bundles after filters (EXECVE + tty != (none)): {meta['kept_bundles']}")
    print(f"After clustering (window={meta['cluster_window']:.3f}s): {meta['num_clusters']}")

    all_stats = all_block.get("stats", {})
    if all_stats:
        print("\nInter-event deltas (cluster->cluster)")
        print(f"  n      = {all_block['n']}")
        print(f"  min    = {all_stats['min']:.6f} s")
        print(f"  median = {all_stats['median']:.6f} s")
        print(f"  max    = {all_stats['max']:.6f} s")
        print(f"  avg    = {all_stats['average']:.6f} s")

    if cmd_block is not None:
        cmd_stats = cmd_block.get("stats", {})
        cmd_name = cmd_block.get("name")
        if cmd_stats:
            print(f"\n{cmd_name}->next-cluster deltas")
            print(f"  n      = {cmd_block['n']}")
            print(f"  min    = {cmd_stats['min']:.6f} s")
            print(f"  median = {cmd_stats['median']:.6f} s")
            print(f"  max    = {cmd_stats['max']:.6f} s")
            print(f"  avg    = {cmd_stats['average']:.6f} s")


def plot_analysis_result(
    result: Dict[str, Any],
    *,
    plot_all: bool = True,
    plot_cmd: bool = True,
    bins_all: int = 50,
    bins_cmd: int = 40,
    print_summary: bool = True,
    print_first_cmd_deltas: int = 20,
    xlim_all: Optional[Tuple[float, float]] = None,
    xlim_cmd: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot helper that consumes analyze_file(...) output.
    """
    if print_summary:
        print_analysis_summary(result)

    all_block = result["all"]
    cmd_block = result.get("cmd")

    if plot_all:
        plot_log_hist(
            all_block["values"],
            title="Histogram of inter-event times (cluster->cluster)",
            xlabel="Inter-event time (seconds, log scale)",
            bins_n=bins_all,
            xlim=xlim_all,
        )

    if plot_cmd and cmd_block is not None and cmd_block["values"]:
        cmd_name = cmd_block["name"]

        plot_log_hist(
            cmd_block["values"],
            title=f"Histogram: {cmd_name} cluster -> next cluster time",
            xlabel=f"Time after {cmd_name} to next action (seconds, log scale)",
            bins_n=bins_cmd,
            xlim=xlim_cmd,
        )

        if print_first_cmd_deltas > 0:
            print(f"\nFirst {min(print_first_cmd_deltas, len(cmd_block['values']))} {cmd_name}->next deltas:")
            for i, d in enumerate(cmd_block["values"][:print_first_cmd_deltas]):
                print(f"  {i:02d}: {d:.6f} s")


def analyze_and_plot_file(
    path: str,
    *,
    cluster_window: float = 0.5,
    cmd: Optional[str] = None,
    plot_all: bool = True,
    plot_cmd: bool = True,
    print_summary: bool = True,
    xlim_all: Optional[Tuple[float, float]] = None,
    xlim_cmd: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: analyze first, then plot.
    """
    result = analyze_file(path, cluster_window=cluster_window, cmd=cmd)
    plot_analysis_result(
        result,
        plot_all=plot_all,
        plot_cmd=plot_cmd,
        print_summary=print_summary,
        xlim_all=xlim_all,
        xlim_cmd=xlim_cmd,
    )
    return result


# -----------------------------
# Cross-file comparison plot
# -----------------------------

def compare_files_plot(
    paths: List[str],
    labels: List[str],
    *,
    cluster_window: float = 0.5,
    metric: Metric = "median",
    series: Series = "all",
    cmd: Optional[str] = None,
    title: Optional[str] = None,
    dev_metric: str = "std",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    y: List[float] = []
    yerr: List[float] = []

    for p in paths:
        res = analyze_file(p, cluster_window=cluster_window, cmd=cmd)
        results.append(res)

        if series == "all":
            stats = res.get("all", {}).get("stats", {})
        elif series == "cmd":
            cmd_block = res.get("cmd") or {}
            stats = cmd_block.get("stats", {})
        else:
            stats = {}

        y.append(stats.get(metric, float("nan")))
        yerr.append(stats.get(dev_metric, float("nan")))

    plt.figure(figsize=(max(6, 0.8 * len(labels)), 4))
    plt.bar(labels, y, yerr=yerr, capsize=4)
    ylabel_series = "all" if series == "all" else (cmd or "cmd")
    plt.ylabel(f"{ylabel_series}.{metric} (seconds)")
    plt.xlabel("File")
    series_label = "all" if series == "all" else (cmd or "cmd")
    plt.title(title or f"{series_label} {metric} by file (cluster_window={cluster_window}s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return results


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"

    # optionally plotting
    selected_labels = ["Armin", "GPT4.1"]


    # 1. score=2.000000 | distance=all_l1 | hyperparameters: bins_all=10, cluster_window=0.5, dataset=Data, log_bins_all=True
    # 2. score=2.500000 | distance=all_l1 | hyperparameters: bins_all=10, cluster_window=2, dataset=Data, log_bins_all=True
    # 10. score=10.900000 | distance=cmd_l1 | hyperparameters: bins_cmd=10, cluster_window=1, cmd=tail, dataset=Data, log_bins_cmd=True
    ### Hyperparameter
    cluster_window = 0.5  # worth changing: try 0.5, 1, 2, 5 to change how aggressively nearby events are merged
    cmd = "tail"  # worth changing: try "curl", "cat", "tail", "grep", "chmod", depending on the behavior you want to isolate
    bins_all = 10  # worth changing: try 10, 25, 50 to vary the granularity of all-event histogram distances
    bins_cmd = bins_all  # worth changing: keep it same as bins_all
    log_bins_all = True  # worth changing: True if times span orders of magnitude, False for linear timing comparison
    log_bins_cmd = log_bins_all  # worth changing: same idea for the command-specific series
    sort_by = "all_l1"  # worth changing: "all_l1", "all_ks", "cmd_l1", "cmd_ks"
    # don't change this
    min_n_cmd = 10

    
    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    labels = [label for label, _ in file_pairs]
    paths = [path for _, path in file_pairs]

    pairwise_results = score_file_pairs(
        paths,
        labels,
        cluster_window=cluster_window,
        cmd=cmd,
        bins_all=bins_all,
        bins_cmd=bins_cmd,
        log_bins_all=log_bins_all,
        log_bins_cmd=log_bins_cmd,
        min_n_cmd=min_n_cmd,
    )

    print_pairwise_distance_report(pairwise_results, sort_by=sort_by)

    sort_mapping = {
        "all_l1": lambda item: item["distances"]["all"]["l1"],
        "all_ks": lambda item: item["distances"]["all"]["ks"],
        "cmd_l1": lambda item: item["distances"]["cmd"]["l1"],
        "cmd_ks": lambda item: item["distances"]["cmd"]["ks"],
    }
    ordered_labels = group_labels_humans_first(labels)
    matrix = build_symmetric_distance_matrix(
        ordered_labels,
        pairwise_results,
        extract_pair=lambda item: (
            item["label_1"],
            item["label_2"],
            sort_mapping[sort_by](item),
        ),
    )
    group_stats = analyze_binary_group_structure(matrix, ordered_labels)
    print_binary_group_structure_report(group_stats)

    plot_distance_heatmap(matrix, ordered_labels, title=f"Advanced time {sort_by} heatmap")
    plot_mds_embedding(matrix, ordered_labels, title=f"Advanced time {sort_by} MDS")

    # ---------------------------------
    # Additional: compare 2 selected log files with shared x-axis
    # ---------------------------------
    selected_file_pairs = [(label, path) for label, path in file_pairs if label in selected_labels]

    selected_results = [
        analyze_file(path, cluster_window=cluster_window, cmd=cmd)
        for label, path in selected_file_pairs
    ]

    shared_xlim_all = compute_global_log_xlim(selected_results, series="all")

    for (label, _), result in zip(selected_file_pairs, selected_results):
        plot_analysis_result(
            result,
            plot_all=True,
            plot_cmd=False,
            print_summary=True,
            xlim_all=shared_xlim_all,
        )

'''

if __name__ == "__main__":
    use_data_wp = True
    dataset = "Data_WP" if use_data_wp else "Data"
    output_path = f"results/statistic_audit_advanced_time{'_wp' if use_data_wp else ''}.csv"

    min_n_cmd = 10
    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    labels = [label for label, _ in file_pairs]
    paths = [path for _, path in file_pairs]
    ordered_labels = group_labels_humans_first(labels)

    ########################################
    # EXPERIMENT 1: overall timing behavior
    ########################################

    configs_all = list(product(
        [0.5, 1, 2, 5],        # cluster_window
        [10, 25, 50],          # bins
        [True, False],         # log_bins
        ["all_l1", "all_ks"],  # metric
    ))

    for i, (cluster_window, bins_all, log_bins_all, sort_by) in enumerate(configs_all, 1):
        bins_cmd = bins_all
        log_bins_cmd = log_bins_all

        print(
            f"[ALL {i}/{len(configs_all)}] "
            f"dataset={dataset} "
            f"cluster_window={cluster_window} "
            f"bins={bins_all} "
            f"log_bins={log_bins_all} "
            f"metric={sort_by}",
            end=" ... ",
            flush=True,
        )

        pairwise_results = score_file_pairs(
            paths,
            labels,
            cluster_window=cluster_window,
            cmd=None,
            bins_all=bins_all,
            bins_cmd=bins_cmd,
            log_bins_all=log_bins_all,
            log_bins_cmd=log_bins_cmd,
            min_n_cmd=min_n_cmd,
        )

        sort_mapping = {
            "all_l1": lambda item: item["distances"]["all"]["l1"],
            "all_ks": lambda item: item["distances"]["all"]["ks"],
        }

        matrix = build_symmetric_distance_matrix(
            ordered_labels,
            pairwise_results,
            extract_pair=lambda item: (
                item["label_1"],
                item["label_2"],
                sort_mapping[sort_by](item),
            ),
        )

        group_stats = analyze_binary_group_structure(matrix, ordered_labels)

        append_statistic_evaluation_row(
            approach="audit_advanced_time_all",
            distance_name=sort_by,
            ordered_labels=ordered_labels,
            hyperparameters={
                "dataset": dataset,
                "cluster_window": cluster_window,
                "bins_all": bins_all,
                "log_bins_all": log_bins_all,
            },
            group_stats=group_stats,
            output_path=output_path,
        )

        print("written")


    ########################################
    # EXPERIMENT 2: command reaction timing
    ########################################

    configs_cmd = list(product(
        [0.5, 1, 2, 5],                                   # cluster_window
        ["tail"],  # command
        [10, 25, 50],                                     # bins
        [True, False],                                    # log_bins
        ["cmd_l1", "cmd_ks"],                             # metric
    ))

    for i, (cluster_window, cmd, bins_all, log_bins_all, sort_by) in enumerate(configs_cmd, 1):
        bins_cmd = bins_all
        log_bins_cmd = log_bins_all

        print(
            f"[CMD {i}/{len(configs_cmd)}] "
            f"dataset={dataset} "
            f"cluster_window={cluster_window} "
            f"cmd={cmd} "
            f"bins={bins_all} "
            f"log_bins={log_bins_all} "
            f"metric={sort_by}",
            end=" ... ",
            flush=True,
        )

        pairwise_results = score_file_pairs(
            paths,
            labels,
            cluster_window=cluster_window,
            cmd=cmd,
            bins_all=bins_all,
            bins_cmd=bins_cmd,
            log_bins_all=log_bins_all,
            log_bins_cmd=log_bins_cmd,
            min_n_cmd=min_n_cmd,
        )

        invalid_pairs = [
            item for item in pairwise_results
            if not item["distances"]["cmd"]["valid"]
        ]

        if invalid_pairs:
            invalid_labels = sorted({
                item["label_1"]
                for item in invalid_pairs
            } | {
                item["label_2"]
                for item in invalid_pairs
            })

            print(
                f"skipped (invalid cmd coverage; {len(invalid_pairs)} invalid pairs; "
                f"affected labels={invalid_labels})"
            )
            continue

        sort_mapping = {
            "cmd_l1": lambda item: item["distances"]["cmd"]["l1"],
            "cmd_ks": lambda item: item["distances"]["cmd"]["ks"],
        }

        matrix = build_symmetric_distance_matrix(
            ordered_labels,
            pairwise_results,
            extract_pair=lambda item: (
                item["label_1"],
                item["label_2"],
                sort_mapping[sort_by](item),
            ),
        )

        group_stats = analyze_binary_group_structure(matrix, ordered_labels)

        append_statistic_evaluation_row(
            approach="audit_advanced_time_cmd",
            distance_name=sort_by,
            ordered_labels=ordered_labels,
            hyperparameters={
                "dataset": dataset,
                "cluster_window": cluster_window,
                "cmd": cmd,
                "bins_cmd": bins_cmd,
                "log_bins_cmd": log_bins_cmd,
            },
            group_stats=group_stats,
            output_path=output_path,
        )

        print("written")
'''
