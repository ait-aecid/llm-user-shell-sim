#!/usr/bin/env python3
"""Analyze timing structure in audit logs and compare actor-specific distributions.

The script groups audit records into execution bundles and short time-window clusters,
then visualizes inter-cluster delays either globally or conditioned on a command.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np

from src.core.stats.data_catalog import get_log_path, analysis_actors


AUDIT_ID_RE = re.compile(r"msg=audit\((?P<ts>\d+(?:\.\d+)?):(?P<serial>\d+)\)")
FIELD_RE_TEMPLATE = r"{key}=(?P<val>\"[^\"]*\"|\S+)"


def extract_audit_id(line: str) -> Optional[Tuple[float, int]]:
    """Extract the audit timestamp and serial number from a raw log line.

    These two fields uniquely identify one audit event bundle. Returns `None`
    when the line does not contain a parseable audit identifier.
    """
    m = AUDIT_ID_RE.search(line)
    if not m:
        return None
    return float(m.group("ts")), int(m.group("serial"))


def extract_field_from_line(line: str, key: str) -> Optional[str]:
    """Read a single key-value field from an audit log line.

    Quoted values are unwrapped so downstream comparisons can treat quoted and
    unquoted fields uniformly. Returns `None` if the field is absent.
    """
    pattern = re.compile(FIELD_RE_TEMPLATE.format(key=re.escape(key)))
    m = pattern.search(line)
    if not m:
        return None
    val = m.group("val")
    if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
        val = val[1:-1]
    return val


@dataclass
class Bundle:
    ts: float
    serial: int
    lines: List[str] = field(default_factory=list)


def read_bundles(path: str) -> List[Bundle]:
    """Group raw audit lines into bundles keyed by audit timestamp and serial.

    Audit events are emitted across multiple lines, so analysis operates on the
    reconstructed bundle rather than on individual records.
    """
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


def has_execve(bundle: Bundle) -> bool:
    """Return whether the bundle contains an EXECVE record."""
    return any("type=EXECVE" in line for line in bundle.lines)


def syscall_line(bundle: Bundle) -> Optional[str]:
    """Return the SYSCALL line for a bundle when present."""
    for line in bundle.lines:
        if "type=SYSCALL" in line:
            return line
    return None


def get_tty_exe_comm(bundle: Bundle) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract basic execution context from the bundle's SYSCALL record.

    The tuple contains `(tty, exe, comm)` and is used to distinguish
    interactive command execution from background or system activity.
    """
    sl = syscall_line(bundle)
    if sl is None:
        return None, None, None
    tty = extract_field_from_line(sl, "tty")
    exe = extract_field_from_line(sl, "exe")
    comm = extract_field_from_line(sl, "comm")
    return tty, exe, comm


def bundle_mentions_cmd(bundle: Bundle, cmd: str) -> bool:
    """Check whether a bundle corresponds to the requested command.

    Matching is intentionally permissive across `comm`, `exe`, and `EXECVE a0`
    because audit records are not fully consistent across environments.
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


def filter_bundles(bundles: List[Bundle]) -> List[Bundle]:
    """Keep only interactive command bundles relevant for timing analysis.

    Bundles without `EXECVE` or without an attached TTY are excluded to avoid
    mixing user-driven activity with background audit noise.
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


def cluster_bundles(bundles: List[Bundle], cluster_window: float = 0.5) -> List[List[Bundle]]:
    """Merge nearby bundles into temporal clusters.

    The fixed window treats short bursts of related audit activity as one event,
    which stabilizes the timing distribution against audit line granularity.
    """
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
    """Compute consecutive delays between ordered timestamps."""
    return [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]


def cmd_to_next_cluster_deltas(clusters: List[List[Bundle]], cmd: str) -> List[float]:
    """Measure time from command-containing clusters to the next cluster start.

    This isolates the delay immediately following one command of interest rather
    than the full inter-event distribution.
    """
    if len(clusters) < 2:
        return []

    starts = [c[0].ts for c in clusters]

    out: List[float] = []
    for i, c in enumerate(clusters[:-1]):
        if any(bundle_mentions_cmd(b, cmd) for b in c):
            out.append(starts[i + 1] - starts[i])

    return out


def analyze_file(path: str, cluster_window: float = 0.5, cmd: Optional[str] = None) -> Dict[str, Any]:
    """Run the full timing analysis pipeline for one audit log.

    The returned structure contains metadata, all inter-cluster delays, and an
    optional command-conditioned delay series for downstream plotting.
    """
    # ---- Parse and structure audit activity ----
    bundles = read_bundles(path)
    kept = filter_bundles(bundles)
    clusters = cluster_bundles(kept, cluster_window=cluster_window)

    # ---- Derive timing series ----
    cluster_starts = [c[0].ts for c in clusters]
    deltas = inter_event_deltas(cluster_starts)

    cmd_deltas: List[float] = []
    if cmd is not None:
        cmd_deltas = cmd_to_next_cluster_deltas(clusters, cmd)

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
        },
        "cmd": {
            "name": cmd,
            "values": cmd_deltas,
        } if cmd is not None else None,
    }


def compute_global_log_xlim(
    results: List[Dict[str, Any]],
    *,
    series: str = "all",
) -> Optional[Tuple[float, float]]:
    """Compute shared log-scale x-limits across multiple result sets.

    A common axis range makes cross-actor histograms visually comparable.
    Returns `None` when no positive finite values are available.
    """
    vals: List[float] = []

    for res in results:
        if series == "all":
            values = res.get("all", {}).get("values", [])
        elif series == "cmd":
            cmd_block = res.get("cmd")
            values = cmd_block.get("values", []) if cmd_block is not None else []
        else:
            raise ValueError(f"Unknown series: {series}")

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
    *,
    label: str,
    xlabel: str,
    bins_n: int = 50,
    xlim: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot a density histogram on a logarithmic time axis.

    Non-positive and non-finite values are discarded because the visualization
    is defined on log-scaled delays only.
    """
    values = [v for v in values if np.isfinite(v) and v > 0]
    if not values:
        print(f"[WARN] No positive finite values to plot for: {label}")
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
    plt.title(label)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def anonymize_actor_labels(actor_names: List[str]) -> Dict[str, str]:
    """Map actor names to display labels for publication-friendly plots.

    Human participants are anonymized sequentially, while GPT-labelled actors
    remain identifiable to preserve the comparison of interest.
    """
    mapping: Dict[str, str] = {}
    human_idx = 1

    for name in actor_names:
        if "gpt" in name.lower():
            mapping[name] = name
        elif name not in mapping:
            mapping[name] = f"Human {human_idx}"
            human_idx += 1

    return mapping


def plot_selected_actors(
    selected_file_pairs: List[Tuple[str, str]],
    *,
    cluster_window: float,
    cmd: Optional[str],
    series: str = "all",
    bins_n: int = 50,
    save_dir: Optional[str] = None,
) -> None:
    """Plot per-actor timing histograms with shared scaling.

    Each selected log is analyzed independently, but the plots reuse a common
    x-range so actor-specific distributions can be compared directly.
    """
    # ---- Analyze all selected logs before plotting ----
    selected_results = [
        analyze_file(path, cluster_window=cluster_window, cmd=cmd)
        for _, path in selected_file_pairs
    ]

    shared_xlim = compute_global_log_xlim(selected_results, series=series)

    actor_names = [label for label, _ in selected_file_pairs]
    display_names = anonymize_actor_labels(actor_names)

    # ---- Render actor-level histograms ----
    for (label, _), result in zip(selected_file_pairs, selected_results):
        display_label = display_names[label]

        if series == "all":
            values = result["all"]["values"]
            xlabel = "Inter-event time (seconds, log scale)"
            title = f"{display_label}"
            save_path = f"{save_dir}/{display_label}_all.pdf" if save_dir else None
        elif series == "cmd":
            cmd_block = result.get("cmd")
            values = cmd_block["values"] if cmd_block is not None else []
            xlabel = f"Time after {cmd} to next cluster (seconds, log scale)"
            title = f"{display_label} - {cmd} to next cluster"
            save_path = f"{save_dir}/{display_label}_{cmd}.pdf" if save_dir else None
        else:
            raise ValueError(f"Unknown series: {series}")

        plot_log_hist(
            values,
            label=title,
            xlabel=xlabel,
            bins_n=bins_n,
            xlim=shared_xlim,
            save_path=save_path,
        )


if __name__ == "__main__":
    # ---- Configuration ----
    use_wordpress = True
    dataset = "WordPress" if use_wordpress else "Nextcloud"

    selected_labels = ["Armin", "Hotti", "Torina", "GPT4.1", "GPT4.1_V2", "Hotti"]

    cluster_window = 0.5
    cmd = "tail"

    file_pairs = [
        (actor, str(get_log_path(actor, "audit", dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    # Restrict plotting to the actors discussed in the current comparison.
    selected_file_pairs = [
        (label, path)
        for label, path in file_pairs
        if label in selected_labels
    ]

    # ---- Plot selected comparison ----
    plot_selected_actors(
        selected_file_pairs,
        cluster_window=cluster_window,
        cmd=cmd,
        series="all",   # or "cmd"
        bins_n=50,
        save_dir="results",
    )


