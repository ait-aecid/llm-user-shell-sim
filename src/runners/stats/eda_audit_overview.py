from __future__ import annotations

"""Generate visual EDA figures contrasting AI vs human audit-log behavior.

Run as a module from the repo root, e.g.::

    python -m src.runners.stats.eda_audit_overview --dataset both

For each dataset the runner extracts per-actor feature families
(:mod:`src.stats_tools.audit_eda`) and writes one PNG per metric plus a
``summary.csv`` of per-actor values into ``results/eda/<dataset>/``.

These figures are **descriptive** EDA over a tiny number of actors
(Nextcloud: 4 AI / 6 human, WordPress: 4 / 4). A class effect can be driven by a
single actor, so every distribution plot overlays the per-actor values and every
figure carries that caveat in its subtitle. No significance testing is implied.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.stats_tools.audit_eda import ActorFeatures, extract_dataset

AI_COLOR = "#d1495b"
HUMAN_COLOR = "#2e86ab"
CAVEAT = "Descriptive EDA; small-n actors (per-actor points overlaid), not inferential."


def _color(is_ai: bool) -> str:
    return AI_COLOR if is_ai else HUMAN_COLOR


def _split(feats: list[ActorFeatures]):
    ai = [f for f in feats if f.is_ai]
    hum = [f for f in feats if not f.is_ai]
    return hum, ai


def _finish(fig, ax, save_dir: Path, name: str, dpi: int) -> None:
    ax.text(0.0, -0.13, CAVEAT, transform=ax.transAxes, fontsize=7, color="gray")
    fig.tight_layout()
    out = save_dir / f"{name}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---- generic chart builders ----

def _box_by_class(feats, getter, title, ylabel, save_dir, name, dpi, logy=False):
    """Box-by-class with per-actor jittered points and actor labels."""
    hum, ai = _split(feats)
    hum_v = [getter(f) for f in hum]
    ai_v = [getter(f) for f in ai]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot([hum_v, ai_v], tick_labels=["Human", "AI"], widths=0.5,
               showfliers=False, medianprops=dict(color="black"))
    for i, (vals, group) in enumerate([(hum_v, hum), (ai_v, ai)], start=1):
        x = np.random.normal(i, 0.05, size=len(vals))
        col = HUMAN_COLOR if i == 1 else AI_COLOR
        ax.scatter(x, vals, color=col, alpha=0.85, zorder=3, s=40, edgecolor="white")
        for xi, v, f in zip(x, vals, group):
            ax.annotate(f.actor, (xi, v), fontsize=6, alpha=0.7,
                        xytext=(4, 0), textcoords="offset points", va="center")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _finish(fig, ax, save_dir, name, dpi)


def _ecdf_by_class(feats, pool_getter, title, xlabel, save_dir, name, dpi, logx=True):
    """Overlaid per-actor ECDFs, colored by class."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for f in feats:
        vals = [v for v in pool_getter(f) if v > 0] if logx else list(pool_getter(f))
        if len(vals) < 2:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=_color(f.is_ai), alpha=0.7, lw=1.3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative fraction of events")
    ax.set_title(title)
    ax.plot([], [], color=HUMAN_COLOR, label="Human")
    ax.plot([], [], color=AI_COLOR, label="AI")
    ax.legend(loc="lower right", fontsize=8)
    _finish(fig, ax, save_dir, name, dpi)


def _share_diff_bar(feats, share_getter, title, save_dir, name, dpi, top=15):
    """Ranked AI-minus-human share difference for a categorical distribution."""
    hum, ai = _split(feats)

    def _mean_share(group):
        agg = Counter()
        for f in group:
            for k, v in share_getter(f).items():
                agg[k] += v
        n = len(group) or 1
        return {k: v / n for k, v in agg.items()}

    hs, as_ = _mean_share(hum), _mean_share(ai)
    keys = set(hs) | set(as_)
    diffs = sorted(((k, as_.get(k, 0) - hs.get(k, 0)) for k in keys),
                   key=lambda kv: abs(kv[1]), reverse=True)[:top]
    diffs.sort(key=lambda kv: kv[1])
    labels = [k for k, _ in diffs]
    vals = [v for _, v in diffs]
    colors = [AI_COLOR if v > 0 else HUMAN_COLOR for v in vals]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(labels) + 1)))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("share difference  (AI − Human)   → AI-leaning")
    ax.set_title(title)
    _finish(fig, ax, save_dir, name, dpi)


def _scatter_span_rate(feats, save_dir, dpi):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for f in feats:
        ax.scatter(f.volume["span_min"], f.volume["events_per_min"],
                   color=_color(f.is_ai), s=60, edgecolor="white", zorder=3)
        ax.annotate(f.actor, (f.volume["span_min"], f.volume["events_per_min"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("session span (min)")
    ax.set_ylabel("events per minute")
    ax.set_title("Session length vs event rate")
    ax.plot([], [], "o", color=HUMAN_COLOR, label="Human")
    ax.plot([], [], "o", color=AI_COLOR, label="AI")
    ax.legend(fontsize=8)
    _finish(fig, ax, save_dir, "A_span_vs_rate", dpi)


def _tool_usage_bar(feats, save_dir, dpi):
    """Per-class mean usage rate of each interactive tool."""
    hum, ai = _split(feats)
    tools = list(next(iter(feats)).commands["_tool_counts"].keys())

    def _rate(group, tool):
        return np.mean([f.commands["_tool_counts"].get(tool, 0) / (f.commands["_cmd_total"] or 1)
                        for f in group]) if group else 0.0

    hvals = [_rate(hum, t) for t in tools]
    avals = [_rate(ai, t) for t in tools]
    x = np.arange(len(tools))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, hvals, width=0.4, color=HUMAN_COLOR, label="Human")
    ax.bar(x + 0.2, avals, width=0.4, color=AI_COLOR, label="AI")
    ax.set_xticks(x)
    ax.set_xticklabels(tools, rotation=45, ha="right")
    ax.set_ylabel("mean fraction of commands")
    ax.set_title("Interactive-tool usage rate by class")
    ax.legend(fontsize=8)
    _finish(fig, ax, save_dir, "C_interactive_tool_usage", dpi)


# ---- summary CSV ----

def _write_summary(feats: list[ActorFeatures], save_dir: Path) -> None:
    rows = []
    for f in feats:
        row = {"actor": f.actor, "class": "AI" if f.is_ai else "Human"}
        for fam in (f.volume, f.timing, f.commands, f.syscall, f.sequence, f.session, f.mitre, f.complexity):
            for k, v in fam.items():
                if not k.startswith("_"):
                    row[k] = v
        rows.append(row)
    if not rows:
        return
    cols = ["actor", "class"] + [c for c in rows[0] if c not in ("actor", "class")]
    with (save_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# ---- driver ----

def run_dataset(dataset: str, save_root: Path, dpi: int) -> None:
    feats = extract_dataset(dataset)
    if not feats:
        print(f"[skip] {dataset}: no parseable actors")
        return
    save_dir = save_root / dataset
    save_dir.mkdir(parents=True, exist_ok=True)

    # A: volume / composition
    _scatter_span_rate(feats, save_dir, dpi)
    _box_by_class(feats, lambda f: f.volume["events_per_min"], "Event rate", "events/min", save_dir, "A_event_rate", dpi)
    _box_by_class(feats, lambda f: f.volume["ratio_path_syscall"], "PATH : SYSCALL ratio", "ratio", save_dir, "A_path_syscall_ratio", dpi)

    # B: timing / rhythm
    _ecdf_by_class(feats, lambda f: f.timing["_pool_event_gaps"], "Inter-event delay (ECDF per actor)", "delay (s)", save_dir, "B_event_gap_ecdf", dpi)
    _ecdf_by_class(feats, lambda f: f.timing["_pool_cmd_gaps"], "Inter-command delay (ECDF per actor)", "delay (s)", save_dir, "B_cmd_gap_ecdf", dpi)
    _box_by_class(feats, lambda f: f.timing["frac_sub100ms_gaps"], "Fraction of sub-100ms event gaps", "fraction", save_dir, "B_sub100ms_fraction", dpi)
    _box_by_class(feats, lambda f: f.timing["long_pause_count_30s"], "Long inter-command pauses (>30s)", "count", save_dir, "B_long_pause_count", dpi)

    # C: command vocabulary
    _box_by_class(feats, lambda f: f.commands["distinct_commands"], "Distinct commands used", "count", save_dir, "C_distinct_commands", dpi)
    _box_by_class(feats, lambda f: f.commands["interactive_tool_rate"], "Interactive-tool usage rate", "fraction of commands", save_dir, "C_interactive_tool_rate", dpi)
    _box_by_class(feats, lambda f: f.commands["mean_argc"], "Mean argument count per command", "mean argc", save_dir, "C_mean_argc", dpi)
    _box_by_class(feats, lambda f: f.commands["chained_command_rate"], "Chained-command rate (pipes/&&/;/redirects)", "fraction", save_dir, "C_chained_rate", dpi)
    _box_by_class(feats, lambda f: f.commands["repeat_command_rate"], "Repeated-command rate", "fraction", save_dir, "C_repeat_rate", dpi)
    _tool_usage_bar(feats, save_dir, dpi)

    # D: syscall fingerprint
    _share_diff_bar(feats, lambda f: f.syscall["_syscall_share"], "Syscall share difference (AI vs Human)", save_dir, "D_syscall_share_diff", dpi)
    _box_by_class(feats, lambda f: f.syscall["syscall_entropy_bits"], "Syscall distribution entropy", "bits", save_dir, "D_syscall_entropy", dpi)
    _box_by_class(feats, lambda f: f.syscall["fail_rate"], "Syscall failure rate", "fraction", save_dir, "D_fail_rate", dpi)
    _box_by_class(feats, lambda f: f.syscall["fileop_comm_rate"], "File-operation command rate", "fraction", save_dir, "D_fileop_rate", dpi)
    _box_by_class(feats, lambda f: f.syscall["privilege_syscall_rate"], "Privilege-transition syscall rate", "fraction", save_dir, "D_privilege_rate", dpi)

    # E: sequence / ordering
    _box_by_class(feats, lambda f: f.sequence["command_bigram_entropy_bits"], "Command bigram entropy", "bits", save_dir, "E_bigram_entropy", dpi)
    _box_by_class(feats, lambda f: f.sequence["novel_bigram_rate"], "Novel command-bigram rate", "fraction", save_dir, "E_novel_bigram_rate", dpi)
    _box_by_class(feats, lambda f: f.sequence["command_gzip_ratio"], "Command-stream compressibility (gzip ratio)", "compressed/raw", save_dir, "E_gzip_ratio", dpi)

    # F: session / process-tree
    _box_by_class(feats, lambda f: f.session["interactive_tty_rate"], "Interactive-tty event rate", "fraction", save_dir, "F_interactive_tty_rate", dpi)
    _box_by_class(feats, lambda f: f.session["mean_proc_branching"], "Process-tree branching factor", "mean children/parent", save_dir, "F_proc_branching", dpi)

    # G: MITRE keys
    _share_diff_bar(feats, lambda f: f.mitre["_key_share"], "MITRE technique-key share difference (AI vs Human)", save_dir, "G_mitre_key_share_diff", dpi)

    # H: complexity indices (template cluster-id distributions)
    for key in sorted(next(iter(feats)).complexity):
        label = key.removeprefix("complexity_")
        _box_by_class(feats, lambda f, k=key: f.complexity[k], f"Complexity index: {label}", label, save_dir, f"H_{key}", dpi)

    _write_summary(feats, save_dir)
    n_ai = sum(f.is_ai for f in feats)
    print(f"[ok] {dataset}: {len(feats)} actors ({n_ai} AI / {len(feats)-n_ai} human) -> {save_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual EDA of audit logs: AI vs human signatures.")
    p.add_argument("--dataset", choices=["Nextcloud", "WordPress", "both"], default="both")
    p.add_argument("--save_dir", default="results/eda", help="Output root for PNGs + summary.csv.")
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_dir)
    datasets = ["Nextcloud", "WordPress"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, save_root, args.dpi)


if __name__ == "__main__":
    main()
