from __future__ import annotations

"""Paper-quality PDF figures for the evaluation sections.

Run from the repo root::

    python -m src.runners.stats.eda_paper_figures

Writes four vector PDFs into ``paper/latex/`` (no titles, no caveat
footers, no per-point actor labels -- captions live in the LaTeX source).
Screen-mode EDA output (``eda_audit_overview``) is unchanged.
"""

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.core.shared.actor_catalog import is_ai_actor
from src.core.stats.data_catalog import analysis_actors, get_log_path
from src.runners.stats.eda_null_overview import feature_matrix, run_null
from src.stats_tools.audit_eda import _command_stream, extract_dataset, parse_audit_file

AI_COLOR = "#d1495b"
HUMAN_COLOR = "#2e86ab"
NULL_COLOR = "#8da0ab"
OUT = Path("paper/latex")

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
})


def _cmd_gap_hist(all_feats: dict) -> None:
    """Pooled inter-command-delay distribution, AI vs human, per dataset.

    Grouped log-spaced histogram: within each bin, human occupies the left
    half and AI the right half, each normalized to the fraction of that
    class's command gaps. The AI pile-up at the 10 s delay cap shows as one
    tall bar. Outliers are clipped into the end bins so the human long-reading
    tail is not silently dropped (ponytail: clip, not a real spike at 1000 s).
    """
    bins = np.logspace(-1, 3, 41)  # 0.1 s .. 1000 s, 40 log bins (half-width)
    left, right = bins[:-1], bins[1:]
    mid = np.sqrt(left * right)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (ds, feats) in zip(axes, all_feats.items()):
        for grp_is_ai, col, label, x0, x1 in ((False, HUMAN_COLOR, "Human", left, mid),
                                              (True, AI_COLOR, "AI", mid, right)):
            vals = [v for f in feats if f.is_ai == grp_is_ai
                    for v in f.timing["_pool_cmd_gaps"] if v > 0]
            if not vals:
                continue
            clipped = np.clip(vals, bins[0], bins[-1])
            frac = np.histogram(clipped, bins=bins)[0] / len(vals)
            ax.bar(x0, frac, width=x1 - x0, align="edge", color=col, label=label, lw=0)
        ax.set_xscale("log")
        ax.axvline(10, color="gray", ls="--", lw=0.8)
        ax.text(10, 0.92, " 10 s cap", transform=ax.get_xaxis_transform(),
                fontsize=7, color="gray")
        ax.set_xlabel(f"inter-command delay (s) — {ds}")
        # the figure's claim: AI carries more mass in the 5-11s delay-cap band than humans
        band = {g: np.mean([(5 <= v <= 11) for f in feats if f.is_ai == g
                            for v in f.timing["_pool_cmd_gaps"] if v > 0]) for g in (False, True)}
        assert band[True] > band[False], (ds, band)
        print(f"[check] {ds} 5-11s cap-band mass: AI={band[True]:.2f} > human={band[False]:.2f}")
    axes[0].set_ylabel("fraction of command gaps")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cmd_gap_hist.pdf", bbox_inches="tight")
    plt.close(fig)


def _null_energy(all_feats: dict) -> None:
    """Energy-distance null over all balanced relabelings, one panel per dataset.

    The exact randomization test made visual: grey = the statistic under every
    balanced regrouping of the same actors, red line = the true AI/human split.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, (ds, feats) in zip(axes, all_feats.items()):
        names, X, is_ai, _ = feature_matrix(feats)
        agg = run_null(names, X, is_ai)["aggregates"]["energy_distance"]
        n = len(agg["null"])
        ax.hist(agg["null"], bins=24, color=NULL_COLOR, edgecolor="white", lw=0.4)
        ax.axvline(agg["true"], color=AI_COLOR, lw=1.8,
                   label=f"true AI/human split\n$p={agg['k']}/{n}={agg['p']:.3f}$")
        ax.set_xlabel(f"energy distance over relabelings — {ds}")
        ax.legend(loc="upper right", fontsize=7)
        print(f"[check] {ds} energy true={agg['true']:.3f}  p={agg['k']}/{n}={agg['p']:.4f}")
    axes[0].set_ylabel("count (of relabelings)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_null_energy.pdf", bbox_inches="tight")
    plt.close(fig)


# Commands shown in the command-usage figure (formerly the tab:cmd-usage table).
TABLE_COMMANDS = ("grep", "tail", "cat", "head", "find", "ls", "sed", "awk",
                  "less", "vim", "systemctl", "journalctl", "curl", "wget", "php", "mysql")


def _cmd_counts(ds: str) -> dict[str, Counter]:
    """Per-actor EXECVE command counts, so values stay traceable to the dataset."""
    return {actor: Counter(_command_stream(parse_audit_file(get_log_path(actor, "audit", dataset=ds))))
            for actor in analysis_actors(ds)}


def _sorted_commands(usage: dict) -> list[str]:
    """TABLE_COMMANDS ordered by mean count over all actors of both datasets."""
    mean = {c: np.mean([cnt.get(c, 0) for counts in usage.values() for cnt in counts.values()])
            for c in TABLE_COMMANDS}
    return sorted(TABLE_COMMANDS, key=mean.get, reverse=True)


def _cmd_usage_bars(usage: dict) -> None:
    """Paired horizontal bars (mean count per actor, sample-std error bars).

    Replaces the per-actor command-usage table; commands sorted by overall
    mean usage so the heavy hitters read top-down.
    """
    cmds = _sorted_commands(usage)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), sharey=True)
    y = np.arange(len(cmds))
    h = 0.38
    for ax, (ds, counts) in zip(axes, usage.items()):
        hum = [a for a in counts if not is_ai_actor(a)]
        ai = [a for a in counts if is_ai_actor(a)]

        for off, group, col, label in ((-h / 2, hum, HUMAN_COLOR, "Human"),
                                       (h / 2, ai, AI_COLOR, "AI")):
            vals = np.array([[counts[a].get(c, 0) for a in group] for c in cmds], float)
            ax.barh(y + off, vals.mean(axis=1), height=h, xerr=vals.std(axis=1, ddof=1),
                    capsize=2, color=col, label=label, error_kw=dict(lw=0.7))
        ax.set_xlim(left=0)
        ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5)
        ax.set_xlabel(f"mean command count per actor — {ds}")
        if ds == "Nextcloud":  # spot-check against the verified table numbers
            vim_h = np.mean([counts[a].get("vim", 0) for a in hum])
            print(f"[check] Nextcloud vim human mean = {vim_h:.1f} (table: 15.8)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cmds, family="monospace")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cmd_usage.pdf", bbox_inches="tight")
    plt.close(fig)


# Linear-axis clip per dataset; outliers beyond it are annotated, not drawn.
BOX_CLIP = {"Nextcloud": 125, "WordPress": None}


def _cmd_usage_box(usage: dict) -> None:
    """Boxplot variant of the command-usage figure (fig_cmd_usage_box.pdf).

    One box per command and class over per-actor counts, linear x-axis.
    Nextcloud is clipped (single human `sed` actor at ~800 would squash the
    panel); clipped points show as an `-> N` note at the axis edge.
    """
    cmds = _sorted_commands(usage)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), sharey=True)
    y = np.arange(len(cmds))
    h = 0.38
    for ax, (ds, counts) in zip(axes, usage.items()):
        hum = [a for a in counts if not is_ai_actor(a)]
        ai = [a for a in counts if is_ai_actor(a)]
        for off, group, col in ((-h / 2, hum, HUMAN_COLOR), (h / 2, ai, AI_COLOR)):
            data = [[counts[a].get(c, 0) for a in group] for c in cmds]
            ax.boxplot(data, positions=y + off, widths=h * 0.85, orientation="horizontal",
                       manage_ticks=False, patch_artist=True,
                       boxprops=dict(facecolor=col, lw=0.6),
                       whiskerprops=dict(color=col, lw=0.8), capprops=dict(color=col, lw=0.8),
                       medianprops=dict(color="white", lw=1.0),
                       flierprops=dict(marker="o", ms=2, mfc=col, mec="none"))
        lim = BOX_CLIP[ds]
        ax.set_xlim(-1.5, lim)
        if lim:
            for i, c in enumerate(cmds):
                for off, group, col in ((-h / 2, hum, HUMAN_COLOR), (h / 2, ai, AI_COLOR)):
                    out = sorted(v for a in group if (v := counts[a].get(c, 0)) > lim)
                    if out:
                        noun = "outliers" if len(out) > 1 else "outlier"
                        ax.text(lim * 0.99, i + off,
                                f"$\\rightarrow$ {noun} at {', '.join(map(str, out))} ",
                                ha="right", va="center", fontsize=6, color=col)
        ax.grid(axis="x", ls="--", lw=0.4, alpha=0.5)
        ax.set_xlabel(f"command count per actor — {ds}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cmds, family="monospace")
    axes[0].invert_yaxis()
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=HUMAN_COLOR, label="Human"),
                            plt.Rectangle((0, 0), 1, 1, color=AI_COLOR, label="AI")],
                   loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cmd_usage_box.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(0)  # jitter reproducibility
    all_feats = {ds: extract_dataset(ds) for ds in ("Nextcloud", "WordPress")}
    usage = {ds: _cmd_counts(ds) for ds in ("Nextcloud", "WordPress")}
    _cmd_gap_hist(all_feats)
    _cmd_usage_bars(usage)
    _cmd_usage_box(usage)
    _null_energy(all_feats)
    for name in ("fig_cmd_gap_hist", "fig_cmd_usage", "fig_cmd_usage_box", "fig_null_energy"):
        p = OUT / f"{name}.pdf"
        assert p.exists() and p.stat().st_size > 1000, p
    print("[ok] 4 paper figures ->", OUT)


if __name__ == "__main__":
    main()
