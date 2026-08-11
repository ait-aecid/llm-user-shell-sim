from __future__ import annotations

"""Exact null-permutation analysis: is the true AI/human split more distinct
than random balanced regroupings of the same actors?

Run as a module from the repo root, e.g.::

    python -m src.runners.stats.eda_null_overview --dataset both

For each dataset the per-actor scalar features from
:mod:`src.stats_tools.audit_eda` are extracted once, then every balanced
relabeling of the actors (C(10,4)=210 for Nextcloud, C(8,4)=70 for WordPress,
including the true one) is evaluated in memory. Statistics:

- per feature: Cliff's delta (rank-based; identical permutation p-values to
  Mann-Whitney U / AUC / Vargha-Delaney A12),
- aggregate: energy distance (Szekely-Rizzo E-statistic) on rank-transformed
  features, plus max-|delta| (Westfall-Young maxT, gives FWER-adjusted
  per-feature p-values) and mean-|delta|.

The null is fully enumerated, so all p-values are exact fractions k/N.
Outputs (PNGs + ``null_summary.csv``) go to ``results/eda/<dataset>/null/``.
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.stats_tools.audit_eda import ActorFeatures, extract_dataset

TRUE_COLOR = "#d1495b"
NULL_COLOR = "#8da0ab"

# Whole-file span metrics are contaminated by ExperimentAggregated session
# concatenation (timestamps jump across experiment boundaries).
EXCLUDED_FEATURES = {"span_min", "events_per_min"}

CAVEAT = ("Exact randomization test over all balanced relabelings; small-n "
          "(actors are the units), p-values are exact fractions.")


def _is_count_feature(name: str) -> bool:
    """Volume-confounded features: absolute counts scale with session length."""
    # complexity_mad* are MADs of raw template counts, so they scale with volume too.
    return name.startswith(("n_", "distinct_", "long_pause_count", "complexity_mad"))


# ---- statistics ----

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta of x vs y: P(x>y) - P(x<y), in [-1, 1]."""
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.sum() / diff.size)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) with tie averaging. n <= 10 here, so O(n^2) is fine."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    for v in np.unique(a):
        mask = a == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def energy_distance(D: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Szekely-Rizzo E-statistic from a precomputed pairwise-distance matrix."""
    between = D[np.ix_(a, b)].mean()
    within_a = D[np.ix_(a, a)].mean()
    within_b = D[np.ix_(b, b)].mean()
    return float(2.0 * between - within_a - within_b)


# ---- feature matrix ----

def feature_matrix(feats: list[ActorFeatures]):
    """Flatten scalar metrics to (names, X[actor, feature], is_ai, actors).

    Keeps only numeric, non-None metrics present for every actor; drops the
    contaminated span features and zero-variance columns.
    """
    rows = []
    for f in feats:
        row = {}
        for fam in (f.volume, f.timing, f.commands, f.syscall, f.sequence, f.session, f.mitre, f.complexity):
            for k, v in fam.items():
                if k.startswith("_") or k in EXCLUDED_FEATURES:
                    continue
                if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                    row[k] = float(v)
        rows.append(row)
    names = sorted(set.intersection(*(set(r) for r in rows)))
    X = np.array([[r[k] for k in names] for r in rows])
    keep = X.std(axis=0) > 0
    names = [n for n, k in zip(names, keep) if k]
    X = X[:, keep]
    is_ai = np.array([f.is_ai for f in feats])
    actors = [f.actor for f in feats]
    return names, X, is_ai, actors


# ---- enumeration ----

def run_null(names: list[str], X: np.ndarray, is_ai: np.ndarray) -> dict:
    """Evaluate every balanced relabeling; return null distributions + exact p's."""
    n, n_ai, n_feat = len(is_ai), int(is_ai.sum()), len(names)
    ranks = np.column_stack([_rankdata(X[:, j]) for j in range(n_feat)])
    D = np.sqrt(((ranks[:, None, :] - ranks[None, :, :]) ** 2).sum(axis=-1))

    labelings = list(itertools.combinations(range(n), n_ai))
    true_ai = tuple(sorted(int(i) for i in np.flatnonzero(is_ai)))
    true_i = labelings.index(true_ai)
    N = len(labelings)

    deltas = np.zeros((N, n_feat))
    energy = np.zeros(N)
    for i, comb in enumerate(labelings):
        ai_idx = np.array(comb)
        hum_idx = np.array([k for k in range(n) if k not in comb])
        for j in range(n_feat):
            deltas[i, j] = cliffs_delta(X[ai_idx, j], X[hum_idx, j])
        energy[i] = energy_distance(D, ai_idx, hum_idx)

    abs_d = np.abs(deltas)
    max_abs = abs_d.max(axis=1)
    mean_abs = abs_d.mean(axis=1)
    tol = 1e-12

    def p_ge(null: np.ndarray, true_val: float) -> tuple[int, float]:
        k = int((null >= true_val - tol).sum())
        return k, k / N

    aggregates = {}
    for label, null in (("energy_distance", energy), ("max_abs_delta", max_abs),
                        ("mean_abs_delta", mean_abs)):
        k, p = p_ge(null, null[true_i])
        aggregates[label] = {"null": null, "true": float(null[true_i]), "k": k, "p": p}

    per_feature = []
    for j, name in enumerate(names):
        k, p = p_ge(abs_d[:, j], abs_d[true_i, j])
        k_wy, p_wy = p_ge(max_abs, abs_d[true_i, j])
        per_feature.append({
            "feature": name,
            "true_delta": float(deltas[true_i, j]),
            "k": k, "p": p, "k_wy": k_wy, "p_wy": p_wy,
            "is_count_feature": _is_count_feature(name),
            "null_abs": abs_d[:, j],
        })

    return {"N": N, "n": n, "n_ai": n_ai, "aggregates": aggregates,
            "per_feature": per_feature, "true_i": true_i}


# ---- plots ----

def _footnote(fig, res: dict) -> None:
    note = CAVEAT
    if res["n_ai"] * 2 == res["n"]:
        note += (f" Balanced {res['n_ai']}v{res['n_ai']}: each labeling and its "
                 f"complement give identical |statistics| ({res['N'] // 2} distinct partitions).")
    fig.text(0.01, -0.01, note, fontsize=7, color="gray")


def _plot_aggregate(res: dict, save_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = {"energy_distance": "Energy distance (rank features)",
              "max_abs_delta": "max |Cliff's δ| (Westfall–Young)",
              "mean_abs_delta": "mean |Cliff's δ|"}
    for ax, (key, agg) in zip(axes, res["aggregates"].items()):
        ax.hist(agg["null"], bins=30, color=NULL_COLOR, edgecolor="white")
        ax.axvline(agg["true"], color=TRUE_COLOR, lw=2, label="true AI/human split")
        ax.set_title(f"{titles[key]}\np = {agg['k']}/{res['N']} = {agg['p']:.4f}", fontsize=10)
        ax.set_xlabel("statistic over relabelings")
    axes[0].set_ylabel(f"count (of {res['N']} labelings)")
    axes[0].legend(fontsize=8)
    _footnote(fig, res)
    fig.tight_layout()
    fig.savefig(save_dir / "NULL_aggregate.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_percentiles(res: dict, save_dir: Path, dpi: int) -> None:
    pf = sorted(res["per_feature"], key=lambda r: (r["p"], -abs(r["true_delta"])))
    labels = [r["feature"] + (" *" if r["is_count_feature"] else "") for r in pf]
    y = np.arange(len(pf))[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(pf) + 2))
    ax.barh(y, [r["p"] for r in pf], color=NULL_COLOR, label="per-feature p (two-sided)")
    ax.scatter([r["p_wy"] for r in pf], y, color="black", s=18, zorder=3,
               label="Westfall–Young adjusted p")
    ax.axvline(0.05, color=TRUE_COLOR, ls="--", lw=1, label="0.05")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel(f"exact empirical p (of {res['N']} labelings; floor = 2/{res['N']})")
    ax.set_title("Per-feature |Cliff's δ|: exact p vs FWER-adjusted p (* = count feature)")
    ax.legend(fontsize=8)
    _footnote(fig, res)
    fig.tight_layout()
    fig.savefig(save_dir / "NULL_feature_percentiles.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_hists(res: dict, save_dir: Path, dpi: int) -> None:
    pf = sorted(res["per_feature"], key=lambda r: r["p"])
    ncols = 5
    nrows = -(-len(pf) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows))
    for ax, r in zip(axes.flat, pf):
        ax.hist(r["null_abs"], bins=np.linspace(0, 1, 26), color=NULL_COLOR)
        ax.axvline(abs(r["true_delta"]), color=TRUE_COLOR, lw=1.8)
        ax.set_title(f"{r['feature']}\np={r['k']}/{res['N']}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(pf):]:
        ax.axis("off")
    fig.suptitle("Null distribution of |Cliff's δ| per feature (red = true split)", fontsize=11)
    _footnote(fig, res)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_dir / "NULL_feature_histograms.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_csv(res: dict, save_dir: Path) -> None:
    cols = ["name", "kind", "true_value", "k", "N", "p", "p_wy_adjusted", "is_count_feature"]
    with (save_dir / "null_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for key, agg in res["aggregates"].items():
            w.writerow({"name": key, "kind": "aggregate", "true_value": agg["true"],
                        "k": agg["k"], "N": res["N"], "p": agg["p"],
                        "p_wy_adjusted": "", "is_count_feature": ""})
        for r in res["per_feature"]:
            w.writerow({"name": r["feature"], "kind": "feature",
                        "true_value": r["true_delta"], "k": r["k"], "N": res["N"],
                        "p": r["p"], "p_wy_adjusted": r["p_wy"],
                        "is_count_feature": r["is_count_feature"]})


# ---- driver ----

def run_dataset(dataset: str, save_root: Path, dpi: int) -> None:
    feats = extract_dataset(dataset)
    if not feats:
        print(f"[skip] {dataset}: no parseable actors")
        return
    names, X, is_ai, actors = feature_matrix(feats)
    res = run_null(names, X, is_ai)
    save_dir = save_root / dataset / "null"
    save_dir.mkdir(parents=True, exist_ok=True)
    _plot_aggregate(res, save_dir, dpi)
    _plot_percentiles(res, save_dir, dpi)
    _plot_feature_hists(res, save_dir, dpi)
    _write_csv(res, save_dir)

    print(f"\n=== {dataset}: {res['n']} actors ({res['n_ai']} AI), "
          f"{len(names)} features, {res['N']} labelings ===")
    for key, agg in res["aggregates"].items():
        print(f"  {key:16s} true={agg['true']:.4f}  p = {agg['k']}/{res['N']} = {agg['p']:.4f}")
    survivors = [r for r in res["per_feature"] if r["p_wy"] <= 0.05]
    print(f"  features surviving Westfall-Young FWER at 0.05: {len(survivors)}")
    for r in sorted(res["per_feature"], key=lambda r: r["p_wy"])[:8]:
        print(f"    {r['feature']:32s} delta={r['true_delta']:+.3f}  "
              f"p={r['k']}/{res['N']}  p_WY={r['k_wy']}/{res['N']}")
    print(f"  -> {save_dir}")


# ---- selfcheck ----

def _selfcheck() -> None:
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == -1.0
    assert cliffs_delta(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert cliffs_delta(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert list(_rankdata(np.array([10.0, 20.0, 20.0, 30.0]))) == [1.0, 2.5, 2.5, 4.0]

    pts = np.array([[0.0], [0.0], [1.0], [1.0]])
    D = np.abs(pts - pts.T)
    assert energy_distance(D, np.array([0, 1]), np.array([2, 3])) == 2.0  # 2*1 - 0 - 0
    assert energy_distance(D, np.array([0, 2]), np.array([1, 3])) == 0.0  # identical groups

    # End-to-end: 10 actors (4 AI = last 4), one perfectly separating feature.
    # Only 2 of 210 labelings (top-4 or bottom-4) reach |delta| = 1.
    X = np.array([[float(i), v] for i, v in
                  enumerate([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])])
    is_ai = np.array([False] * 6 + [True] * 4)
    res = run_null(["sep", "noise"], X, is_ai)
    sep = next(r for r in res["per_feature"] if r["feature"] == "sep")
    assert res["N"] == 210
    assert sep["true_delta"] == 1.0 and sep["k"] == 2 and abs(sep["p"] - 2 / 210) < 1e-12
    assert res["aggregates"]["max_abs_delta"]["true"] == 1.0
    assert res["aggregates"]["energy_distance"]["p"] <= 0.05
    print("selfcheck OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact null-permutation analysis of AI/human distinctness")
    parser.add_argument("--dataset", choices=["Nextcloud", "WordPress", "both"], default="both")
    parser.add_argument("--save_dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--selfcheck", action="store_true",
                        help="run assert-based checks on synthetic data and exit")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    datasets = ["Nextcloud", "WordPress"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_dataset(ds, args.save_dir, args.dpi)


if __name__ == "__main__":
    main()
