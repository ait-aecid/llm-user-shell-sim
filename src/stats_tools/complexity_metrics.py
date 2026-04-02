from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from textwrap import shorten
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon

from src.core.shared.loader import (
    LoadConfig,
    _assign_templates_and_cids_global,
    _create_template_miner,
    _get_drain_ini,
    _infer_log_type,
    _preprocess_line,
    _read_lines,
)
from src.core.stats.data_catalog import analysis_actors, get_log_path



METRIC_KEYS = (
    #"gini",
    #"kurtosis",
    #"entropy",
    #"mad",
    #"l1",
    #"js",
    "gini_seq",
    "kurtosis_seq",
    "entropy_seq",
    "mad_seq",
    "l1_seq",
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




def run(config: dict) -> dict:
    dataset = config["dataset"]
    log_type = config["log_type"]
    window_size = config["window_size"]
    stride = config["stride"]
    preprocess_mode = config.get("preprocess_mode", "soft")
    drain_ini_path = config.get("drain_ini_path")

    file_pairs = [
        (actor, str(get_log_path(actor, log_type, dataset=dataset)))
        for actor in analysis_actors(dataset)
    ]

    labels = [label for label, _ in file_pairs]

    pairwise_results = score_file_pairs(
        file_pairs,
        window_size=window_size,
        stride=stride,
        preprocess_mode=preprocess_mode,
        drain_ini_path=drain_ini_path,
    )

    return {
        "labels": labels,
        "pairwise_results": pairwise_results,
    }


def build_sweep_configs(
    *,
    dataset: str,
    log_types: list[str],
    window_sizes: list[int],
    strides: list[int],
    preprocess_mode: str = "soft",
    drain_ini_path: str | None = None,
) -> list[dict]:
    configs: list[dict] = []

    for log_type in log_types:
        for window_size in window_sizes:
            for stride in strides:
                configs.append(
                    {
                        "dataset": dataset,
                        "log_type": log_type,
                        "window_size": window_size,
                        "stride": stride,
                        "preprocess_mode": preprocess_mode,
                        "drain_ini_path": drain_ini_path,
                    }
                )

    return configs

def get_invalid_labels_for_metric(
    pairwise_results: list[dict[str, Any]],
    metric_name: str,
) -> list[str]:
    invalid_pairs = [
        item for item in pairwise_results
        if not np.isfinite(item["distances"][metric_name])
    ]

    if not invalid_pairs:
        return []

    return sorted({
        item["label_1"] for item in invalid_pairs
    } | {
        item["label_2"] for item in invalid_pairs
    })





