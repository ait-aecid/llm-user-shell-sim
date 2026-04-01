# core/benchmark.py

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Optional, Dict, Any, Iterator

import torch


@contextmanager
def bench(
    enabled: bool,
    name: str,
    *,
    meta_fn: Optional[Callable[[], Dict[str, Any]]] = None,
) -> Iterator[None]:
    """
    Lightweight reusable benchmarking context manager.

    Example usage:

        examples = []

        with bench(args.benchmark, "load_examples",
                   meta_fn=lambda: {"n": len(examples)}):
            examples = load_examples(cfg)


    Parameters
    ----------
    enabled:
        If False → does nothing (zero overhead except one if-check)

    name:
        Name printed in output

    meta_fn:
        Optional function returning a dictionary with extra info
        printed after timing.

        Example:
            lambda: {"n": len(examples)}

    Output example:

        [BENCH] load_examples: 3.214s | n=15233
    """

    # Fast exit → almost zero overhead
    if not enabled:
        yield
        return

    # GPU operations are asynchronous → synchronize for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    try:
        yield

    finally:

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.perf_counter() - t0

        meta = ""

        if meta_fn is not None:
            try:
                m = meta_fn() or {}

                if m:
                    meta = " | " + " ".join(
                        f"{k}={v}" for k, v in m.items()
                    )

            except Exception:
                meta = ""

        print(f"  [BENCH] {name}: {dt:.3f}s{meta}")