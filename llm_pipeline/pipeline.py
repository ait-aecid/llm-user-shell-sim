# llm_pipeline/pipeline.py
#
# Pipeline 4: Bundle + Embedding-Retrieval (RAG) + Hybrid (Embeddings-fastpath + LLM fallback)
#
# SPEED UPDATES:
# - Batch embeddings during prediction (GPU-friendly)
# - Avoid per-query renormalization for cosine if embeddings are normalized once
# - Optional ANN retrieval backend (FAISS) to avoid full scan
#
# Dependencies:
#   pip install numpy openai sentence-transformers torch tqdm
# Optional for ANN:
#   pip install faiss-cpu   (or faiss-gpu)

from __future__ import annotations

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple, Any, Set

import json

from transformers.utils import logging
logging.set_verbosity_error()
import time

import numpy as np
from tqdm.auto import tqdm

from core.data import Example
from core.env import load_project_env
from core.splits import Split
from core.eval import EvalResult, evaluate_classifier

from sentence_transformers import SentenceTransformer
from openai import OpenAI

load_project_env()

# Optional FAISS
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_OK = False


# -----------------------------
# Bundling
# -----------------------------
def _normalize_bundle(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _bundle_texts(
    texts: List[str],
    *,
    bundle_size: int,
    stride: int,
    strategy: str,
    drop_last: bool,
) -> List[str]:
    if bundle_size <= 0:
        raise ValueError("bundle_size must be > 0")

    n = len(texts)
    if n == 0:
        return []

    bundles: List[str] = []

    if strategy == "fixed":
        for start in range(0, n, bundle_size):
            end = start + bundle_size
            if end > n and drop_last:
                break
            chunk = texts[start:min(end, n)]
            b = _normalize_bundle("\n".join(chunk))
            if b:
                bundles.append(b)

    elif strategy == "sliding":
        stride = max(1, stride)
        if bundle_size > n:
            if drop_last:
                return []
            b = _normalize_bundle("\n".join(texts))
            return [b] if b else []

        for start in range(0, n - bundle_size + 1, stride):
            end = start + bundle_size
            chunk = texts[start:end]
            b = _normalize_bundle("\n".join(chunk))
            if b:
                bundles.append(b)

    else:
        raise ValueError(f"Unknown strategy='{strategy}', expected 'fixed' or 'sliding'.")

    return bundles


# -----------------------------
# Similarity helpers
# -----------------------------
def _dot_sims(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    query: (D,), mat: (N, D) -> sims: (N,)
    Assumes BOTH query and mat rows are already L2-normalized.
    Then cosine == dot product.
    """
    q = query.astype(np.float32, copy=False)
    m = mat.astype(np.float32, copy=False)
    return (m @ q).astype(np.float32)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / (np.linalg.norm(x) + 1e-12)


# -----------------------------
# In-memory index (NumPy or FAISS)
# -----------------------------
class _InMemoryPerClassIndex:
    """
    Stores embeddings + bundle text per label.
    Retrieval is done per class to guarantee class-balanced examples.

    Backends:
      - numpy: full scan (dot product) + argpartition topk
      - faiss: ANN / exact IP search (depending on index used)
    """

    def __init__(self, *, backend: str = "numpy", faiss_hnsw_m: int = 32):
        self._emb: Dict[str, np.ndarray] = {}
        self._txt: Dict[str, List[str]] = {}

        self._backend = backend
        self._faiss_hnsw_m = int(faiss_hnsw_m)

        # per-class faiss indices (if used)
        self._faiss_index: Dict[str, Any] = {}

    @property
    def backend(self) -> str:
        return self._backend

    def add(self, label: str, emb: np.ndarray, txt: List[str]) -> None:
        if len(txt) != emb.shape[0]:
            raise ValueError("emb and txt must align")

        # Keep float32 and ensure normalized rows (important for fast dot-cosine)
        emb = emb.astype(np.float32, copy=False)
        emb = _l2_normalize_rows(emb)

        self._emb[label] = emb
        self._txt[label] = list(txt)

        # Build FAISS index if requested + available
        if self._backend == "faiss":
            if not _FAISS_OK:
                # silently fall back to numpy if faiss isn't installed
                self._backend = "numpy"
                return

            d = emb.shape[1]
            # HNSW IP index: fast ANN for inner product (cosine if normalized)
            # If you want exact, you can swap to IndexFlatIP.
            index = faiss.IndexHNSWFlat(d, self._faiss_hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.add(emb)
            self._faiss_index[label] = index

    def topk(self, label: str, query_emb: np.ndarray, k: int) -> List[Tuple[float, str]]:
        if k <= 0:
            return []
        if label not in self._emb or self._emb[label].shape[0] == 0:
            return []

        q = _l2_normalize_vec(query_emb.astype(np.float32, copy=False))

        if self._backend == "faiss" and _FAISS_OK and label in self._faiss_index:
            index = self._faiss_index[label]
            k_eff = min(k, self._emb[label].shape[0])
            # FAISS expects (nq, d)
            D, I = index.search(q.reshape(1, -1), k_eff)
            sims = D[0]
            idxs = I[0]
            out: List[Tuple[float, str]] = []
            for sim, i in zip(sims, idxs):
                if i < 0:
                    continue
                out.append((float(sim), self._txt[label][int(i)]))
            return out

        # numpy fallback
        sims = _dot_sims(q, self._emb[label])
        k_eff = min(k, sims.shape[0])
        idx = np.argpartition(-sims, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-sims[idx])]
        return [(float(sims[i]), self._txt[label][int(i)]) for i in idx]


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class RAGLLMConfig:
    # bundling hyperparams
    bundle_size: int = 50
    bundle_strategy: str = "fixed"   # fixed | sliding
    sliding_stride: int = 25         # only used if sliding
    drop_last_incomplete: bool = True

    # retrieval hyperparams
    per_class_k: int = 5
    max_chars_per_retrieved: int = 1400  # prompt budget control

    # retrieval backend
    retrieval_backend: str = "numpy"     # "numpy" or "faiss"
    faiss_hnsw_m: int = 32              # only used if retrieval_backend="faiss"

    # OPEN-SOURCE embedding (local, free)
    local_embedding_model: str = "BAAI/bge-base-en-v1.5"
    local_embedding_batch_size: int = 32
    local_embedding_device: str = "cuda"  # "cuda" or "cpu"
    local_normalize_embeddings: bool = True

    # Prediction embedding batching
    predict_embedding_batch_size: int = 64  # NEW: batch size for embedding VAL/TEST

    # OpenAI chat classification (fallback)
    chat_model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_output_tokens: int = 30
    timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 1.5

    # Hybrid gating to reduce LLM calls
    use_llm_fallback: bool = True
    llm_uncertainty_margin: float = 0.08  # call LLM if |score| < margin
    score_agg: str = "mean"               # "mean" or "median"

    # misc
    seed: int = 42


# -----------------------------
# OpenAI helpers (fallback only)
# -----------------------------
def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…(truncated)…"


def _build_messages(
    query_bundle: str,
    *,
    label_a: str,
    label_b: str,
    retrieved_a: List[Tuple[float, str]],
    retrieved_b: List[Tuple[float, str]],
    max_chars_per_retrieved: int,
) -> Tuple[str, str]:
    sys = (
        "You are a strict binary classifier for log bundles.\n"
        "Return ONLY valid JSON with exactly one key: \"label\".\n"
        f"Valid labels are: \"{label_a}\" and \"{label_b}\".\n"
        "No extra keys. No commentary. No markdown."
    )

    def fmt_block(name: str, items: List[Tuple[float, str]]) -> str:
        lines = [f"Class {name} retrieved examples:"]
        for i, (sim, txt) in enumerate(items, 1):
            lines.append(f"[{name} ex {i}] sim={sim:.4f}\n{_truncate(txt, max_chars_per_retrieved)}\n")
        return "\n".join(lines)

    user = (
        f"Task: classify the QUERY bundle as either {label_a} or {label_b}.\n"
        "Use the retrieved labeled examples as reference.\n\n"
        f"{fmt_block(label_a, retrieved_a)}\n\n"
        f"{fmt_block(label_b, retrieved_b)}\n\n"
        "QUERY bundle:\n"
        f"{_truncate(query_bundle, 9000)}\n\n"
        f"Output JSON only, like: {{\"label\":\"{label_a}\"}}"
    )
    return sys, user


def _parse_label(raw: str, *, valid_labels: Set[str]) -> str:
    raw = (raw or "").strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict) or set(obj.keys()) != {"label"}:
        raise ValueError(f"Expected JSON with exactly {{'label'}}, got: {raw[:200]}")
    lab = obj["label"]
    if lab not in valid_labels:
        raise ValueError(f"Invalid label '{lab}', expected one of {sorted(valid_labels)}")
    return str(lab)


def _chat_classify(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
    system_msg: str,
    user_msg: str,
    valid_labels: Set[str],
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                timeout=timeout_s,
            )
            content = resp.choices[0].message.content
            return _parse_label(content, valid_labels=valid_labels)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_backoff_s * (attempt + 1))
                continue
            raise last_err


def _aggregate_sims(sims: np.ndarray, *, agg: str) -> float:
    if sims.size == 0:
        return float("nan")
    if agg == "mean":
        return float(np.mean(sims))
    if agg == "median":
        return float(np.median(sims))
    raise ValueError(f"Unknown score_agg='{agg}', expected 'mean' or 'median'.")


# -----------------------------
# Local embedding helpers (SentenceTransformers)
# -----------------------------
def _load_local_embedder(cfg: RAGLLMConfig) -> SentenceTransformer:
    return SentenceTransformer(cfg.local_embedding_model, device=cfg.local_embedding_device)


def _embed_texts_local(
    embedder: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int,
    normalize: bool,
    desc: str = "embed",
    verbose: bool = True,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    bs = max(1, int(batch_size))
    out_chunks: List[np.ndarray] = []

    rng = range(0, len(texts), bs)
    it = rng if not verbose else tqdm(rng, desc=desc, leave=False)
    for i in it:
        chunk = texts[i:i + bs]
        emb = embedder.encode(
            chunk,
            batch_size=len(chunk),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        out_chunks.append(emb.astype(np.float32, copy=False))

    out = np.vstack(out_chunks) if out_chunks else np.zeros((0, 1), dtype=np.float32)

    # Ensure normalized if requested
    if normalize:
        out = _l2_normalize_rows(out)
    return out


# -----------------------------
# Primitive: run ONE config
# -----------------------------
def run_one(
    examples: List[Example],
    split: Split,
    cfg: RAGLLMConfig,
    evaluate_test: bool = True,
    *,
    verbose: bool = True,
) -> Dict[str, EvalResult]:
    if verbose:
        print("\n" + "=" * 80)
        print("[LLM] RUN_ONE START")
        print(f"[LLM] Config: {cfg}")
        print(f"[LLM] evaluate_test: {evaluate_test}")
        print("=" * 80)

    llm_client: Optional[OpenAI] = None
    if cfg.use_llm_fallback:
        llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if verbose:
        print("\n[LLM] Loading local embedding model...")
    embedder = _load_local_embedder(cfg)
    if verbose:
        print(f"[LLM] Local embedder: {cfg.local_embedding_model} on {cfg.local_embedding_device}")
        print(f"[LLM] Retrieval backend: {cfg.retrieval_backend} (faiss_available={_FAISS_OK})")

    X_all = np.array([ex.text for ex in examples], dtype=object)
    y_all = np.array([ex.label for ex in examples], dtype=object)
    labels_sorted = sorted(set(map(str, y_all.tolist())))

    if verbose:
        print(f"[LLM] Total examples: {len(X_all)}")
        print(f"[LLM] Labels present: {labels_sorted}")

    if len(labels_sorted) != 2:
        raise ValueError(f"Expected exactly 2 labels, got {labels_sorted}")

    label_a, label_b = labels_sorted[0], labels_sorted[1]
    valid_labels = {label_a, label_b}

    X_train = X_all[split.train_idx].tolist()
    y_train = [str(x) for x in y_all[split.train_idx].tolist()]

    X_val = X_all[split.val_idx].tolist()
    y_val = [str(x) for x in y_all[split.val_idx].tolist()]

    X_test = X_all[split.test_idx].tolist()
    y_test = [str(x) for x in y_all[split.test_idx].tolist()]

    if verbose:
        print("\n[LLM] --- Split sizes ---")
        print(f"[LLM] TRAIN: {len(X_train)}")
        print(f"[LLM] VAL  : {len(X_val)}")
        print(f"[LLM] TEST : {len(X_test)}")

    # ----- build TRAIN bundles per class -----
    train_lines_by_class: Dict[str, List[str]] = {label_a: [], label_b: []}
    for txt, lab in zip(X_train, y_train):
        train_lines_by_class[lab].append(str(txt))

    train_bundles_by_class: Dict[str, List[str]] = {}
    for lab in (label_a, label_b):
        bs = _bundle_texts(
            train_lines_by_class[lab],
            bundle_size=cfg.bundle_size,
            stride=cfg.sliding_stride,
            strategy=cfg.bundle_strategy,
            drop_last=cfg.drop_last_incomplete,
        )
        train_bundles_by_class[lab] = bs
        if verbose:
            print(f"[LLM] {lab}: {len(bs)} TRAIN bundles")

    # ----- embed TRAIN bundles (local) -----
    if verbose:
        print("\n[LLM] Embedding TRAIN bundles (local, free)...")

    index = _InMemoryPerClassIndex(backend=cfg.retrieval_backend, faiss_hnsw_m=cfg.faiss_hnsw_m)

    for lab in (label_a, label_b):
        if verbose:
            print(f"[LLM]   Embedding {lab}: {len(train_bundles_by_class[lab])} bundles")

        emb = _embed_texts_local(
            embedder,
            train_bundles_by_class[lab],
            batch_size=cfg.local_embedding_batch_size,
            normalize=cfg.local_normalize_embeddings,
            desc=f"[LLM] embed TRAIN {lab}",
            verbose=verbose,
        )
        index.add(lab, emb, train_bundles_by_class[lab])

    # ----- helper: predict list of bundles (batched embeddings + retrieval + optional LLM) -----
    def predict_bundles(name: str, bundles: List[str]) -> List[str]:
        if verbose:
            print(f"\n[LLM] Predicting {name}: {len(bundles)} bundles")
        preds: List[str] = []
        
        llm_available = bool(cfg.use_llm_fallback)
        
        if len(bundles) == 0:
            if verbose:
                print(f"[LLM] ⚠ WARNING: No bundles for {name}!")
            return preds

        # Normalize once (not per item)
        bundles_norm = [_normalize_bundle(b) for b in bundles]

        # Batch embed all queries once (GPU-friendly)
        q_embs = _embed_texts_local(
            embedder,
            bundles_norm,
            batch_size=cfg.predict_embedding_batch_size,
            normalize=cfg.local_normalize_embeddings,
            desc=f"[LLM] embed QUERIES {name}",
            verbose=verbose,
        )

        llm_calls = 0
        fast_calls = 0

        it = range(len(bundles_norm))
        it = it if not verbose else tqdm(it, desc=f"[LLM] predict {name}", leave=False)

        for i in it:
            b_norm = bundles_norm[i]
            q_emb = q_embs[i]  # already normalized if cfg.local_normalize_embeddings

            r_a = index.topk(label_a, q_emb, cfg.per_class_k)
            r_b = index.topk(label_b, q_emb, cfg.per_class_k)

            sims_a = np.array([sim for sim, _ in r_a], dtype=np.float32)
            sims_b = np.array([sim for sim, _ in r_b], dtype=np.float32)

            force_llm = (sims_a.size == 0 or sims_b.size == 0)

            if force_llm:
                score = 0.0
            else:
                a_agg = _aggregate_sims(sims_a, agg=cfg.score_agg)
                b_agg = _aggregate_sims(sims_b, agg=cfg.score_agg)
                score = float(a_agg - b_agg)

            use_llm = llm_available and (force_llm or abs(score) < cfg.llm_uncertainty_margin)
            
            '''
            if verbose:
                print(
                    f"[LLM-GATE] "
                    f"a_mean={a_agg if sims_a.size>0 else 'EMPTY'} "
                    f"b_mean={b_agg if sims_b.size>0 else 'EMPTY'} "
                    f"score={score:.6f} "
                    f"force_llm={force_llm}"
                )
            '''

            if use_llm:
                if llm_client is None:
                    raise RuntimeError("use_llm_fallback=True but OpenAI client is not available.")
                llm_calls += 1

                sys, user = _build_messages(
                    b_norm,
                    label_a=label_a,
                    label_b=label_b,
                    retrieved_a=r_a,
                    retrieved_b=r_b,
                    max_chars_per_retrieved=cfg.max_chars_per_retrieved,
                )

                try:
                    pred = _chat_classify(
                        llm_client,
                        model=cfg.chat_model,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_output_tokens,
                        timeout_s=cfg.timeout_s,
                        max_retries=cfg.max_retries,
                        retry_backoff_s=cfg.retry_backoff_s,
                        system_msg=sys,
                        user_msg=user,
                        valid_labels=valid_labels,
                    )
                    preds.append(pred)

                except Exception as e:
                    msg = str(e)

                    # quota exhaustion / rate limit / billing issues
                    if ("insufficient_quota" in msg) or ("Error code: 429" in msg) or ("429" in msg):
                        llm_available = False  # <-- THIS IS THE KEY LINE

                        if verbose:
                            print("[LLM] ⚠ OpenAI unavailable → embedding-only for rest of this run")

                        # embedding-only fallback for THIS item
                        fast_calls += 1
                        pred = label_a if score >= 0 else label_b
                        preds.append(pred)

                    else:
                        raise

            else:
                fast_calls += 1
                pred = label_a if score >= 0 else label_b
                preds.append(pred)

            if verbose and hasattr(it, "set_postfix"):
                total = llm_calls + fast_calls
                it.set_postfix(
                    fast=fast_calls,
                    llm=llm_calls,
                    llm_rate=f"{(llm_calls / max(1, total)):.0%}",
                    backend=index.backend,
                )

        if verbose:
            total = llm_calls + fast_calls
            if total > 0:
                print(f"[LLM] [{name}] Summary: fast={fast_calls}, llm={llm_calls}, llm_rate={llm_calls/total:.1%}")

        return preds

    # ----- bundle split helper (by true class) -----
    def bundle_split(name: str, texts: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        lines_by_class: Dict[str, List[str]] = {label_a: [], label_b: []}
        for t, lab in zip(texts, labels):
            if lab not in lines_by_class:
                raise ValueError(f"Unexpected label '{lab}' in {name}; expected {sorted(valid_labels)}")
            lines_by_class[lab].append(str(t))

        bundles: List[str] = []
        bundle_labels: List[str] = []

        for lab in (label_a, label_b):
            bs = _bundle_texts(
                lines_by_class[lab],
                bundle_size=cfg.bundle_size,
                stride=cfg.sliding_stride,
                strategy=cfg.bundle_strategy,
                drop_last=cfg.drop_last_incomplete,
            )
            if verbose:
                print(f"[LLM] {lab}: {len(bs)} {name} bundles")
            bundles.extend(bs)
            bundle_labels.extend([lab] * len(bs))

        return bundles, bundle_labels

    # ----- VAL -----
    X_val_bundles, y_val_bundles = bundle_split("VAL", X_val, y_val)

    if len(y_val_bundles) == 0:
        raise ValueError(
            "No VAL bundles were produced. Likely causes:\n"
            f"- bundle_size={cfg.bundle_size} is too large for VAL per-class line counts\n"
            f"- drop_last_incomplete={cfg.drop_last_incomplete} discards partial bundles\n"
            "Fix: lower bundle_size, or set drop_last_incomplete=False, or add more data."
        )

    y_val_pred_list = predict_bundles("VAL", X_val_bundles)

    y_val_true = np.array(y_val_bundles, dtype=object)
    y_val_pred = np.array(y_val_pred_list, dtype=object)

    out: Dict[str, EvalResult] = {
        "val": evaluate_classifier(y_val_true, y_val_pred, labels=labels_sorted)
    }

    # ----- TEST -----
    if evaluate_test:
        X_test_bundles, y_test_bundles = bundle_split("TEST", X_test, y_test)

        if len(y_test_bundles) == 0:
            raise ValueError("No TEST bundles were produced. Fix: lower bundle_size or set drop_last_incomplete=False.")

        y_test_pred_list = predict_bundles("TEST", X_test_bundles)

        y_test_true = np.array(y_test_bundles, dtype=object)
        y_test_pred = np.array(y_test_pred_list, dtype=object)

        out["test"] = evaluate_classifier(y_test_true, y_test_pred, labels=labels_sorted)

    return out


# -----------------------------
# Hyperparameter search (fixed split)
# -----------------------------
@dataclass(frozen=True)
class Candidate:
    cfg: RAGLLMConfig


def search(
    examples: List[Example],
    split: Split,
    candidates: Iterable[Candidate],
    *,
    metric: str = "f1_macro",
    evaluate_test_for_all: bool = False,
    verbose: bool = True,
) -> Tuple[Candidate, EvalResult, EvalResult, List[Tuple[Candidate, EvalResult]]]:
    if metric not in {"f1_macro", "f1_weighted", "accuracy"}:
        raise ValueError("metric must be one of: f1_macro, f1_weighted, accuracy")

    def score(res: EvalResult) -> float:
        return getattr(res, metric)

    candidates = list(candidates)
    if verbose:
        print(f"\n[LLM] Starting search over {len(candidates)} candidates...\n")

    best: Optional[Candidate] = None
    best_val: Optional[EvalResult] = None
    best_test: Optional[EvalResult] = None
    all_val: List[Tuple[Candidate, EvalResult]] = []

    pbar = tqdm(candidates, desc="[LLM] candidates", disable=not verbose)

    for cand in pbar:
        pbar.set_postfix(
            bs=cand.cfg.bundle_size,
            k=cand.cfg.per_class_k,
            fb="on" if cand.cfg.use_llm_fallback else "off",
            margin=f"{cand.cfg.llm_uncertainty_margin:.2f}",
            agg=cand.cfg.score_agg,
            backend=cand.cfg.retrieval_backend,
        )

        out = run_one(
            examples,
            split,
            cand.cfg,
            evaluate_test=evaluate_test_for_all,
            verbose=verbose,
        )
        val_res = out["val"]
        all_val.append((cand, val_res))

        if verbose:
            print(f"[LLM] VAL {metric}: {score(val_res):.4f}")

        if best_val is None or score(val_res) > score(best_val):
            best = cand
            best_val = val_res
            best_test = out.get("test") if evaluate_test_for_all else None
            if verbose:
                print("[LLM] -> New BEST candidate")

    assert best is not None and best_val is not None

    if best_test is None:
        if verbose:
            print("\n[LLM] Evaluating BEST candidate on TEST...\n")
        out_best = run_one(examples, split, best.cfg, evaluate_test=True, verbose=verbose)
        best_test = out_best["test"]

    return best, best_val, best_test, all_val