# experiments/llm_search.py
# Run from project root:
#   python -m experiments.llm_search
#
# Updated to use:
# - core.loader.load_examples() (your real logs)
# - predefined group splits (same pattern as your other experiments)
#
# NOTE:
# If you set cfg.use_llm_fallback=False, you do NOT need OPENAI_API_KEY and will run 100% locally.

from __future__ import annotations

import numpy as np

from core.loader import load_examples, LoadConfig
from core.splits import make_splits
from core.reporting import print_split_stats

# Pipeline API
from llm_pipeline.pipeline import Candidate, RAGLLMConfig, search


def main():
    # ---- Load real dataset via core.loader ----
    # IMPORTANT: LLM/RAG pipelines typically work better on *bundles/windows* of lines
    # than on single lines. Consider window_mode="lines" (or "cids") to reduce N and add context.
    examples = load_examples(
        LoadConfig(
            log_files=("audit.log",),          # adjust if you want more sources
            prefix_with_log_type=False,
            preprocess_mode="soft",            # "soft" keeps some structure; "aggressive" can help too
            window_mode="cids",               # recommended for RAG: fewer, richer examples
            # window_size=10,
            # window_stride=5,
            # join_token=" <EOL> ",
            # If you want Drain3 template-ID windows instead:
            window_size=5,
            window_stride=2,
            cid_prefix="CID",
            max_lines_per_file=400,
        )
    )

    y = np.array([e.label for e in examples], dtype=object)
    groups = np.array([e.group for e in examples], dtype=object)

    # ---- Predefined group split (matches your other experiments) ----
    split = make_splits(
        y,
        groups=groups,
        val_groups=["Armin", "GPT4.1"],
        test_groups=["Benni", "GPT4.1_V2"],
    )

    print_split_stats(examples, split)

    # -----------------------------
    # Candidate grid (keep small; LLM fallback calls can cost money)
    # -----------------------------
    candidates = []

    for bundle_size in [5, 10]:
        for per_class_k in [2, 5]:
            for margin in [0.05, 0.10]:
                for agg in ["mean", "median"]:
                    candidates.append(
                        Candidate(
                            cfg=RAGLLMConfig(
                                # bundling
                                bundle_size=bundle_size,
                                bundle_strategy="fixed",
                                sliding_stride=max(1, bundle_size // 2),
                                drop_last_incomplete=True,

                                # retrieval
                                per_class_k=per_class_k,
                                max_chars_per_retrieved=1000,

                                # local embeddings
                                local_embedding_model="BAAI/bge-base-en-v1.5",
                                local_embedding_batch_size=32,
                                local_embedding_device="cpu",
                                local_normalize_embeddings=True,

                                # hybrid gating
                                use_llm_fallback=True,
                                llm_uncertainty_margin=margin,
                                score_agg=agg,

                                # LLM fallback
                                chat_model="gpt-4.1-mini",
                                temperature=0.0,
                                max_output_tokens=30,

                                seed=42,
                            )
                        )
                    )


    best, best_val, best_test, all_val = search(
        examples,
        split,
        candidates,
        metric="f1_macro",
        evaluate_test_for_all=False,  # recommended
    )

    print("\nBest config:")
    print(best.cfg)
    print(f"Best VAL f1_macro:  {best_val.f1_macro:.4f}")
    print(f"Best TEST f1_macro: {best_test.f1_macro:.4f}")

    print("\n=== Best VAL classification report ===")
    print(best_val.per_class_report)
    print("Confusion matrix (val):")
    print(best_val.confusion)

    # Optional leaderboard
    top = sorted(all_val, key=lambda x: x[1].f1_macro, reverse=True)[:10]
    print("\n=== Top 10 VAL candidates ===")
    for i, (cand, res) in enumerate(top, 1):
        print(f"{i:02d}  f1_macro={res.f1_macro:.4f}  {cand.cfg}")


if __name__ == "__main__":
    main()
