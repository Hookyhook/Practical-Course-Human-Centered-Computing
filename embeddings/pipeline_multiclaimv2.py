"""
Pipeline 2 — MulticlaimV2  (Social Media Post → Fact-Check Retrieval)
======================================================================
Corpus:    data/raw/fact_checks.csv
Baseline:  data/processed/multiclaimv2_preprocessed.csv  (unperturbed posts)
Perturbed: data/processed/multiclaimv2_verified.csv      (verified perturbations)
Qrels:     data/raw/fact_check_post_mapping.csv

Two-phase evaluation (H200 / CUDA only):
  Phase 1 — embed full preprocessed baseline posts, compute MAP@k + MAP@k_rr
  Phase 2 — embed verified perturbed posts, merge with baseline metrics

Usage (from repo root):
  python embeddings/pipeline_multiclaimv2.py --models me5,bge-m3,labse
  bash embeddings/run_gpu_pipelines.sh
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (
    MODEL_REGISTRY, RERANKER_REGISTRY, DEFAULT_RERANKER, DEFAULT_K_VALUES,
    PerfConfig, require_cuda, load_bi_encoder, load_or_embed, build_faiss_index,
    faiss_search, rerank_candidates, build_result_rows, build_baseline_lookup,
    print_delta_table, filter_verified_queries, add_performance_args, add_reranker_args,
    perf_config_from_args, reranker_key_from_args, release_gpu_memory,
)

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
CACHE_DIR  = Path(".embedding_cache")
OUTPUT_DIR = Path("results/multiclaimv2")

BASELINE_FILE  = DATA_DIR / "processed" / "multiclaimv2_preprocessed.csv"
PERTURBED_FILE = DATA_DIR / "processed" / "multiclaimv2_verified.csv"
FC_FILE        = DATA_DIR / "raw" / "fact_checks.csv"
MAPPING_FILE   = DATA_DIR / "raw" / "fact_check_post_mapping.csv"


def build_corpus(fc: pd.DataFrame) -> tuple[list[str], dict, dict]:
    fc = fc.drop_duplicates(subset="fact_check_id", keep="first").sort_values("fact_check_id")
    corpus_texts = fc["claim"].tolist()
    fc_ids = fc["fact_check_id"].tolist()
    fc_id_to_idx = {fc_id: i for i, fc_id in enumerate(fc_ids)}
    idx_to_fc_id = {i: fc_id for i, fc_id in enumerate(fc_ids)}
    return corpus_texts, fc_id_to_idx, idx_to_fc_id


def _build_post_qrels(
    queries_df: pd.DataFrame,
    post_to_fc_idxs: dict[str, set],
) -> tuple[list, list]:
    gold_corpus_idxs = []
    query_cluster_sets = []
    for _, row in queries_df.iterrows():
        relevant_idxs = post_to_fc_idxs.get(row["post_id"], set())
        if not relevant_idxs:
            gold_corpus_idxs.append(None)
            query_cluster_sets.append(set())
        else:
            gold_corpus_idxs.append(next(iter(relevant_idxs)))
            query_cluster_sets.append(relevant_idxs)
    return gold_corpus_idxs, query_cluster_sets


def run(
    model_keys: list | None = None,
    top_k: int = 50,
    k_values: tuple = DEFAULT_K_VALUES,
    only_verified: bool = True,
    reranker_key: str | None = DEFAULT_RERANKER,
    perf: PerfConfig | None = None,
):
    require_cuda()
    perf = perf or PerfConfig()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_keys = model_keys or list(MODEL_REGISTRY.keys())

    reranker_spec = RERANKER_REGISTRY.get(reranker_key) if reranker_key else None

    print("\n" + "=" * 65)
    print("PIPELINE: MulticlaimV2 — Post → Fact-Check Retrieval (two-phase)")
    print("Corpus:   fact_checks.csv")
    print(f"Baseline: {BASELINE_FILE.name}")
    print(f"Perturbed:{PERTURBED_FILE.name}")
    print("Qrels:    fact_check_post_mapping.csv")
    print("Metric:   MAP@k, multi-relevant (qrel-based)")
    print(f"Models:   {model_keys}")
    print(f"Reranker: {reranker_key or 'disabled'}")
    if reranker_spec:
        print(f"          {reranker_spec.hf_id}  "
              f"batch={reranker_spec.batch_size or perf.reranker_batch_size}")
    print(f"Encode batch: {perf.encode_batch_size}  |  bf16  |  faiss-gpu")
    print("=" * 65)

    print("\nLoading qrels and fact-checks...")
    mapping = pd.read_csv(MAPPING_FILE, dtype=str)
    fc = pd.read_csv(FC_FILE, dtype=str).fillna("")
    corpus_texts, fc_id_to_idx, _ = build_corpus(fc)

    post_to_fc_idxs: dict[str, set] = defaultdict(set)
    for _, row in mapping.iterrows():
        corpus_idx = fc_id_to_idx.get(row["fact_check_id"])
        if corpus_idx is not None:
            post_to_fc_idxs[row["post_id"]].add(corpus_idx)

    print(f"  Corpus: {len(corpus_texts):,} fact-checks")

    print("\nLoading baseline queries...")
    baseline_df = pd.read_csv(BASELINE_FILE, dtype=str).fillna("")
    baseline_df.reset_index(drop=True, inplace=True)
    baseline_gold, baseline_clusters = _build_post_qrels(baseline_df, post_to_fc_idxs)
    baseline_texts = baseline_df["post_body"].tolist()
    n_no_qrel = sum(1 for g in baseline_gold if g is None)
    if n_no_qrel:
        print(f"  WARNING: {n_no_qrel} baseline queries have no qrel entry")
    print(f"  Baseline: {len(baseline_df):,} queries")

    print("\nLoading perturbed queries...")
    perturbed_df = pd.read_csv(PERTURBED_FILE, dtype=str).fillna("")
    n_before = len(perturbed_df)
    perturbed_df = filter_verified_queries(perturbed_df, only_verified=only_verified)
    perturbed_df.reset_index(drop=True, inplace=True)
    if only_verified:
        print(f"  Verified only: {len(perturbed_df):,} / {n_before:,}")
    pert_gold, pert_clusters = _build_post_qrels(perturbed_df, post_to_fc_idxs)
    pert_texts = perturbed_df["perturbed_text"].tolist()
    print(f"  Perturbed: {len(perturbed_df):,}  |  "
          f"{perturbed_df['perturbation_name'].nunique()} perturbation types  |  "
          f"{perturbed_df['pipeline_lang'].nunique()} languages")

    all_rows = []

    for model_key in model_keys:
        spec = MODEL_REGISTRY[model_key]
        model_batch = spec.batch_size or perf.encode_batch_size
        print(f"\n{'─'*55}")
        print(f"Model: {model_key}  ({spec.hf_id})  batch={model_batch}")
        model = load_bi_encoder(spec, perf)

        corpus_vecs = load_or_embed(
            CACHE_DIR / f"mcv2_{model_key}_factchecks_corpus.npy",
            corpus_texts, model, prefix=spec.passage_prefix, batch_size=model_batch)
        # Corpus embedding is the biggest single encode call (often long
        # fact-check text) — hand any fragmented/cached blocks back before
        # building the FAISS index, rather than letting them accumulate
        # silently until a later phase OOMs.
        release_gpu_memory(label=f"post-corpus-embed [{model_key}]")

        index = build_faiss_index(corpus_vecs, use_gpu=perf.faiss_gpu)

        # ── Phase 1: baseline ──────────────────────────────────────────────────
        print("  Phase 1 — baseline (unperturbed)...")
        baseline_vecs = load_or_embed(
            CACHE_DIR / f"mcv2_{model_key}_baseline_queries.npy",
            baseline_texts, model, prefix=spec.query_prefix, batch_size=model_batch)

        baseline_retrieved = faiss_search(index, baseline_vecs, top_k)

        # Model + FAISS index stay resident (both needed again in Phase 2), but
        # this still returns any fragmented/cached CUDA blocks from Phase 1's
        # encode + search calls back to the allocator before Phase 2 — which is
        # a larger, more variable-length encode call (perturbed text) and was
        # where the OOM was actually happening.
        release_gpu_memory(label=f"post-phase1 [{model_key}]")

        # ── Phase 2: perturbed (embed + search before reranking) ───────────────
        print("  Phase 2 — perturbed (verified)...")
        pert_vecs = load_or_embed(
            CACHE_DIR / f"mcv2_{model_key}_pert_queries.npy",
            pert_texts, model, prefix=spec.query_prefix, batch_size=model_batch)

        pert_retrieved = faiss_search(index, pert_vecs, top_k)

        release_gpu_memory(model, index, label=f"bi-encoder + FAISS [{model_key}]")

        baseline_rr = None
        if reranker_spec:
            baseline_rr = rerank_candidates(
                baseline_texts, corpus_texts, baseline_retrieved,
                reranker_spec=reranker_spec,
                batch_size=reranker_spec.batch_size or perf.reranker_batch_size,
                cache_path=CACHE_DIR / f"mcv2_{model_key}_{reranker_key}_baseline_rr.npy",
                desc=f"Reranking baseline [{model_key}]",
                perf=perf,
            )

        # Keep lookup for fallback; paired mode takes priority in build_result_rows
        baseline_lookup = build_baseline_lookup(
            baseline_df, baseline_gold, baseline_retrieved, baseline_clusters,
            k_values=k_values, reranked=baseline_rr,
        )

        # Paired baseline: post_id → position in baseline arrays
        baseline_nid_to_pos = {str(pid): pos
                               for pos, pid in enumerate(baseline_df["post_id"])}

        pert_rr = None
        if reranker_spec:
            pert_rr = rerank_candidates(
                pert_texts, corpus_texts, pert_retrieved,
                reranker_spec=reranker_spec,
                batch_size=reranker_spec.batch_size or perf.reranker_batch_size,
                cache_path=CACHE_DIR / f"mcv2_{model_key}_{reranker_key}_pert_rr.npy",
                desc=f"Reranking pert [{model_key}]",
                perf=perf,
            )

        rows = build_result_rows(
            perturbed_df, pert_gold, pert_retrieved,
            model_key, "multiclaimv2",
            cluster_sets=pert_clusters,
            k_values=k_values,
            baseline_lookup=baseline_lookup,
            pert_reranked=pert_rr,
            reranker_key=reranker_key,
            # Paired baseline params
            baseline_retrieved=baseline_retrieved,
            baseline_reranked=baseline_rr,
            baseline_gold_idxs=baseline_gold,
            baseline_cluster_sets=baseline_clusters,
            baseline_nid_to_pos=baseline_nid_to_pos,
            id_col="post_id",
        )
        all_rows.extend(rows)
        print_delta_table(pd.DataFrame(rows), model_key, "MulticlaimV2")

        df_partial = pd.DataFrame(all_rows)
        df_partial.to_csv(OUTPUT_DIR / "multiclaimv2_results.csv", index=False)
        release_gpu_memory(label=f"end of model [{model_key}]")

    df_out = pd.DataFrame(all_rows)
    out_path = OUTPUT_DIR / "multiclaimv2_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults → {out_path}")
    return df_out


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MulticlaimV2 — Post to Fact-Check Retrieval Pipeline (H200)")
    p.add_argument("--models", default=",".join(MODEL_REGISTRY.keys()))
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--k-values", default="1,5,10,20")
    p.add_argument("--all-queries", action="store_true",
                   help="Include unverified rows (verified != True)")
    add_performance_args(p)
    add_reranker_args(p)
    args = p.parse_args()

    run(
        model_keys=[k.strip() for k in args.models.split(",") if k.strip() in MODEL_REGISTRY],
        top_k=args.top_k,
        k_values=tuple(int(k) for k in args.k_values.split(",")),
        only_verified=not args.all_queries,
        reranker_key=reranker_key_from_args(args),
        perf=perf_config_from_args(args),
    )
