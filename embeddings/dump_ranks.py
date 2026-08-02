"""
dump_ranks.py — per-query gold-target ranks from the cached embeddings
=====================================================================
The evaluation pipelines aggregate to MAP@k per (model, perturbation, language)
cell, so no per-query rank survives into results/*.csv.  Several worked
examples in the paper need exactly that.  This script reconstructs ranks from
`.embedding_cache/` without re-embedding anything and without a GPU.

What it reports, per (model, query, perturbation):

    rank_orig      rank of the best relevant corpus item for the ORIGINAL query
    rank_pert      same, for the perturbed query
    rank_orig_rr   rank after cross-encoder reranking (from the cached
                   reranked index arrays; None if outside the top-k window)
    rank_pert_rr   same, perturbed

Ranks are computed over the FULL corpus by inner product on normalised
vectors — the same scoring FAISS uses — so a gold item that falls out of the
top-50 still gets a real rank instead of "not retrieved".  For Task A the
query's own claim is excluded from both the corpus ranking and the relevant
set, matching `filter_self_from_retrieved` / `cluster_sets_excluding_self`.

Usage
-----
  # named examples (Task A NIDs, Task B post_ids)
  python embeddings/dump_ranks.py --task A --ids 24,71 \
      --models qwen3-embedding-8b,all-distilroberta \
      --perturbations P_dialect_jamaican,P_dialect_aae \
      --out results/ranks_dialect.csv

  # the §3.2 worked example: one claim under every perturbation, one model
  python embeddings/dump_ranks.py --task A --ids 24 --models bge-m3 \
      --out results/ranks_worked_example.csv

  # App. D: find queries where an edit IMPROVED the rank for one model
  python embeddings/dump_ranks.py --task A --scan \
      --perturbations A1_emoji_relevant,A2_hashtagification \
      --models llama-embed-nemotron-8b --languages EN \
      --limit 3000 --sort improved --top 25 --out results/ranks_improved.csv

Requires: numpy, pandas (torch is used automatically if importable, for speed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import (  # noqa: E402
    MODEL_REGISTRY, DEFAULT_RERANKER, filter_verified_queries,
    cluster_sets_excluding_self,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = Path(".embedding_cache")


# ─── data loading (mirrors the two pipelines exactly) ─────────────────────────

def _resolve(*candidates: Path) -> Path:
    """The Task A baseline is checked in both with and without a .csv suffix."""
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"none of these exist: {[str(c) for c in candidates]}")


def load_task_a():
    corpus_df = pd.read_csv(DATA_DIR / "raw" / "MultiClaim.csv", dtype=str).fillna("")
    corpus_texts = corpus_df["Claim"].tolist()
    nid_to_idx = {nid: i for i, nid in enumerate(corpus_df["NID"].tolist())}
    cluster_to_idxs: dict[str, set] = {}
    for i, cid in enumerate(corpus_df["ClusterID"].tolist()):
        cluster_to_idxs.setdefault(cid, set()).add(i)

    baseline_df = pd.read_csv(
        _resolve(DATA_DIR / "processed" / "multiclaimnet_preprocessed.csv",
                 DATA_DIR / "processed" / "multiclaimnet_preprocessed"),
        dtype=str).fillna("")
    baseline_df.reset_index(drop=True, inplace=True)

    pert_df = pd.read_csv(DATA_DIR / "processed" / "multiclaimnet_verified.csv",
                          dtype=str).fillna("")
    pert_df = filter_verified_queries(pert_df, only_verified=True)
    pert_df.reset_index(drop=True, inplace=True)

    def qrels(df):
        gold = [nid_to_idx.get(nid) for nid in df["NID"]]
        rel = cluster_sets_excluding_self(
            [cluster_to_idxs.get(cid, set()) for cid in df["ClusterID"]], gold)
        return gold, rel

    return dict(
        prefix="mcnet", id_col="NID", corpus_texts=corpus_texts,
        baseline_df=baseline_df, baseline_text_col="Claim",
        pert_df=pert_df, qrels=qrels,
        corpus_cache="multiclaim_raw_corpus",
    )


def load_task_b():
    mapping = pd.read_csv(DATA_DIR / "raw" / "fact_check_post_mapping.csv", dtype=str)
    fc = pd.read_csv(DATA_DIR / "raw" / "fact_checks.csv", dtype=str).fillna("")
    fc = fc.drop_duplicates(subset="fact_check_id", keep="first").sort_values("fact_check_id")
    corpus_texts = fc["claim"].tolist()
    fc_id_to_idx = {f: i for i, f in enumerate(fc["fact_check_id"].tolist())}

    post_to_fc: dict[str, set] = {}
    for _, row in mapping.iterrows():
        idx = fc_id_to_idx.get(row["fact_check_id"])
        if idx is not None:
            post_to_fc.setdefault(row["post_id"], set()).add(idx)

    baseline_df = pd.read_csv(DATA_DIR / "processed" / "multiclaimv2_preprocessed.csv",
                              dtype=str).fillna("")
    baseline_df.reset_index(drop=True, inplace=True)

    pert_df = pd.read_csv(DATA_DIR / "processed" / "multiclaimv2_verified.csv",
                          dtype=str).fillna("")
    pert_df = filter_verified_queries(pert_df, only_verified=True)
    pert_df.reset_index(drop=True, inplace=True)

    def qrels(df):
        rel = [post_to_fc.get(pid, set()) for pid in df["post_id"]]
        gold = [next(iter(s)) if s else None for s in rel]
        return gold, rel

    return dict(
        prefix="mcv2", id_col="post_id", corpus_texts=corpus_texts,
        baseline_df=baseline_df, baseline_text_col="post_body",
        pert_df=pert_df, qrels=qrels,
        corpus_cache="factchecks_corpus",
    )


# ─── scoring ──────────────────────────────────────────────────────────────────

def _rank_of(corpus, qvecs, targets, exclude, chunk=200_000):
    """
    Rank (1-based, full corpus) of the best item in `targets[i]` for query i.

    corpus  : (N, D) array, memory-mapped
    qvecs   : (Q, D)
    targets : list of sets of corpus indices (the relevant set)
    exclude : list of corpus indices to drop from the ranking (self hit), or None
    Returns (ranks, chosen_idx) — chosen_idx is the highest-scoring relevant item.
    """
    Q = len(qvecs)
    qv = np.ascontiguousarray(qvecs, dtype=np.float32)

    # score of each query's relevant candidates, and its best one
    best_idx = np.full(Q, -1, dtype=np.int64)
    best_score = np.full(Q, -np.inf, dtype=np.float32)
    for i, tgt in enumerate(targets):
        if not tgt:
            continue
        idxs = np.fromiter(tgt, dtype=np.int64)
        s = np.asarray(corpus[idxs], dtype=np.float32) @ qv[i]
        j = int(np.argmax(s))
        best_idx[i], best_score[i] = idxs[j], s[j]

    greater = np.zeros(Q, dtype=np.int64)
    N = corpus.shape[0]
    for start in range(0, N, chunk):
        block = np.asarray(corpus[start:start + chunk], dtype=np.float32)
        sims = block @ qv.T                      # (chunk, Q)
        greater += (sims > best_score[None, :]).sum(axis=0)

    ranks = greater + 1
    # a self hit that outranks the target inflated the count by one
    if exclude is not None:
        for i, ex in enumerate(exclude):
            if ex is None or best_idx[i] < 0:
                continue
            self_score = float(np.asarray(corpus[ex], dtype=np.float32) @ qv[i])
            if self_score > best_score[i]:
                ranks[i] -= 1
    ranks = np.where(best_idx < 0, -1, ranks)
    return ranks, best_idx


def _rr_rank(rr_array, row, target_idx):
    """Rank of target_idx inside a cached reranked candidate row, else None."""
    if rr_array is None or target_idx < 0 or row >= len(rr_array):
        return None
    hit = np.where(rr_array[row] == target_idx)[0]
    return int(hit[0]) + 1 if len(hit) else None


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["A", "B"], required=True,
                    help="A = claim clustering (mcnet), B = post->fact-check (mcv2)")
    ap.add_argument("--models", default="bge-m3",
                    help="comma-separated MODEL_REGISTRY keys")
    ap.add_argument("--ids", default="", help="comma-separated NIDs (A) / post_ids (B)")
    ap.add_argument("--perturbations", default="", help="comma-separated perturbation_name")
    ap.add_argument("--languages", default="", help="comma-separated pipeline_lang")
    ap.add_argument("--scan", action="store_true",
                    help="ignore --ids and sweep every matching query (use --limit)")
    ap.add_argument("--limit", type=int, default=200, help="max queries in --scan mode")
    ap.add_argument("--sort", choices=["none", "improved", "worsened"], default="none",
                    help="order the output by rank change")
    ap.add_argument("--top", type=int, default=0, help="keep only the first N rows after --sort")
    ap.add_argument("--reranker", default=DEFAULT_RERANKER)
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--chunk", type=int, default=200_000, help="corpus rows scored per block")
    ap.add_argument("--out", default="results/ranks.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="check data alignment and cache paths, then exit")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    task = load_task_a() if args.task == "A" else load_task_b()
    id_col = task["id_col"]
    pert_df, baseline_df = task["pert_df"], task["baseline_df"]

    sel = pd.Series(True, index=pert_df.index)
    if args.perturbations:
        sel &= pert_df["perturbation_name"].isin(args.perturbations.split(","))
    if args.languages:
        sel &= pert_df["pipeline_lang"].isin(args.languages.split(","))
    if args.ids and not args.scan:
        sel &= pert_df[id_col].astype(str).isin(args.ids.split(","))
    rows = pert_df[sel]
    if args.scan:
        rows = rows.head(args.limit)
    if rows.empty:
        sys.exit("no perturbed rows matched the filters")

    base_pos = {str(v): i for i, v in enumerate(baseline_df[id_col])}
    keep = rows[id_col].astype(str).isin(base_pos)
    if (~keep).any():
        print(f"  dropping {(~keep).sum()} rows whose {id_col} is absent from the baseline file")
    rows = rows[keep]

    pert_positions = rows.index.to_numpy()                       # into pert_df / pert cache
    base_positions = np.array([base_pos[str(v)] for v in rows[id_col]])
    pert_gold, pert_rel = task["qrels"](rows)
    base_slice = baseline_df.iloc[base_positions]
    base_gold, base_rel = task["qrels"](base_slice)

    print(f"Task {args.task}: {len(rows)} query/perturbation pairs, "
          f"{rows[id_col].nunique()} distinct {id_col}s, "
          f"{rows['perturbation_name'].nunique()} perturbation types")

    out = []
    for model_key in args.models.split(","):
        if model_key not in MODEL_REGISTRY:
            sys.exit(f"unknown model key: {model_key}")
        p = task["prefix"]
        paths = {
            "corpus": cache / f"{p}_{model_key}_{task['corpus_cache']}.npy",
            "base": cache / f"{p}_{model_key}_baseline_queries.npy",
            "pert": cache / f"{p}_{model_key}_pert_queries.npy",
            "base_rr": cache / f"{p}_{model_key}_{args.reranker}_baseline_rr.npy",
            "pert_rr": cache / f"{p}_{model_key}_{args.reranker}_pert_rr.npy",
        }
        missing = [k for k in ("corpus", "base", "pert") if not paths[k].exists()]
        if missing:
            print(f"  [{model_key}] SKIP — missing cache: "
                  f"{', '.join(paths[k].name for k in missing)}")
            continue
        if args.dry_run:
            print(f"  [{model_key}] caches present: "
                  f"{', '.join(k for k, v in paths.items() if v.exists())}")
            continue

        corpus = np.load(paths["corpus"], mmap_mode="r")
        base_vecs = np.load(paths["base"], mmap_mode="r")
        pert_vecs = np.load(paths["pert"], mmap_mode="r")
        if len(pert_vecs) != len(pert_df):
            sys.exit(f"[{model_key}] pert cache has {len(pert_vecs)} rows but the verified "
                     f"frame has {len(pert_df)} — the cache predates the current CSV")
        base_rr = np.load(paths["base_rr"], mmap_mode="r") if paths["base_rr"].exists() else None
        pert_rr = np.load(paths["pert_rr"], mmap_mode="r") if paths["pert_rr"].exists() else None

        print(f"  [{model_key}] corpus {corpus.shape}, scoring {len(rows)} queries...")
        exclude_o = base_gold if args.task == "A" else None
        exclude_p = pert_gold if args.task == "A" else None
        r_o, tgt_o = _rank_of(corpus, base_vecs[base_positions], base_rel, exclude_o, args.chunk)
        r_p, tgt_p = _rank_of(corpus, pert_vecs[pert_positions], pert_rel, exclude_p, args.chunk)

        for k, (_, row) in enumerate(rows.iterrows()):
            out.append({
                "task": args.task, "model": model_key,
                id_col: row[id_col], "language": row.get("pipeline_lang", ""),
                "perturbation": row["perturbation_name"], "family": row.get("family", ""),
                "rank_orig": int(r_o[k]), "rank_pert": int(r_p[k]),
                "rank_delta": int(r_p[k] - r_o[k]),
                "rank_orig_rr": _rr_rank(base_rr, base_positions[k], int(tgt_o[k])),
                "rank_pert_rr": _rr_rank(pert_rr, pert_positions[k], int(tgt_p[k])),
                "gold_corpus_idx": int(tgt_o[k]),
                "n_relevant": len(base_rel[k]),
                "original_text": base_slice.iloc[k][task["baseline_text_col"]],
                "perturbed_text": row["perturbed_text"],
                "gold_text": task["corpus_texts"][int(tgt_o[k])] if tgt_o[k] >= 0 else "",
            })

    if args.dry_run:
        print("dry run OK — data frames aligned, no ranks computed")
        return
    if not out:
        sys.exit("nothing computed (all models skipped?)")

    df = pd.DataFrame(out)
    for c in ("rank_orig_rr", "rank_pert_rr"):
        df[c] = df[c].astype("Int64")     # keeps "outside the rerank window" as <NA>
    if args.sort == "improved":
        df = df.sort_values("rank_delta")                 # most negative = biggest improvement
    elif args.sort == "worsened":
        df = df.sort_values("rank_delta", ascending=False)
    if args.top:
        df = df.head(args.top)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n{len(df)} rows → {args.out}")
    cols = ["model", id_col, "perturbation", "rank_orig", "rank_pert",
            "rank_orig_rr", "rank_pert_rr"]
    print(df[cols].to_string(index=False, max_rows=40))


if __name__ == "__main__":
    main()
