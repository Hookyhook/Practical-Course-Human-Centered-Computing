"""
Shared utilities for both embedding pipelines.
Imported by pipeline_multiclaimnet.py and pipeline_multiclaimv2.py.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DEFAULT_K_VALUES = (1, 5, 10, 20)
DEFAULT_ENCODE_BATCH = 2048
DEFAULT_RERANKER_BATCH = 1024

# Shared task description used to build instruction-style prefixes for the
# decoder-based / instruction-aware embedding models below (SFR-Embedding-Mistral,
# Qwen3-Embedding-8B, llama-embed-nemotron-8b, instructor-large).
_TASK_DESCRIPTION = "Given a claim, retrieve fact-checks that verify or refute it"


@dataclass
class ModelSpec:
    """Bi-encoder registry entry."""
    hf_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    normalize: bool = True
    trust_remote_code: bool = False
    # Per-model override for encode batch size. None → use PerfConfig/CLI default.
    # Set for the large (7-8B param) decoder-based models so a single global
    # --encode-batch-size doesn't OOM them while staying large for small models.
    batch_size: int | None = None
    max_seq_length: int | None = None
    # Force a specific attention implementation. Needed for
    # llama-embed-nemotron-8b, whose custom modeling code hard-asserts
    # `_attn_implementation in ("flash_attention_2", "eager")` — the
    # transformers default of "sdpa" is not in that list and would crash on
    # the first forward pass (not at load time) if left unset.
    attn_implementation: str | None = None


# ─── Bi-encoder model registry ─────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ── Existing multilingual anchors (unchanged) ───────────────────────────
    "labse": ModelSpec("sentence-transformers/LaBSE"),
    "me5": ModelSpec(
        "intfloat/multilingual-e5-large",
        query_prefix="query: ",    # e5 requires instruction prefixes
        passage_prefix="passage: ",
    ),
    "mpnet-multi": ModelSpec("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    "minilm-multi": ModelSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "bge-m3": ModelSpec("BAAI/bge-m3"),

    # ── From the base paper (English-only models — report separately from the
    #    multilingual set, see docs/pipeline_methodology.md) ────────────────
    "all-mpnet-base-v2": ModelSpec("sentence-transformers/all-mpnet-base-v2"),
    "all-minilm-l12": ModelSpec("sentence-transformers/all-MiniLM-L12-v2"),
    "all-distilroberta": ModelSpec("sentence-transformers/all-distilroberta-v1"),
    "sentence-t5-large": ModelSpec("sentence-transformers/sentence-t5-large"),
    # instructor-large: running with NO special adaptation on purpose, to see
    # whether the known sentence-transformers>=2.6 pooling incompatibility
    # (github.com/UKPLab/sentence-transformers/issues/2567) actually triggers
    # on our pinned version. If it crashes, this is the first thing to revisit.
    "instructor-large": ModelSpec(
        "hkunlp/instructor-large",
        query_prefix="Represent the claim for retrieving relevant fact-checks: ",
        passage_prefix="Represent the fact-check claim for retrieval: ",
    ),
    "sfr-embedding-mistral": ModelSpec(
        "Salesforce/SFR-Embedding-Mistral",
        query_prefix=f"Instruct: {_TASK_DESCRIPTION}\nQuery: ",
        batch_size=32,   # 7B params — global default batch would OOM
        # Without a cap, sentence-transformers defaults to this model's native
        # context length. One unusually long text in a batch (e.g. a verbose
        # perturbation rewrite) then dictates that whole batch's memory
        # footprint. 1024 comfortably covers our claims/posts and fact-check
        # text (FactCheckTweet's own avg fact-check length is ~682 tokens)
        # without leaving the door open to a runaway outlier. See
        # qwen3-embedding-8b's OOM postmortem for why this matters.
        max_seq_length=1024,
    ),

    # ── New, currently top-ranked multilingual models (beyond the paper) ────
    "qwen3-embedding-8b": ModelSpec(
        "Qwen/Qwen3-Embedding-8B",
        query_prefix=f"Instruct: {_TASK_DESCRIPTION}\nQuery: ",
        # OOM postmortem (MulticlaimV2, Phase 2, batch 0/8595): no cap meant
        # sentence-transformers defaulted to Qwen3's native context length;
        # one long perturbed text padded the whole batch to it, requesting a
        # single 12.78 GiB MLP intermediate tensor and exceeding the 139.8 GiB
        # GPU (128.12 GiB already in use before that one batch even ran, well
        # beyond what model weights + the FAISS index account for). Capping
        # max_seq_length + a smaller batch are the two actual levers here —
        # the earlier release_gpu_memory() checkpoints address fragmentation
        # from repeated calls, not a single oversized batch like this one.
        max_seq_length=1024,
        batch_size=16,   # halved from 32 as extra headroom alongside the cap
    ),
    # Experimental / bonus: very recently released, less battle-tested than
    # Qwen3-Embedding-8B. Smoke-test on a handful of texts before trusting it
    # in a full run — trust_remote_code=True is confirmed needed (repo ships
    # llama_bidirectional_model.py, a custom bidirectional-attention Llama
    # variant). attn_implementation="eager" is also required: that custom
    # code hard-asserts attention is "flash_attention_2" or "eager" and would
    # crash on the first forward pass under the transformers default ("sdpa").
    # No extra pip packages needed — the custom file only imports torch and
    # transformers internals, unlike NV-Embed-v2's einops/datasets gap.
    # License note: "customized-nscl-v1", not plain Apache — worth checking
    # before using results outside coursework.
    "llama-embed-nemotron-8b": ModelSpec(
        "nvidia/llama-embed-nemotron-8b",
        query_prefix=f"Instruct: {_TASK_DESCRIPTION}\nQuery: ",
        trust_remote_code=True,
        batch_size=32,   # 8B params
        attn_implementation="eager",
        # Same uncapped-context risk as qwen3-embedding-8b — see its OOM
        # postmortem above. Capping preemptively rather than waiting to hit it.
        max_seq_length=1024,
    ),
}

# ─── Cross-encoder reranker registry ───────────────────────────────────────────
@dataclass
class RerankerSpec:
    """Cross-encoder registry entry."""
    hf_id: str
    max_length: int = 512
    # Per-reranker override for the reranking batch size.
    # None → use PerfConfig/CLI default.
    batch_size: int | None = None


RERANKER_REGISTRY: dict[str, RerankerSpec] = {
    # bge-reranker-v2-m3 — multilingual cross-encoder (classification head,
    # single forward pass per batch), trained on MS-MARCO + multilingual data,
    # 560M params, fast.
    "bge-reranker-m3": RerankerSpec("BAAI/bge-reranker-v2-m3"),
}
DEFAULT_RERANKER = "bge-reranker-m3"


@dataclass
class PerfConfig:
    device: str = "cuda"
    encode_batch_size: int = DEFAULT_ENCODE_BATCH
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH
    use_bf16: bool = True
    faiss_gpu: bool = True


def require_cuda() -> str:
    import sys
    import torch
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA GPU required (H200). No CPU fallback.")
    return "cuda"


def resolve_device(device: str | None = None) -> str:
    """CUDA-only device resolver (kept for imports from legacy scripts)."""
    if device and device != "cuda" and not device.startswith("cuda:"):
        import sys
        sys.exit(f"ERROR: Only CUDA devices supported, got {device!r}")
    return device or require_cuda()


def add_performance_args(parser, *, default_encode_batch: int = DEFAULT_ENCODE_BATCH,
                         default_reranker_batch: int = DEFAULT_RERANKER_BATCH) -> None:
    g = parser.add_argument_group("performance (H200 defaults)")
    g.add_argument("--encode-batch-size", type=int, default=default_encode_batch,
                   help=f"Bi-encoder batch size (default: {default_encode_batch})")
    g.add_argument("--reranker-batch", type=int, default=default_reranker_batch,
                   help=f"Cross-encoder batch size (default: {default_reranker_batch})")


def add_reranker_args(parser) -> None:
    g = parser.add_argument_group("reranker")
    g.add_argument("--no-rerank", action="store_true",
                   help="Disable cross-encoder reranking (default: on)")
    g.add_argument("--rerank", default=DEFAULT_RERANKER,
                   choices=list(RERANKER_REGISTRY.keys()),
                   help=f"Cross-encoder reranker key (default: {DEFAULT_RERANKER})")


def perf_config_from_args(args) -> PerfConfig:
    return PerfConfig(
        encode_batch_size=getattr(args, "encode_batch_size", DEFAULT_ENCODE_BATCH),
        reranker_batch_size=getattr(args, "reranker_batch", DEFAULT_RERANKER_BATCH),
    )


def reranker_key_from_args(args) -> str | None:
    if getattr(args, "no_rerank", False):
        return None
    return getattr(args, "rerank", DEFAULT_RERANKER)


def filter_verified_queries(df: pd.DataFrame, only_verified: bool = True) -> pd.DataFrame:
    """Keep rows where the LLM verifier marked the perturbation as valid."""
    if not only_verified:
        return df
    if "verified" not in df.columns:
        raise ValueError("verified column required in perturbed query file")
    mask = df["verified"].astype(str).str.strip().str.lower() == "true"
    return df[mask].copy()


def load_bi_encoder(spec: "ModelSpec | str", perf: PerfConfig | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    if isinstance(spec, str):
        spec = ModelSpec(spec)   # legacy callers passing a bare model_id

    perf = perf or PerfConfig()
    device = resolve_device(perf.device)
    hf_model_kwargs: dict = {}
    if perf.use_bf16 and device.startswith("cuda"):
        hf_model_kwargs["torch_dtype"] = torch.bfloat16
    if spec.attn_implementation is not None:
        hf_model_kwargs["attn_implementation"] = spec.attn_implementation

    kwargs: dict = {}
    if hf_model_kwargs:
        kwargs["model_kwargs"] = hf_model_kwargs
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = SentenceTransformer(spec.hf_id, device=device, **kwargs)
    if spec.max_seq_length is not None:
        model.max_seq_length = spec.max_seq_length
    suffix = " bf16" if perf.use_bf16 and device.startswith("cuda") else ""
    extra = " trust_remote_code" if spec.trust_remote_code else ""
    print(f"  Device: {device}{suffix}{extra}")
    return model


def load_cross_encoder(spec: "RerankerSpec | str", perf: PerfConfig | None = None):
    from sentence_transformers import CrossEncoder

    if isinstance(spec, str):
        spec = RerankerSpec(spec)   # legacy callers passing a bare model_id

    perf = perf or PerfConfig()
    device = resolve_device(perf.device)
    # HF-model-level kwargs (torch_dtype etc.) must be nested under model_kwargs=
    # — passing torch_dtype as a bare top-level kwarg to CrossEncoder(...) raises
    # "TypeError: CrossEncoder.__init__() got an unexpected keyword argument
    # 'torch_dtype'" since CrossEncoder doesn't forward unrecognized top-level
    # kwargs.
    model_kwargs: dict = {}
    if perf.use_bf16 and device.startswith("cuda"):
        import torch
        model_kwargs["torch_dtype"] = torch.bfloat16

    return CrossEncoder(
        spec.hf_id,
        max_length=spec.max_length,
        device=device,
        model_kwargs=model_kwargs,
    )


def release_gpu_memory(*objs, label: str = "") -> None:
    """Drop references and return cached CUDA memory to the driver."""
    import gc
    import torch

    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if label:
        free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3) if torch.cuda.is_available() else 0
        print(f"  GPU memory released ({label}) — {free_gb:.1f} GiB free")


# ─── MAP@k ─────────────────────────────────────────────────────────────────────

def average_precision_at_k(retrieved_relevances: list, n_relevant: int, k: int) -> float:
    """
    AP@k for one query.  Standard TREC definition:
        AP@k = (1 / min(R, k)) * sum_{i=1}^{k}  P@i * rel(i)
    where R = n_relevant.
    """
    if n_relevant == 0:
        return 0.0
    denom = min(n_relevant, k)
    ap, running_hits = 0.0, 0
    for i, rel in enumerate(retrieved_relevances[:k], start=1):
        if rel:
            running_hits += 1
            ap += running_hits / i
    return ap / denom


def compute_map_at_k(
    query_retrieved_idxs: np.ndarray,   # (Q, top_k) FAISS output
    gold_idxs: list,                    # per-query single relevant corpus idx (or None)
    cluster_sets: list | None,          # per-query set of relevant corpus idxs (multi-relevant)
    k_values: tuple = DEFAULT_K_VALUES,
) -> dict:
    """
    Compute MAP@k for all k in k_values.
    - If cluster_sets is None  → single-relevant (gold_idxs used), MAP = MRR
    - If cluster_sets given    → multi-relevant (cluster_sets used), proper MAP

    Skips queries where gold is None (not found in corpus).
    Returns: MAP@k, MRR, R@1 for each k, plus n (valid queries).
    """
    max_k = max(k_values)
    aps = {k: [] for k in k_values}
    rr_list, r1_list = [], []

    for q_i, (gold_idx, retrieved) in enumerate(zip(gold_idxs, query_retrieved_idxs)):
        if gold_idx is None:
            continue

        relevant_set = cluster_sets[q_i] if cluster_sets is not None else {gold_idx}
        n_relevant = len(relevant_set)

        rel = [int(doc_idx in relevant_set) for doc_idx in retrieved[:max_k]]

        first_rank = next((r + 1 for r, x in enumerate(rel) if x), None)
        rr_list.append(1.0 / first_rank if first_rank else 0.0)
        r1_list.append(int(first_rank == 1) if first_rank else 0)

        for k in k_values:
            aps[k].append(average_precision_at_k(rel, n_relevant, k))

    n = len(rr_list)
    if n == 0:
        return {f"MAP@{k}": 0.0 for k in k_values} | {"MRR": 0.0, "R@1": 0.0, "n": 0}

    result = {f"MAP@{k}": float(np.mean(aps[k])) for k in k_values}
    result["MRR"] = float(np.mean(rr_list))
    result["R@1"] = float(np.mean(r1_list))
    result["n"] = n
    return result


def cluster_sets_excluding_self(
    cluster_sets: list[set],
    self_idxs: list,
) -> list[set]:
    """Remove each query's own corpus index from its relevant set."""
    return [
        (cluster_set - {self_idx}) if self_idx is not None else cluster_set
        for cluster_set, self_idx in zip(cluster_sets, self_idxs)
    ]


def filter_self_from_retrieved(
    retrieved: np.ndarray,
    self_idxs: list,
    top_k: int,
) -> np.ndarray:
    """
    Drop each query's own corpus index from FAISS results.

    Call with top_k+1 candidates retrieved upstream so each row stays length
    top_k after filtering when the self hit appears in the result list.
    """
    Q, _ = retrieved.shape
    out = np.full((Q, top_k), -1, dtype=retrieved.dtype)
    for q in range(Q):
        self_idx = self_idxs[q]
        kept = [int(idx) for idx in retrieved[q] if idx >= 0 and idx != self_idx]
        n = min(len(kept), top_k)
        if n:
            out[q, :n] = kept[:n]
    return out


# ─── Embedding + FAISS ─────────────────────────────────────────────────────────

def embed_texts(texts: list, model, prefix: str = "", batch_size: int = DEFAULT_ENCODE_BATCH,
                show_progress: bool = True) -> np.ndarray:
    """Return L2-normalized float32 embeddings."""
    encode_kwargs: dict = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": show_progress,
    }
    if prefix:
        texts = [prefix + t for t in texts]
    vecs = model.encode(texts, **encode_kwargs)
    return vecs.astype(np.float32)


def build_faiss_index(corpus_vecs: np.ndarray, use_gpu: bool = True):
    """Flat inner-product index on GPU (cosine sim on normalized vecs)."""
    import faiss

    if not use_gpu:
        raise ValueError("CPU FAISS is not supported — GPU required")
    corpus_vecs = np.ascontiguousarray(corpus_vecs.astype(np.float32))
    index = faiss.IndexFlatIP(corpus_vecs.shape[1])
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    print("  FAISS: GPU index")
    index.add(corpus_vecs)
    return index


def faiss_search(index, query_vecs: np.ndarray, top_k: int) -> np.ndarray:
    """Run FAISS search; query vectors must be float32 contiguous."""
    query_vecs = np.ascontiguousarray(query_vecs.astype(np.float32))
    _, idxs = index.search(query_vecs, top_k)
    return idxs


def load_or_embed(cache_path: Path, texts: list, model, prefix: str = "",
                  show_progress: bool = True,
                  batch_size: int = DEFAULT_ENCODE_BATCH) -> np.ndarray:
    """Load cached embeddings or compute and save them."""
    if cache_path.exists():
        print(f"  Cache hit: {cache_path.name}")
        return np.load(str(cache_path))
    print(f"  Embedding {len(texts):,} texts → {cache_path.name}")
    vecs = embed_texts(texts, model, prefix=prefix, show_progress=show_progress,
                       batch_size=batch_size)
    np.save(str(cache_path), vecs)
    return vecs


# ─── Cross-encoder reranking ───────────────────────────────────────────────────

def rerank_candidates(
    queries: list[str],
    corpus_texts: list[str],
    retrieved_idxs: np.ndarray,          # (Q, top_k) integer FAISS indices
    reranker_spec: "RerankerSpec | str",
    batch_size: int | None = None,
    cache_path: Path | None = None,
    desc: str = "Reranking",
    perf: PerfConfig | None = None,
) -> np.ndarray:
    """
    Cross-encoder reranking of FAISS top-k candidates.

    Scores all (query, candidate) pairs in large GPU batches (not per-query),
    then returns candidates sorted by descending score per query.

    Three-delta metrics (see paper):
      Δretrieval  = MAP@k(orig, first-stage)  − MAP@k(pert, first-stage)
      Δrecovery   = MAP@k(pert, reranked)     − MAP@k(pert, first-stage)  [gain from reranker]
      Δoverall    = MAP@k(orig, full-pipeline) − MAP@k(pert, full-pipeline)

    Results are cached to cache_path so subsequent runs are instant.

    batch_size, if given, overrides reranker_spec.batch_size, which overrides
    the DEFAULT_RERANKER_BATCH — matters a lot here since decoder-based
    rerankers (Qwen3) need a much smaller batch than classification-style
    cross-encoders (bge) to avoid OOM.
    """
    if isinstance(reranker_spec, str):
        reranker_spec = RerankerSpec(reranker_spec)   # legacy callers

    if cache_path is not None and Path(cache_path).exists():
        print(f"  Cache hit: {Path(cache_path).name}")
        return np.load(str(cache_path))

    effective_batch = batch_size or reranker_spec.batch_size or DEFAULT_RERANKER_BATCH

    print(f"  Loading reranker: {reranker_spec.hf_id}")
    reranker = load_cross_encoder(reranker_spec, perf)

    Q, top_k = retrieved_idxs.shape
    reranked = np.full((Q, top_k), -1, dtype=retrieved_idxs.dtype)

    # Flatten all pairs so the cross-encoder runs in large batches on GPU.
    all_pairs: list[tuple[str, str]] = []
    meta: list[tuple[int, int]] = []  # (query_idx, corpus_idx)
    for q_i in range(Q):
        query_text = queries[q_i]
        for c in retrieved_idxs[q_i]:
            c = int(c)
            if c < 0:
                continue
            all_pairs.append((query_text, corpus_texts[c]))
            meta.append((q_i, c))

    all_scores: list[float] = []
    for b_start in tqdm(range(0, len(all_pairs), effective_batch), desc=desc):
        batch = all_pairs[b_start: b_start + effective_batch]
        # Explicitly pass batch_size — CrossEncoder.predict() defaults to 32
        # internally, silently ignoring our per-model effective_batch tuning
        # (this was previously a real bug: registry batch_size overrides had
        # no effect on the actual call, every model ran at batch_size=32).
        scores = reranker.predict(batch, batch_size=effective_batch, show_progress_bar=False)
        all_scores.extend(scores.tolist())

    by_query: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (q_i, c), score in zip(meta, all_scores):
        by_query[q_i].append((score, c))
    for q_i in range(Q):
        ranked = sorted(by_query.get(q_i, []), key=lambda x: x[0], reverse=True)
        for j, (_, c) in enumerate(ranked[:top_k]):
            reranked[q_i, j] = c

    if cache_path is not None:
        np.save(str(cache_path), reranked)
        print(f"  Saved reranked indices → {Path(cache_path).name}")

    release_gpu_memory(reranker, label="cross-encoder")
    return reranked


# ─── Result aggregation ────────────────────────────────────────────────────────

MetricsDict = dict[str, float]


def _metric_groups(
    queries_df: pd.DataFrame,
    include_perturbation_groups: bool,
) -> list[tuple[str, str, pd.Index]]:
    groups: list[tuple[str, str, pd.Index]] = [("ALL", "ALL", queries_df.index)]
    if "pipeline_lang" in queries_df.columns:
        for lg in queries_df["pipeline_lang"].unique():
            groups.append(("ALL", lg, queries_df.index[queries_df["pipeline_lang"] == lg]))
    if include_perturbation_groups and "perturbation_name" in queries_df.columns:
        for pt in queries_df["perturbation_name"].unique():
            groups.append((pt, "ALL", queries_df.index[queries_df["perturbation_name"] == pt]))
            for lg in queries_df["pipeline_lang"].unique():
                mask_idx = queries_df.index[
                    (queries_df["perturbation_name"] == pt) & (queries_df["pipeline_lang"] == lg)
                ]
                if len(mask_idx):
                    groups.append((pt, lg, mask_idx))
    return groups


def compute_grouped_metrics(
    queries_df: pd.DataFrame,
    gold_corpus_idxs: list,
    retrieved: np.ndarray,
    cluster_sets: list | None,
    k_values: tuple = DEFAULT_K_VALUES,
    reranked: np.ndarray | None = None,
    include_perturbation_groups: bool = True,
) -> dict[tuple[str, str], dict[str, MetricsDict]]:
    """
    Compute MAP@k (and optional reranked MAP) per group.

    Returns: {(pert_name, lang): {"first": metrics, "rr": metrics|None}}
    """
    lookup: dict[tuple[str, str], dict[str, MetricsDict]] = {}

    for pert_name, lang, idx in _metric_groups(queries_df, include_perturbation_groups):
        if len(idx) == 0:
            continue
        pos = [queries_df.index.get_loc(i) for i in idx]
        sub_gold = [gold_corpus_idxs[p] for p in pos]
        sub_ret = retrieved[pos]
        sub_clusters = ([cluster_sets[p] for p in pos] if cluster_sets is not None else None)

        entry: dict[str, MetricsDict] = {
            "first": compute_map_at_k(sub_ret, sub_gold, sub_clusters, k_values),
        }
        if reranked is not None:
            entry["rr"] = compute_map_at_k(reranked[pos], sub_gold, sub_clusters, k_values)
        else:
            entry["rr"] = None
        lookup[(pert_name, lang)] = entry

    return lookup


def build_baseline_lookup(
    baseline_df: pd.DataFrame,
    gold_corpus_idxs: list,
    retrieved: np.ndarray,
    cluster_sets: list | None,
    k_values: tuple = DEFAULT_K_VALUES,
    reranked: np.ndarray | None = None,
) -> dict[tuple[str, str], dict[str, MetricsDict]]:
    """Phase-1 baseline metrics keyed by (pert_name, lang) — no perturbation groups."""
    return compute_grouped_metrics(
        baseline_df, gold_corpus_idxs, retrieved, cluster_sets, k_values,
        reranked=reranked, include_perturbation_groups=False,
    )


def _baseline_entry_for_lang(
    baseline_lookup: dict[tuple[str, str], dict[str, MetricsDict]],
    lang: str,
) -> dict[str, MetricsDict]:
    key = ("ALL", "ALL") if lang == "ALL" else ("ALL", lang)
    return baseline_lookup.get(key, {"first": {}, "rr": None})


def build_result_rows(
    queries_df: pd.DataFrame,
    gold_corpus_idxs: list,
    pert_retrieved: np.ndarray,
    model_key: str,
    dataset_name: str,
    cluster_sets: list | None,
    k_values: tuple = DEFAULT_K_VALUES,
    baseline_lookup: dict[tuple[str, str], dict[str, MetricsDict]] | None = None,
    pert_reranked: np.ndarray | None = None,
    reranker_key: str | None = None,
    # Legacy single-pass mode (baseline_lookup=None): pass both orig and pert arrays
    orig_retrieved: np.ndarray | None = None,
    orig_reranked: np.ndarray | None = None,
    # Paired baseline mode: compute orig on the same NID subset as each pert group
    baseline_retrieved: np.ndarray | None = None,
    baseline_reranked: np.ndarray | None = None,
    baseline_gold_idxs: list | None = None,
    baseline_cluster_sets: list | None = None,
    baseline_nid_to_pos: dict[str, int] | None = None,
    id_col: str = "NID",
) -> list[dict]:
    """
    Compute MAP@k for each (perturbation_name, language) combo plus aggregate rows.

    Paired baseline mode (baseline_nid_to_pos provided) — RECOMMENDED:
      For each (pert, lang) group, orig is computed over the same NID subset that
      was verified for that group.  This matches the paper's methodology and avoids
      selection-bias inflation of Δretrieval for low-coverage perturbations.
      Pass: baseline_retrieved, baseline_gold_idxs, baseline_cluster_sets,
            baseline_nid_to_pos, id_col (default "NID").
      Optionally pass baseline_reranked for paired orig_rr.

    Two-phase mode (baseline_lookup provided, no baseline_nid_to_pos):
      MAP@k_orig comes from the pre-computed global language-level baseline.
      Kept for backward compatibility; produces unpaired deltas.

    Legacy single-pass mode (baseline_lookup=None, orig_retrieved provided):
      Requires orig_retrieved aligned with queries_df.
    """
    paired_mode = baseline_nid_to_pos is not None and baseline_retrieved is not None

    if not paired_mode and baseline_lookup is None and orig_retrieved is None:
        raise ValueError("Provide baseline_nid_to_pos+baseline_retrieved (paired), "
                         "baseline_lookup (global), or orig_retrieved (legacy).")

    has_reranker = pert_reranked is not None
    if has_reranker:
        if paired_mode:
            has_reranker = baseline_reranked is not None
        elif baseline_lookup is not None:
            has_rr_baseline = any(v.get("rr") for v in baseline_lookup.values())
            has_reranker = has_rr_baseline
        else:
            has_reranker = orig_reranked is not None and pert_reranked is not None

    rows = []

    groups = _metric_groups(queries_df, include_perturbation_groups=True)

    for pert_name, lang, idx in groups:
        if len(idx) == 0:
            continue

        pos = [queries_df.index.get_loc(i) for i in idx]
        sub_gold     = [gold_corpus_idxs[p] for p in pos]
        sub_pert     = pert_retrieved[pos]
        sub_clusters = ([cluster_sets[p] for p in pos] if cluster_sets is not None else None)

        if paired_mode:
            # Paired: resolve each query's NID → position in baseline arrays
            group_nids = queries_df[id_col].iloc[pos].tolist()
            base_pos = [baseline_nid_to_pos[str(n)] for n in group_nids
                        if str(n) in baseline_nid_to_pos]
            if base_pos:
                bp = np.array(base_pos, dtype=np.intp)
                sub_base_gold = [baseline_gold_idxs[p] for p in base_pos]
                sub_base_cl   = ([baseline_cluster_sets[p] for p in base_pos]
                                 if baseline_cluster_sets is not None else None)
                orig_m    = compute_map_at_k(baseline_retrieved[bp],
                                             sub_base_gold, sub_base_cl, k_values)
                orig_rr_m = (compute_map_at_k(baseline_reranked[bp],
                                              sub_base_gold, sub_base_cl, k_values)
                             if baseline_reranked is not None else None)
            else:
                # Fallback: no overlap with baseline (shouldn't happen)
                bl = _baseline_entry_for_lang(baseline_lookup or {}, lang)
                orig_m    = bl.get("first", {})
                orig_rr_m = bl.get("rr")
        elif baseline_lookup is not None:
            bl = _baseline_entry_for_lang(baseline_lookup, lang)
            orig_m = bl["first"]
            orig_rr_m = bl.get("rr")
        else:
            sub_orig = orig_retrieved[pos]
            orig_m = compute_map_at_k(sub_orig, sub_gold, sub_clusters, k_values)
            orig_rr_m = (
                compute_map_at_k(orig_reranked[pos], sub_gold, sub_clusters, k_values)
                if orig_reranked is not None else None
            )

        pert_m = compute_map_at_k(sub_pert, sub_gold, sub_clusters, k_values)

        row: dict = {
            "model":             model_key,
            "reranker":          reranker_key or "",
            "dataset":           dataset_name,
            "perturbation_name": pert_name,
            "language":          lang,
        }

        if pert_name != "ALL":
            fam_rows = queries_df[queries_df["perturbation_name"] == pert_name]
            row["family"] = fam_rows["family"].iloc[0] if "family" in fam_rows.columns else ""
        else:
            row["family"] = "ALL"

        for k in k_values:
            row[f"MAP@{k}_orig"]  = orig_m.get(f"MAP@{k}", 0.0)
            row[f"MAP@{k}_pert"]  = pert_m[f"MAP@{k}"]
            row[f"delta_MAP@{k}"] = row[f"MAP@{k}_orig"] - pert_m[f"MAP@{k}"]

        row["MRR_orig"]  = orig_m.get("MRR", 0.0)
        row["MRR_pert"]  = pert_m["MRR"]
        row["delta_MRR"] = row["MRR_orig"] - pert_m["MRR"]
        row["R@1_orig"]  = orig_m.get("R@1", 0.0)
        row["R@1_pert"]  = pert_m["R@1"]
        row["n"]         = pert_m["n"]

        if has_reranker and pert_reranked is not None:
            sub_pert_rr = pert_reranked[pos]
            pert_rr_m = compute_map_at_k(sub_pert_rr, sub_gold, sub_clusters, k_values)

            if orig_rr_m is None:
                orig_rr_m = orig_m

            for k in k_values:
                row[f"MAP@{k}_orig_rr"]        = orig_rr_m.get(f"MAP@{k}", row[f"MAP@{k}_orig"])
                row[f"MAP@{k}_pert_rr"]        = pert_rr_m[f"MAP@{k}"]
                row[f"delta_recovery_MAP@{k}"] = pert_rr_m[f"MAP@{k}"] - pert_m[f"MAP@{k}"]
                row[f"delta_overall_MAP@{k}"]  = row[f"MAP@{k}_orig_rr"] - pert_rr_m[f"MAP@{k}"]

            row["MRR_orig_rr"]        = orig_rr_m.get("MRR", row["MRR_orig"])
            row["MRR_pert_rr"]        = pert_rr_m["MRR"]
            row["delta_recovery_MRR"] = pert_rr_m["MRR"] - pert_m["MRR"]
            row["delta_overall_MRR"]  = row["MRR_orig_rr"] - pert_rr_m["MRR"]

        rows.append(row)

    return rows


def print_delta_table(df: pd.DataFrame, model_key: str, dataset_name: str):
    """Print the Δretrieval table (and reranker deltas if present) for one model."""
    has_rr = "delta_recovery_MAP@20" in df.columns

    print(f"\n{'─'*70}")
    print(f"Δretrieval — {model_key} — {dataset_name}")

    # Overall row
    overall = df[(df["model"] == model_key) & (df["perturbation_name"] == "ALL")
                 & (df["language"] == "ALL")]
    if not overall.empty:
        r = overall.iloc[0]
        line = (f"  OVERALL  MAP@20: orig={r['MAP@20_orig']:.3f}  "
                f"pert={r['MAP@20_pert']:.3f}  Δretrieval={r['delta_MAP@20']:+.3f}")
        if has_rr:
            line += (f"  |  pert_rr={r['MAP@20_pert_rr']:.3f}"
                     f"  Δrecovery={r['delta_recovery_MAP@20']:+.3f}"
                     f"  Δoverall={r['delta_overall_MAP@20']:+.3f}")
        print(line)

    # Per perturbation type
    pert_rows = df[(df["model"] == model_key) & (df["perturbation_name"] != "ALL")
                   & (df["language"] == "ALL")].sort_values("delta_MAP@20", ascending=False)
    if has_rr:
        hdr = f"  {'Perturbation':30s}  {'orig':>6}  {'pert':>6}  {'Δretr':>7}  {'pert_rr':>7}  {'Δrecov':>7}  {'Δovrl':>7}"
    else:
        hdr = f"  {'Perturbation':30s}  {'MAP@20 orig':>11}  {'MAP@20 pert':>11}  {'Δ':>7}"
    print(f"\n{hdr}")
    for _, r in pert_rows.iterrows():
        if has_rr:
            print(f"  {r['perturbation_name']:30s}  {r['MAP@20_orig']:>6.3f}  {r['MAP@20_pert']:>6.3f}"
                  f"  {r['delta_MAP@20']:>+7.3f}  {r['MAP@20_pert_rr']:>7.3f}"
                  f"  {r['delta_recovery_MAP@20']:>+7.3f}  {r['delta_overall_MAP@20']:>+7.3f}")
        else:
            print(f"  {r['perturbation_name']:30s}  {r['MAP@20_orig']:>11.3f}  "
                  f"{r['MAP@20_pert']:>11.3f}  {r['delta_MAP@20']:>+7.3f}")

    # Per language
    lang_rows = df[(df["model"] == model_key) & (df["perturbation_name"] == "ALL")
                   & (df["language"] != "ALL")].sort_values("delta_MAP@20", ascending=False)
    if has_rr:
        hdr2 = f"  {'Language':10s}  {'orig':>6}  {'pert':>6}  {'Δretr':>7}  {'pert_rr':>7}  {'Δrecov':>7}  {'Δovrl':>7}"
    else:
        hdr2 = f"  {'Language':10s}  {'MAP@20 orig':>11}  {'MAP@20 pert':>11}  {'Δ':>7}"
    print(f"\n{hdr2}")
    for _, r in lang_rows.iterrows():
        if has_rr:
            print(f"  {r['language']:10s}  {r['MAP@20_orig']:>6.3f}  {r['MAP@20_pert']:>6.3f}"
                  f"  {r['delta_MAP@20']:>+7.3f}  {r['MAP@20_pert_rr']:>7.3f}"
                  f"  {r['delta_recovery_MAP@20']:>+7.3f}  {r['delta_overall_MAP@20']:>+7.3f}")
        else:
            print(f"  {r['language']:10s}  {r['MAP@20_orig']:>11.3f}  "
                  f"{r['MAP@20_pert']:>11.3f}  {r['delta_MAP@20']:>+7.3f}")
