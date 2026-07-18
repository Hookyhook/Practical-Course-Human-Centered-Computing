#!/usr/bin/env bash
# H200 / CUDA run script. Executes from repo root.
#
# Setup (once):
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#   pip install -r embeddings/requirements.txt
#   pip uninstall faiss-cpu -y 2>/dev/null; pip install faiss-gpu
#
# Tune batch sizes if OOM:
#   ENCODE_BATCH=256 RERANKER_BATCH=128 bash embeddings/run_gpu_pipelines.sh
#
# Model groups (pick one via MODEL_GROUP, or pass MODELS directly to override):
#   MODEL_GROUP=anchors  — original lightweight multilingual set,
#                          safe to run with no setup beyond requirements.txt.
#   MODEL_GROUP=paper     — the 6 English-only models added from the base paper.
#                          instructor-large has no special handling (deliberately,
#                          see pipeline_utils.py); sfr-embedding-mistral is 7B —
#                          already batch-capped in the registry.
#   MODEL_GROUP=new       — Qwen3-Embedding-8B + llama-embed-nemotron-8b (bonus,
#                          the latter is experimental/less battle-tested).
#   MODEL_GROUP=all       (default) — all 13 models.
#
# Within paper/all, the least-proven models (qwen3-embedding-8b,
# llama-embed-nemotron-8b — gated access, custom remote code, least
# battle-tested) run FIRST, ahead of the anchors and the rest of the paper
# models, so any remaining load/compatibility issue surfaces in minutes
# rather than after the lightweight models have already finished.
#
# llama-embed-nemotron-8b is gated on HuggingFace — run
# `huggingface-cli login` (or set HF_TOKEN) and accept its license first,
# or MODEL_GROUP=new/all will fail on that model specifically.
#
# Running paper/new/all pulls 7-8B param models — expect this to take
# meaningfully longer and use far more VRAM than the anchors-only group.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Reduces CUDA fragmentation OOMs when bi-encoder / reranker alternate.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Least-proven models first (gated access / custom remote code / newest),
# so issues surface immediately instead of after everything else finishes.
PRIORITY_MODELS="qwen3-embedding-8b,llama-embed-nemotron-8b"
ANCHORS_MODELS="me5,bge-m3,labse,mpnet-multi,minilm-multi"
PAPER_MODELS="all-mpnet-base-v2,all-minilm-l12,all-distilroberta,sentence-t5-large,instructor-large,sfr-embedding-mistral"

MODEL_GROUP="${MODEL_GROUP:-all}"
case "$MODEL_GROUP" in
  anchors) DEFAULT_MODELS="$ANCHORS_MODELS" ;;
  paper)   DEFAULT_MODELS="$PAPER_MODELS" ;;
  new)     DEFAULT_MODELS="qwen3-embedding-8b,llama-embed-nemotron-8b" ;;
  all)     DEFAULT_MODELS="$PRIORITY_MODELS,$ANCHORS_MODELS,$PAPER_MODELS" ;;
  *)       echo "Unknown MODEL_GROUP='$MODEL_GROUP' (want: anchors|paper|new|all)"; exit 1 ;;
esac

MODELS="${MODELS:-$DEFAULT_MODELS}"
ENCODE_BATCH="${ENCODE_BATCH:-512}"
RERANKER_BATCH="${RERANKER_BATCH:-256}"
V2_ENCODE_BATCH="${V2_ENCODE_BATCH:-256}"
V2_RERANKER_BATCH="${V2_RERANKER_BATCH:-128}"

echo "MODEL_GROUP=$MODEL_GROUP  →  MODELS=$MODELS"

echo "=== MulticlaimNet ==="
python embeddings/pipeline_multiclaimnet.py --models "$MODELS" \
  --encode-batch-size "$ENCODE_BATCH" --reranker-batch "$RERANKER_BATCH"

echo ""
echo "=== MulticlaimV2 ==="
python embeddings/pipeline_multiclaimv2.py --models "$MODELS" \
  --encode-batch-size "$V2_ENCODE_BATCH" --reranker-batch "$V2_RERANKER_BATCH"

echo ""
echo "Done. Results in results/multiclaimnet/ and results/multiclaimv2/"
