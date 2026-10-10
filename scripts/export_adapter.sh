#!/bin/sh
set -eu

ADAPTER_DIR=${1:?usage: export_adapter.sh ADAPTER_DIR OUTPUT_DIR}
OUTPUT_DIR=${2:?usage: export_adapter.sh ADAPTER_DIR OUTPUT_DIR}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$REPO_DIR/.." && pwd)

SIF_PATH=${SIF_PATH:-$REPO_DIR/containers/structure-over-scale-cu128.sif}
MODEL_DIR=${MODEL_DIR:-$PROJECT_DIR/models/qwen3-8b}
CACHE_DIR=${CACHE_DIR:-$PROJECT_DIR/cache}

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

export SINGULARITYENV_HF_HOME=/mnt/cache/huggingface
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_PYTHONPATH=/workspace/structure_over_scale/src

exec singularity exec --nv --cleanenv \
    --bind "$REPO_DIR:/workspace/structure_over_scale:ro" \
    --bind "$MODEL_DIR:/mnt/model:ro" \
    --bind "$ADAPTER_DIR:/mnt/adapter:ro" \
    --bind "$OUTPUT_DIR:/mnt/output" \
    --bind "$CACHE_DIR:/mnt/cache" \
    --pwd /workspace/structure_over_scale \
    "$SIF_PATH" \
    python src/export_adapter.py \
        --base-model /mnt/model \
        --adapter /mnt/adapter \
        --output /mnt/output
