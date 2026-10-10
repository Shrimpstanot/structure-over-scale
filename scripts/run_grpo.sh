#!/bin/sh
set -eu

RUN_ID=${1:?usage: run_grpo.sh RUN_ID [NPROC] [SEED] [PAPER_BATCH_SIZE] [GROUP_SIZE]}
NPROC=${2:-4}
SEED=${3:-42}
PAPER_BATCH_SIZE=${4:-64}
GROUP_SIZE=${5:-}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$REPO_DIR/.." && pwd)

SIF_PATH=${SIF_PATH:-$REPO_DIR/containers/structure-over-scale-cu128.sif}
MODEL_DIR=${MODEL_DIR:-$PROJECT_DIR/models/qwen3-8b}
DATASET_DIR=${DATASET_DIR:-$PROJECT_DIR/datasets/deepmath-hard}
RUNS_DIR=${RUNS_DIR:-$PROJECT_DIR/runs}
CACHE_DIR=${CACHE_DIR:-$PROJECT_DIR/cache}

RESULT_REL=$RUN_ID/seed-$SEED
if [ -n "$GROUP_SIZE" ] && [ "$GROUP_SIZE" != 8 ]; then
    RESULT_REL=$RESULT_REL/g-$GROUP_SIZE
fi
RESULT_REL=$RESULT_REL/bs-$PAPER_BATCH_SIZE
RESULT_HOST=$RUNS_DIR/$RESULT_REL
CONFIG_HOST=$RESULT_HOST/resolved.yaml
CONFIG_CONTAINER=/mnt/runs/$RESULT_REL/resolved.yaml

mkdir -p "$RESULT_HOST" "$CACHE_DIR/huggingface" "$CACHE_DIR/torch" "$CACHE_DIR/home"

export SINGULARITYENV_HOME=/mnt/cache/home
export SINGULARITYENV_HF_HOME=/mnt/cache/huggingface
export SINGULARITYENV_HF_DATASETS_CACHE=/mnt/cache/huggingface/datasets
export SINGULARITYENV_TRANSFORMERS_CACHE=/mnt/cache/huggingface/transformers
export SINGULARITYENV_TORCH_HOME=/mnt/cache/torch
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_HF_DATASETS_OFFLINE=1
export SINGULARITYENV_PYTHONPATH=/workspace/structure_over_scale/src

set -- "$RUN_ID" \
    --seed "$SEED" \
    --paper-batch-size "$PAPER_BATCH_SIZE" \
    --nproc-per-node "$NPROC" \
    --output "$CONFIG_CONTAINER"
if [ -n "$GROUP_SIZE" ]; then
    set -- "$@" --group-size "$GROUP_SIZE"
fi

singularity exec --nv --cleanenv \
    --bind "$REPO_DIR:/workspace/structure_over_scale:ro" \
    --bind "$MODEL_DIR:/mnt/model:ro" \
    --bind "$DATASET_DIR:/mnt/data:ro" \
    --bind "$RUNS_DIR:/mnt/runs" \
    --bind "$CACHE_DIR:/mnt/cache" \
    --pwd /workspace/structure_over_scale \
    "$SIF_PATH" \
    python src/materialize_config.py "$@"

exec singularity exec --nv --cleanenv \
    --bind "$REPO_DIR:/workspace/structure_over_scale:ro" \
    --bind "$MODEL_DIR:/mnt/model:ro" \
    --bind "$DATASET_DIR:/mnt/data:ro" \
    --bind "$RUNS_DIR:/mnt/runs" \
    --bind "$CACHE_DIR:/mnt/cache" \
    --pwd /workspace/structure_over_scale \
    "$SIF_PATH" \
    python src/train_grpo.py --config "$CONFIG_CONTAINER" --nproc_per_node "$NPROC"
