#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for PAPER_BATCH_SIZE in 64 128 256
do
    for RUN_ID in \
        lora-r8 lora-r64 lora-r128 lora-r256 dora-r16 quanta-d3-no-kv
    do
        "$SCRIPT_DIR/run_grpo.sh" "$RUN_ID" 4 42 "$PAPER_BATCH_SIZE"
    done
done
