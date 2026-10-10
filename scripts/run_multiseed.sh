#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for SEED in 42 43 44
do
    for RUN_ID in \
        dora-r16 dora-r16-no-kv \
        quanta-d3 quanta-d3-no-v quanta-d3-no-kv fft
    do
        "$SCRIPT_DIR/run_grpo.sh" "$RUN_ID" 4 "$SEED" 64
    done
done
