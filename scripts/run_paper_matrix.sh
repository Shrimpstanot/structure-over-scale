#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for RUN_ID in \
    lora-r1 lora-r4 lora-r8 lora-r16 lora-r32 lora-r64 lora-r128 lora-r256 \
    dora-r8 dora-r16 dora-r32 dora-r64 dora-r16-no-kv \
    quanta-d3 quanta-d3-no-v quanta-d3-no-kv \
    quanta-d4 quanta-d4-no-kv fft
do
    "$SCRIPT_DIR/run_grpo.sh" "$RUN_ID" 4 42 64
done
