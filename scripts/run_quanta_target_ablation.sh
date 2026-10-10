#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

for RUN_ID in \
    quanta-d3 quanta-d3-no-v quanta-d3-no-kv \
    quanta-d4 quanta-d4-no-kv
do
    "$SCRIPT_DIR/run_grpo.sh" "$RUN_ID" 4 42 64
done
