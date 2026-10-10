#!/bin/sh
set -eu

# Optional source reconstruction: completion batch changes with G to retain
# eight prompt groups per update. This is not a paper-controlled G-only sweep.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SELECTION=${1:-both}

case "$SELECTION" in
    g4) GROUP_SIZES=4 ;;
    g16) GROUP_SIZES=16 ;;
    both) GROUP_SIZES="4 16" ;;
    *)
        printf 'Usage: %s [g4|g16|both]\n' "$0" >&2
        exit 2
        ;;
esac

for GROUP_SIZE in $GROUP_SIZES
do
    case "$GROUP_SIZE" in
        4) PAPER_BATCH_SIZE=32 ;;
        16) PAPER_BATCH_SIZE=128 ;;
    esac

    for RUN_ID in lora-r8 lora-r256 quanta-d3 fft
    do
        "$SCRIPT_DIR/run_grpo.sh" \
            "$RUN_ID" 4 42 "$PAPER_BATCH_SIZE" "$GROUP_SIZE"
    done
done
