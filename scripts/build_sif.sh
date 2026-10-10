#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
OUTPUT=${1:-$REPO_DIR/containers/structure-over-scale-cu128.sif}
OUTPUT_DIR=$(dirname -- "$OUTPUT")
BUILD_UID=$(id -u)
BUILD_GID=$(id -g)

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"
sudo -n singularity build --force "$OUTPUT" containers/structure-over-scale.def
sudo -n chown "$BUILD_UID:$BUILD_GID" "$OUTPUT"
