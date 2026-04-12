#!/bin/bash
# Export a RAON HuggingFace checkpoint to SGLang bundle format.
#
# Edit the preset variables below, then run:
#   bash scripts/export.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_PATH="/home/chang168/personaplex/Raon-Speech/checkpoints"
OUTPUT_PATH="${REPO_DIR}/output/sglang-bundle"
DTYPE="bfloat16"
EXTRA_ARGS=()

echo "=== RAON HF to SGLang Export ==="
echo "Input:      ${INPUT_PATH}"
echo "Output:     ${OUTPUT_PATH}"
echo "Dtype:      ${DTYPE}"
echo "================================"

exec python -m raon.export \
    --input_path "${INPUT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --dtype "${DTYPE}" \
    "${EXTRA_ARGS[@]}"
