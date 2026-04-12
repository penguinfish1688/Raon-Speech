#!/bin/bash
# Run the RAON full-duplex Gradio demo shell.
#
# Edit the preset variables below, then run:
#   bash demo/run_gradio_duplex_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PYTHON_BIN="python3"
HOST="0.0.0.0"
PORT="7861"
SHARE="false"
API_BASE=""
WS_PATH="/ws"
MODEL_PATH="${REPO_DIR}/output/sglang-bundle"
RESULT_ROOT="${REPO_DIR}/output/fd-gradio-demo"
SPEAKER_AUDIO=""
COMPILE_AUDIO_MODULES="true"
COMPILE_MAX_SEQUENCE_LENGTH="8192"

cd "$REPO_DIR"
ARGS=(
    demo/gradio_duplex_demo.py
    --host "$HOST"
    --port "$PORT"
    --ws-path "$WS_PATH"
    --model-path "$MODEL_PATH"
    --result-root "$RESULT_ROOT"
    --compile-audio-modules "$COMPILE_AUDIO_MODULES"
    --compile-max-sequence-length "$COMPILE_MAX_SEQUENCE_LENGTH"
)

if [ "$SHARE" = "true" ]; then
    ARGS+=(--share)
fi
if [ -n "$API_BASE" ]; then
    ARGS+=(--api-base "$API_BASE")
fi
if [ -n "$SPEAKER_AUDIO" ]; then
    ARGS+=(--speaker-audio "$SPEAKER_AUDIO")
fi

exec "$PYTHON_BIN" "${ARGS[@]}"
