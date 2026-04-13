#!/usr/bin/env bash
set -euo pipefail

BASE_DATA_DIR="./test_data"
GEN_DIR="veo3.1"
DO_CLEAN="${1:-0}" # default to not clean, set to "1" to clean first

if [[ "$DO_CLEAN" == "1" ]]; then
  phyavbench clean --base-data-dir "$BASE_DATA_DIR" --gen-dirs "$GEN_DIR"
fi

phyavbench extract \
  --video-dir "$BASE_DATA_DIR/$GEN_DIR/video" \
  --audio-output-dir "$BASE_DATA_DIR/$GEN_DIR/audio" \
  --embedding-output-dir "$BASE_DATA_DIR/$GEN_DIR/audio_embedding" \
  --model all

phyavbench score \
  "$BASE_DATA_DIR/$GEN_DIR/audio_embedding" \
  ./embeddings/gt_a2b \
  --output-dir "./output/$GEN_DIR" \
  --report-name "cprs.md"
