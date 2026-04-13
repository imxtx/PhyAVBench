#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_DATA_DIR="./test_data"
GEN_DIRS=( # put generated mp4 files in the "video" subdir of each directory below
  "javisdit"
  "javisdit++"
  "klingv2-6"
  "ltx"
  "ovi"
  "seedance-1-5-pro"
  "sora2"
  "veo3.1"
  "wan2.6"
)

GT_DIR="./embeddings/gt_a2b"
DO_CLEAN="${1:-0}" # default to not clean, set to "1" to clean first

if [[ ! -d "$GT_DIR" ]]; then
  echo "[error] ground-truth directory not found: $GT_DIR" >&2
  exit 1
fi

if [[ "$DO_CLEAN" == "1" ]]; then
  echo "[run] cleaning model artifacts first"
  phyavbench clean \
    --base-data-dir "$BASE_DATA_DIR" \
    --gen-dirs "${GEN_DIRS[@]}"
fi

echo "[run] batch scoring across models: ${GEN_DIRS[*]}"

BATCH_SCORE_ARGS=(
  --base-data-dir "$BASE_DATA_DIR"
  --gen-dirs "${GEN_DIRS[@]}"
  --ground-truth-embedding-dir "$GT_DIR"
  --output-dir "$REPO_ROOT/output"
  --report-name "cprs_result.md"
  --model all
)

phyavbench batch-score "${BATCH_SCORE_ARGS[@]}"

echo "[done] merged report written: $REPO_ROOT/output/cprs_result.md"
