#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

CSV="${CSV:-harmbench_behaviors_text_test.csv}"
MODEL="${MODEL:-qwen/qwen3-vl-235b-a22b-instruct}"
BATCH_ROOT="${BATCH_ROOT:-outputs/batch_harmbench}"
SEED="${SEED:-0}"
DISTRACTORS="${DISTRACTORS:--1}"
RUNS="${RUNS:-5}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-10}"
DECODE_OUT="${DECODE_OUT:-decode_accuracy.json}"
PLOTS_DIR="${PLOTS_DIR:-plots}"
TILES_DIR="${TILES_DIR:-assets/object_tiles}"

python batch_generate.py --csv "$CSV" --output-root "$BATCH_ROOT" --seed "$SEED" --distractor-count "$DISTRACTORS" \
  --glyph-mode images --glyph-image-dir "$TILES_DIR"
python vlm_validator.py --csv "$CSV" --all-slugs --batch-root "${BATCH_ROOT}_img" \
  --model "$MODEL" --temperature "$TEMP" --runs "$RUNS" --save-output --concurrency "$CONCURRENCY"
python judge_responses.py --csv "$CSV" --batch-root "${BATCH_ROOT}_img" --concurrency "$CONCURRENCY"
python compute_decoding_accuracy.py --batch-root "${BATCH_ROOT}_img" --output-file "$DECODE_OUT"
python plot_judge_results.py --batch-root "${BATCH_ROOT}_img" --output-dir "$PLOTS_DIR"
