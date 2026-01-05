#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

CSV="${CSV:-harmbench_behaviors_text_test.csv}"
MODEL="${MODEL:-qwen/qwen3-vl-235b-a22b-instruct}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SEED_ROOT_PREFIX="${SEED_ROOT_PREFIX:-outputs/batch_harmbench_cipher_seed}"
MERGED_ROOT="${MERGED_ROOT:-outputs/batch_harmbench_multicipher}"
DISTRACTORS="${DISTRACTORS:--1}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-10}"
PLOTS_DIR="${PLOTS_DIR:-plots}"

for s in $SEEDS; do
  seed_root="${SEED_ROOT_PREFIX}${s}"
  python batch_generate.py --csv "$CSV" --output-root "$seed_root" --seed "$s" --distractor-count "$DISTRACTORS"
  python vlm_validator.py --csv "$CSV" --all-slugs --batch-root "$seed_root" \
    --model "$MODEL" --temperature "$TEMP" --runs 1 --save-output --concurrency "$CONCURRENCY"
done

merge_glob="${SEED_ROOT_PREFIX}*"
python merge_vlm_runs.py --input-glob "$merge_glob" --output-root "$MERGED_ROOT"
python judge_responses.py --csv "$CSV" --batch-root "$MERGED_ROOT" --concurrency "$CONCURRENCY"
python plot_judge_results.py --batch-root "$MERGED_ROOT" --output-dir "$PLOTS_DIR"
