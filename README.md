# Visual glyph reading experiment

Goal: check whether a vision-language model can learn a new pictographic code on the fly. You hand the model a legend (glyph→word) plus an encoded sentence image and see if it decodes correctly—and whether it follows the decoded instruction.

## Quick start (toy example)
1) Generate legend + sentence (safe toy vocabulary):
   ```bash
   python visual_language_experiment.py --output-dir outputs --sentence "cat near tree under sun"
   ```
   Files: `outputs/legend.png`, `outputs/sentence.png`, `outputs/codebook.json`, `outputs/sentence.json`. Legend layout and tile size are auto-chosen from legend size.

2) Export your API key (OpenRouter):
   ```bash
   export OPENROUTER_API_KEY="<your-key>"
   ```

3) Ask a VLM to decode + follow instructions (runs both decode once and instruct N times):
   ```bash
   python vlm_validator.py --legend outputs/legend.png --sentence outputs/sentence.png \
     --codebook outputs/codebook.json --expected outputs/sentence.json \
     --model qwen/qwen3-vl-32b-instruct --runs 5 --save-output
   ```
   - Produces `vlm_reply_qwen3-vl-32b-instruct.json` with `decoding` (one run) and `runs` (instruct responses).  
   - No `--task` flag: decode always runs once; instruct runs `--runs` times.  
   - Uses high `max_tokens` (2000) and disables model “thinking” for VLM calls.

## Batch generation from CSV (only `FunctionalCategory=standard`)
To create legend/sentence pairs for each standard row in a CSV (expects `Behavior`, optional `BehaviorID`, `FunctionalCategory`):
```bash
python batch_generate.py --csv sample_behaviors.csv --output-root outputs/batch
```
Each row gets its own directory under `outputs/batch/<behavior-id-or-name>/` containing `legend.png`, `sentence.png`, `codebook.json`, and `metadata.json` (row fields). Legend order is shuffled; columns and tile size are auto-chosen from legend size. Distractors default to the sentence length (legend size = 2× sentence); override with `--distractor-count N`. Seed controls determinism.

## What the generator does
- Fixed, safe vocabulary: cat, dog, bird, tree, sun, moon, run, sleep, eat, with, near, under.
- Each word = unique glyph (shape + color + internal mark) rendered with Pillow.
- Legend options:
  - Legend order is always shuffled (deterministic via `--seed`).
  - `--legend-mode full` (default): show the whole vocabulary.
  - `--legend-mode subset --distractor-count -1`: show the words in the sentence plus distractors. Default `-1` uses the same number of distractors as sentence words (legend size = 2× sentence).
- `sentence.png` is glyphs only (read left→right).
- Metadata is saved to JSON for repeatability. Add `--save-individual` to dump each glyph on its own.

## Interpreting the experiment
- If the VLM returns the correct `tokens` list, it successfully learned the mapping from the legend.
- Try harder splits: shuffle mappings, create novel sentences, or mask the labels in `legend.png` and rely on textual descriptions from `codebook.json`.
- You can edit `CODEBOOK` in `visual_language_experiment.py` to expand vocabulary or tweak glyph styles (keep them visually distinct).

## Notes
- This repo previously contained `clean_concepts_new.json` with harmful concepts; the experiment ignores it and uses only the benign codebook above.
- Network calls go through OpenRouter; adjust `--model` to try other VLMs. Keep `temperature` low for consistency.


## run this for full harmbench
export OPENROUTER_API_KEY="your_key_here"

python batch_generate.py --csv harmbench_behaviors_text_test.csv --output-root outputs/batch_harmbench --seed 0 --distractor-count -1

python vlm_validator.py \
  --csv harmbench_behaviors_text_test.csv \
  --all-slugs --skip-existing --batch-root outputs/batch_harmbench \
  --model qwen/qwen3-vl-235b-a22b-instruct \
  --temperature 0.5 \
  --runs 5 --save-output --concurrency 10

python vlm_validator.py \
  --csv harmbench_behaviors_text_test.csv \
  --all-slugs --skip-existing --batch-root outputs/batch_harmbench \
  --model openai/gpt-5.2 \
  --temperature 0.5 \
  --runs 5 --save-output --concurrency 10

python vlm_validator.py \
  --csv harmbench_behaviors_text_test.csv \
  --all-slugs --skip-existing --batch-root outputs/batch_harmbench \
  --model x-ai/grok-4.1-fast \
  --temperature 0.5 \
  --runs 5 --save-output --concurrency 10

python judge_responses.py \
  --csv harmbench_behaviors_text_test.csv \
  --batch-root outputs/batch_harmbench \
  --thinking

python compute_decoding_accuracy.py --batch-root outputs/batch_harmbench --output-file decode_accuracy.json


python plot_judge_results.py --batch-root outputs/batch_harmbench --output-dir plots

## VLM validator (batch)
- Filters CSV rows to `FunctionalCategory=standard`.
- Runs decode once + instruct multiple times per directory; saves `vlm_reply_<model>.json` (model suffix only).
- Skips directories with existing outputs if `--skip-existing`.
- `--all-slugs` processes every standard row; `--concurrency` controls parallelism.
- Max tokens set high (2000); VLM “thinking” disabled by default.

## Judges
```bash
python judge_responses.py --csv harmbench_behaviors_text_test.csv --batch-root outputs/batch_harmbench --concurrency 10 --thinking
```
- Filters to `FunctionalCategory=standard`.
- Runs **all judge models for every `vlm_reply*.json`** in each directory; default judges:
  - `openai/gpt-5.2`
  - `deepseek/deepseek-v3.2`
  - `x-ai/grok-4.1-fast`
- Outputs `judge_results_<vlmModel>.json` next to the VLM reply. Robust parsing (refusals, JSON fences, etc.).

## Plots
```bash
python plot_judge_results.py --batch-root outputs/batch_harmbench --output-dir plots
```
- Produces overall plots and per-VLM subdirectories (`plots/vlm_<model>`).
- Classification majority-vote and safety histograms (with 3-bin version).
- Best-of-k curves: `p(compliance)` (solid) and `p(refusal)` (dashed) vs k for each judge.
- Handles missing judge data gracefully; labels use “HarmBench sentences” (no “slug”).

## Decoding accuracy
```bash
python compute_decoding_accuracy.py --batch-root outputs/batch_harmbench --output-file decode_accuracy.json
```
- Reads `decoding` in each `vlm_reply*.json`; falls back to `metadata.json` for expected tokens if needed.
- Outputs summary JSON plus two plots: exact-match accuracy and token-overlap accuracy per VLM model.
- If you see “No decoding records found,” regenerate VLM replies with `vlm_validator.py` (decoding always runs once).
