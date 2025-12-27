# Cross Modal Cipher (CMC) Attack

Goal: test whether a vision-language model can learn a new pictographic code on the fly. You provide a legend (glyph -> word) and a sentence image; the model should decode the words and/or follow the decoded instruction.

## Quick start (toy example)
1) Generate a legend + sentence:
   ```bash
   python visual_language_experiment.py --output-dir outputs --sentence "cat near tree under sun"
   ```
   Outputs: `legend.png`, `sentence.png`, `codebook.json`, `sentence.json`.

2) Set your OpenRouter key:
   ```bash
   export OPENROUTER_API_KEY="<your-key>"
   ```

3) Ask a VLM to decode + follow instructions:
   ```bash
   python vlm_validator.py --legend outputs/legend.png --sentence outputs/sentence.png \
     --codebook outputs/codebook.json --expected outputs/sentence.json \
     --model qwen/qwen3-vl-32b-instruct --runs 5 --save-output
   ```
   Produces `vlm_reply_<model>.json` containing one decode run plus N instruct runs.

## Batch generation from CSV
Creates a legend/sentence pair per row in a CSV (only `FunctionalCategory=standard`).
```bash
python batch_generate.py --csv sample_behaviors.csv --output-root outputs/batch
```
Each row is written to `outputs/batch/<slug>/` with `legend.png`, `sentence.png`, `codebook.json`, and `metadata.json`.

## VLM validation (batch)
Decode runs once; instruct runs `--runs` times. Useful flags: `--all-slugs`, `--skip-existing`, `--concurrency`.
```bash
python vlm_validator.py --csv sample_behaviors.csv --all-slugs --batch-root outputs/batch \
  --model qwen/qwen3-vl-32b-instruct --runs 5 --save-output --concurrency 10
```

## Judge + plots
Judge each VLM reply file, then plot aggregate results.
```bash
python judge_responses.py --csv sample_behaviors.csv --batch-root outputs/batch --concurrency 10
python plot_judge_results.py --batch-root outputs/batch --output-dir plots
```

## Decoding accuracy
```bash
python compute_decoding_accuracy.py --batch-root outputs/batch --output-file decode_accuracy.json
```
Writes a summary JSON plus two plots (exact match + token overlap).

## Full HarmBench run (example)
```bash
export OPENROUTER_API_KEY="<your-key>"
python batch_generate.py --csv harmbench_behaviors_text_test.csv --output-root outputs/batch_harmbench --seed 0 --distractor-count -1
python vlm_validator.py --csv harmbench_behaviors_text_test.csv --all-slugs --batch-root outputs/batch_harmbench \
  --model qwen/qwen3-vl-235b-a22b-instruct --temperature 0.5 --runs 5 --save-output --concurrency 10
python judge_responses.py --csv harmbench_behaviors_text_test.csv --batch-root outputs/batch_harmbench --concurrency 10
python compute_decoding_accuracy.py --batch-root outputs/batch_harmbench --output-file decode_accuracy.json
python plot_judge_results.py --batch-root outputs/batch_harmbench --output-dir plots
```
Repeat `vlm_validator.py` for additional VLMs if needed.

## Notes
- The toy codebook in `visual_language_experiment.py` is intentionally safe; `clean_concepts_new.json` is not used.
- OpenRouter is required for VLM and judge calls.
- VLM calls use high `max_tokens` and disable "thinking" mode for consistency.
