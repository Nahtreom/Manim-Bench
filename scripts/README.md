# Scripts Overview

English | [中文](README.zh.md)

This directory keeps the current public workflow and the utilities it depends on.

## Dataset Preparation

- `prepare_reference_dataset.py`: build reference-answer JSONL files from task manifests and Markdown sources.
- `curate_dataset.py`: dataset cleanup utilities with two subcommands:
  - `clean-waits`: AST-aware cleanup of `self.wait(...)` calls.
  - `replace-image-rows`: replace image-containing rows with unused no-image candidates.

## Generation

- `generate_code.py`: call the configured LLM and save cleaned Manim code outputs plus metadata.

## Rendering and Audit

- `render_directory.py`: render a directory of `.py` scene files into `.mp4` outputs.
- `audit_single.py`: run spatial auditing on one Manim script and emit a segment-level report.
- `audit_batch.py`: batch wrapper for `audit_single.py` with resumable JSON summaries.

## Metrics

- `padvc.py`: core PADVC implementation; set `PADVC_HF_CACHE`, `PADVC_ZH_MODEL`, or `PADVC_EN_MODEL` for offline embedding models.
- `score_padvc.py`: compute per-sample `PADVC_raw`, `PADVC_center`, and `uPADVC`.
- `fit_reference_padvc.py`: fit PADVC center parameters from reference videos.
- `score_td.py`: compute per-sample `TD_raw`, `TD_center`, and `uTD`.
- `fit_reference_td.py`: fit TD center parameters from reference videos.
- `compute_text_expansion.py`: estimate text expansion directly from generated code.

## Utilities

- `merge_padvc_shards.py`: merge PADVC shard outputs into one result set.
- `error_taxonomy.py`: classify code-generation failures into the benchmark taxonomy.
- `repo_config.py`: repository-local path and environment helpers.
- `check_environment.py`: check Python packages, external commands, and key environment variables.
- `run_render_padvc_pipeline.sh`: convenience wrapper for render + PADVC scoring when audit results already exist.
- `run_evaluation_pipeline.sh`: end-to-end wrapper for render, audit, PADVC, TD, and optional text expansion.

## Command Examples

### Check Environment

```bash
python scripts/check_environment.py
```

### Prepare Reference Data

```bash
python scripts/prepare_reference_dataset.py \
  --task-manifest examples/task_manifest.example.json \
  --output-jsonl results/reference_answers.jsonl
```

### Curate Dataset

```bash
python scripts/curate_dataset.py clean-waits \
  --input-jsonl data/train.jsonl \
  --output-dir data/curated

python scripts/curate_dataset.py replace-image-rows \
  --target-jsonl data/train.jsonl \
  --candidate-jsonl easy=data/candidates_easy.jsonl medium=data/candidates_medium.jsonl hard=data/candidates_hard.jsonl \
  --output-jsonl data/train_no_images.jsonl \
  --target-language zh
```

### Generate Code

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```

### Render and Audit

```bash
python scripts/render_directory.py \
  --input-dir results/example_generation/cleaned_scripts \
  --output-dir results/example_eval/videos \
  --results-json results/example_eval/render_results.json \
  --workers 4

python scripts/audit_batch.py \
  --input-dir results/example_generation/cleaned_scripts \
  --output-dir results/example_eval/audit \
  --workers 4 \
  --no-images
```

### Score PADVC and TD

```bash
python scripts/score_padvc.py \
  --task-manifest results/example_generation/task_manifest.json \
  --audit-results results/example_eval/audit/results.json \
  --render-results results/example_eval/render_results.json \
  --sample-jsonl examples/sample_prompts.jsonl \
  --norm-params examples/params/padvc_norm_params.example.json \
  --output-dir results/example_eval/padvc_final \
  --ocr-cache-dir results/example_eval/ocr_cache \
  --quiet-padvc

python scripts/score_td.py \
  --input-jsonl results/example_eval/padvc_final/padvc_scores.jsonl \
  --params-json examples/params/td_center_params.example.json \
  --output-dir results/example_eval/td_final \
  --workers 8
```

### Text Expansion

```bash
python scripts/compute_text_expansion.py \
  --generation-dir results/example_generation \
  --skip-analysis-refresh
```

### End-to-End Evaluation

```bash
scripts/run_evaluation_pipeline.sh \
  results/example_generation/cleaned_scripts \
  results/example_eval_run \
  results/example_generation/task_manifest.json \
  examples/sample_prompts.jsonl \
  examples/params/padvc_norm_params.example.json \
  examples/params/td_center_params.example.json \
  example_model
```
