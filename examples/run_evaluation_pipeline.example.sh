#!/usr/bin/env bash
set -euo pipefail

# Example only: replace these paths with your local generation outputs and fitted reference parameters.
export PADVC_HF_CACHE="${PADVC_HF_CACHE:-./.cache/hf/hub}"
export RENDER_WORKERS="${RENDER_WORKERS:-4}"
export AUDIT_WORKERS="${AUDIT_WORKERS:-4}"
export PADVC_SHARDS="${PADVC_SHARDS:-4}"
export TD_WORKERS="${TD_WORKERS:-8}"

scripts/run_evaluation_pipeline.sh   results/example_generation/cleaned_scripts   results/example_eval_run   results/example_generation/task_manifest.json   examples/sample_prompts.jsonl   examples/params/padvc_norm_params.example.json   examples/params/td_center_params.example.json   example_model
