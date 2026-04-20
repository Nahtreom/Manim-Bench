# Data Formats

This repository does not ship the full benchmark dataset or experiment outputs. It only defines the file formats expected by the public pipeline.

## 1. Prompt JSONL

Used by `scripts/generate_code.py` when you already have final prompts.

Each line is a JSON object with at least:

```json
{"id": "sample_0001", "instruction": "Write a Manim scene for ..."}
```

Optional fields such as `language`, `difficulty`, or `source_id` are preserved when possible.

## 2. Task Manifest JSON

Used by `scripts/prepare_reference_dataset.py` and by evaluation scripts that need to map a sample id to its markdown source or reference code.

Example:

```json
[
  {
    "id": "example_0001",
    "md_path": "examples/markdowns/example_0001.md",
    "source_file_path": "examples/reference_code/example_0001.py"
  }
]
```

## 3. Reference JSONL

Produced by `scripts/prepare_reference_dataset.py`.

Each line contains:

- `id`
- `instruction`
- `output`
- `md_path`
- `source_file_path`

This file is typically used to fit reference-center parameters for PADVC and TD.

## 4. Generation Run Layout

`scripts/generate_code.py` writes a directory with the following structure:

- `cleaned_scripts/`: cleaned Manim code, one `.py` per sample
- `raw_outputs/`: raw model responses
- `metadata/`: per-sample metadata including token usage and latency
- `prompt_snapshots/`: final prompts sent to the model
- `markdown_snapshots/`: optional markdown snapshots when prompts are built from markdown files
- `results.json`: run-level summary
- `task_manifest.json`: manifest for downstream rendering and scoring

## 5. Evaluation Run Layout

`scripts/run_evaluation_pipeline.sh` writes:

- `videos/`: rendered `.mp4` files
- `render_results.json`
- `audit/`
- `padvc_shards/`
- `padvc_final/`
- `td_final/`
- `ocr_cache/`
- `logs/`

## 6. What Not to Commit

Do not commit:

- raw private lecture notes
- licensed course material
- full benchmark JSONL files
- full model generations
- rendered benchmark videos
- OCR caches and temporary artifacts
