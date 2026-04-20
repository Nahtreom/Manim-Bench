# Manim-Bench Toolkit

English | [中文](README.zh.md)

A lightweight toolkit for building and evaluating Manim-code generation benchmarks. It covers dataset curation, LLM generation, rendering, deterministic spatial audit, PADVC/TD scoring, and text-expansion analysis.

This public repository intentionally does **not** include private datasets, full experiment outputs, paper drafts, or benchmark result tables. It only contains reusable code, format documentation, and tiny toy examples.

## Repository Layout

- `scripts/`: command-line tools for generation, rendering, audit, and metrics
- `manim_bench/llm_call/`: minimal LLM client wrapper
- `docs/`: public technical documentation for data formats, metrics, and audit semantics
- `examples/`: small toy inputs and configuration examples
- `data/`: local-only dataset workspace; gitignored except for `data/README.md`
- `results/`: local-only output workspace; gitignored except for `results/README.md`

## What Belongs in Git

Keep:

- reusable pipeline code
- small sanitized examples
- public documentation
- config templates
- environment and dependency instructions

Do not keep:

- private lecture notes or licensed course material
- full benchmark datasets
- model generations or rendered videos
- paper drafts, paper tables, ablation notes, or case-study outputs
- API keys, OCR caches, Hugging Face caches, and temporary artifacts

## System Requirements

Recommended environment:

- Linux or macOS
- Python 3.10+
- Manim Community Edition 0.19.0
- FFmpeg
- Cairo / Pango / `pkg-config` build libraries
- LaTeX toolchain for `Tex` and `MathTex`
- CJK-capable fonts if you render Chinese text

Ubuntu example:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg pkg-config libcairo2-dev libpango1.0-dev \
  texlive texlive-latex-extra texlive-fonts-recommended \
  texlive-xetex dvisvgm ghostscript \
  fonts-noto-cjk fontconfig
```

macOS example:

```bash
brew install ffmpeg cairo pango pkg-config mactex-no-gui font-noto-sans-cjk
```

## Python Installation

Create an isolated environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then check the environment:

```bash
python scripts/check_environment.py
```

## OCR and Small Models

PADVC needs OCR and text-similarity models.

OCR backends:

- default: `paddleocr`
- optional: `rapidocr-onnxruntime` via `PADVC_OCR_BACKEND=rapidocr`

Text-similarity models used by `scripts/padvc.py`:

- Chinese: `shibing624/text2vec-base-chinese`
- English/multilingual: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

The PADVC implementation defaults to offline Hugging Face mode. Prepare model snapshots before running PADVC, then point the scripts to them:

```bash
export PADVC_HF_CACHE=/path/to/huggingface/hub
# or set explicit local snapshot directories
export PADVC_ZH_MODEL=/path/to/text2vec-base-chinese
export PADVC_EN_MODEL=/path/to/paraphrase-multilingual-MiniLM-L12-v2
```

Useful cache variables:

```bash
export PADVC_OCR_CACHE_DIR=results/ocr_cache
export PADVC_DEBUG=0
```

## LLM Configuration

Copy the template and fill in your provider settings:

```bash
cp manim_bench/llm_call/config.example.json manim_bench/llm_call/config.json
```

`config.json` is gitignored. You can also select a different config path with:

```bash
export MANIM_BENCH_LLM_CONFIG=/path/to/config.json
```

## Minimal Workflow

Generate code from prompt JSONL:

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```

Run rendering, spatial audit, PADVC, TD, and text expansion:

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

For stage-by-stage commands, see `scripts/README.md`.

## Reference Parameters

`PADVC_center` and `TD_center` require reference statistics. Fit them on your own curated reference set:

```bash
python scripts/fit_reference_padvc.py \
  --input-jsonl your_reference_scores_or_manifest.jsonl \
  --video-root results/reference_videos \
  --output-dir results/reference_padvc

python scripts/fit_reference_td.py \
  --input-jsonl your_reference_video_manifest.jsonl \
  --output-dir results/reference_td
```

The example parameter files under `examples/params/` are placeholders for smoke tests only.

## Documentation

- `docs/data_format.md`: expected JSON/JSONL layouts
- `docs/spatial_audit.md`: spatial-audit semantics
- `docs/metrics.md`: PADVC, TD, and text-expansion overview
- `docs/code_error_taxonomy.md`: code-failure categories
- `scripts/README.md`: script inventory and command examples
