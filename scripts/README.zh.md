# 脚本说明

[English](README.md) | 中文

这个目录包含当前公开工作流所需的流水线脚本与辅助工具。

## 数据准备

- `prepare_reference_dataset.py`：根据 task manifest、Markdown 和参考代码构造 reference-answer JSONL。
- `curate_dataset.py`：数据清洗工具，包含两个子命令：
  - `clean-waits`：基于 AST 感知清洗 `self.wait(...)`
  - `replace-image-rows`：将含图片条目替换为未使用的无图候选条目

## 生成

- `generate_code.py`：调用配置好的 LLM，保存清洗后的 Manim 代码和元数据。

## 渲染与审计

- `render_directory.py`：批量渲染目录下的 `.py` 场景文件并输出 `.mp4`。
- `audit_single.py`：对单个 Manim 脚本执行空间审计，并输出分段级报告。
- `audit_batch.py`：`audit_single.py` 的批处理封装，支持断点续跑。

## 指标

- `padvc.py`：PADVC 核心实现；离线模型可通过 `PADVC_HF_CACHE`、`PADVC_ZH_MODEL`、`PADVC_EN_MODEL` 配置。
- `score_padvc.py`：计算逐样本 `PADVC_raw`、`PADVC_center` 和 `uPADVC`。
- `fit_reference_padvc.py`：根据参考视频拟合 PADVC center 参数。
- `score_td.py`：计算逐样本 `TD_raw`、`TD_center` 和 `uTD`。
- `fit_reference_td.py`：根据参考视频拟合 TD center 参数。
- `compute_text_expansion.py`：直接从生成代码估计文本扩展度。

## 工具脚本

- `merge_padvc_shards.py`：合并多个 PADVC shard 输出。
- `error_taxonomy.py`：按 benchmark taxonomy 分类代码生成失败类型。
- `repo_config.py`：仓库级路径与环境变量辅助函数。
- `check_environment.py`：检查 Python 包、外部命令和关键环境变量。
- `run_render_padvc_pipeline.sh`：在已有 audit 结果时，执行 render + PADVC 的便捷脚本。
- `run_evaluation_pipeline.sh`：端到端执行 render、audit、PADVC、TD 和可选的 text expansion。

## 命令示例

### 环境检查

```bash
python scripts/check_environment.py
```

### 构造参考数据

```bash
python scripts/prepare_reference_dataset.py \
  --task-manifest examples/task_manifest.example.json \
  --output-jsonl results/reference_answers.jsonl
```

### 数据清洗

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

### 代码生成

```bash
python scripts/generate_code.py \
  --input-jsonl examples/sample_prompts.jsonl \
  --instruction-field instruction \
  --model your-model-name \
  --workers 2 \
  --temperature 0.7 \
  --output-dir results/example_generation
```

### 渲染与审计

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

### 计算 PADVC 与 TD

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

### 文本扩展度

```bash
python scripts/compute_text_expansion.py \
  --generation-dir results/example_generation \
  --skip-analysis-refresh
```

### 端到端评测

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
