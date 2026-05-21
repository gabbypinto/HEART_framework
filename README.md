# HEART

HEART is a research codebase for iterative reasoning experiments. It supports
multi-turn answer generation, ground-truth judging, filtering incorrect rows for
later iterations, GPQA-style rule-based judging, and LiveCodeBench runs through
API or local vLLM backends.

This repository is packaged so the core code can be installed and run locally.
Generated outputs, cluster logs, private datasets, caches, and exploratory
experiments are intentionally not included.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For API-backed experiments:

```bash
pip install -e ".[api]"
```

For LiveCodeBench dataset loading:

```bash
pip install -e ".[livecode]"
```

For local GPU/vLLM runs:

```bash
pip install -e ".[vllm]"
```

## Environment

Copy `.env.example` to `.env` and fill in only the providers you plan to use.
Credentials are read from environment variables such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, and `HF_TOKEN`.

Path defaults are project-relative:

```bash
HEART_HOME=
HEART_DATA_DIR=
HEART_RESULTS_DIR=
HEART_LOG_DIR=
```

If unset, HEART uses `examples/data`, `results`, and `logs` inside the
checkout.

## Quickstart

Run the no-network smoke test:

```bash
heart-smoke
```

This reads `examples/data/gpqa_sample_predictions.csv`, extracts answer letters
from `clean_model_response_it_0`, applies rule-based GPQA judging, and writes a
judged CSV under `results/smoke`.

Check the main CLIs:

```bash
heart-run --help
heart-lcb --help
```

## HEART Pipeline

`heart-run` is the main experiment CLI. The common flow is:

```bash
heart-run \
  --id gpqa_demo \
  --benchmark GPQA \
  --subject science \
  --model gemini-3-flash \
  --prompt_strategy baseline \
  --dataset validation \
  --stage generate \
  --start_it 0 \
  --input examples/data/gpqa_sample_predictions.csv \
  --run-parallel \
  --num-chunks 1
```

For rule-based judging on GPQA-style model outputs:

```bash
heart-run \
  --id gpqa_demo \
  --benchmark GPQA \
  --subject science \
  --model gemini-3-flash \
  --prompt_strategy baseline \
  --dataset validation \
  --stage gt_judging \
  --start_it 0 \
  --input path/to/raw_output.parquet \
  --run-parallel \
  --rule_based_judging \
  --num-chunks 1
```

API calls require the matching credentials and provider configuration in the
environment. Local model stages require `.[vllm]` and a GPU environment with the
model available from Hugging Face or a local path.

## LiveCodeBench

LiveCodeBench runs from a JSON config:

```bash
heart-lcb --config examples/configs/lcb_local_example.json
```

The example is intentionally tiny and configured for a local vLLM backend. For
API backends, set `"backend": "api"` and configure the provider credentials.

## Development

Run the local checks:

```bash
python -m py_compile $(find src/heart -name '*.py')
pytest
```

The publishable tree should stay small and free of generated artifacts. Keep
datasets as small synthetic examples under `examples/data`; store real outputs
under `results`, which is ignored by Git.
