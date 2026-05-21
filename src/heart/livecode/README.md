# HEART / LiveCodeBench (`livecode`)

This module integrates the [LiveCodeBench](https://livecodebench.github.io) code-generation benchmark into the HEART multi-turn experiment framework.  It is fully self-contained — all necessary LCB evaluation code is bundled under `livecode/lcb_runner/` and the module does **not** depend on `~/LiveCodeBench` at runtime.

---

## Directory layout

```
HEART/
├── main_lcb.py                         # Entry point (CLI)
├── configs/
│   ├── lcb_gemini_example.json         # Gemini API config examples
│   └── lcb_gemma_example.json          # Gemma 12B / vLLM config examples
├── scripts/
│   ├── run_livecode_test.sh            # Smoke test: Mode 1 → Mode 2
│   ├── run_livecode_gemini_full.sh     # Full Gemini-3-Flash experiment
│   ├── run_livecode_gemma_test.sh      # Gemma-3-12B hard smoke test (vLLM)
│   └── run_livecode_gemma_medium.sh    # Gemma-3-12B medium smoke test (vLLM)
└── code/
    └── livecode/
        ├── README.md                   # This file
        ├── __init__.py
        ├── data_loader.py              # HuggingFace dataset loading & filtering
        ├── prompts.py                  # Turn-1 and Turn-2 prompt builders
        ├── emotion_prompts.py          # All 28 refinement strategies
        ├── judge.py                    # Code extraction and test-case evaluation
        ├── vllm_backend.py             # Local model backend (vLLM)
        ├── experiment_runner.py        # LCBExperimentRunner (Mode 1)
        │                                 LCBFinalSubmissionRunner (Mode 2)
        ├── test_cases/                 # Pre-generated LLM-verified test cases
        │   ├── HARD_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json   (350 hard)
        │   ├── HARD_GEMINI-2.5-PRO_gen_verified_test_cases.json          (350 hard)
        │   ├── MEDIUM_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json  (383 medium)
        │   └── MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json        (18 med + 7 hard)
        ├── test_case_generation/       # Pipeline to reproduce test case files
        │   ├── README.md               # Generation workflow documentation
        │   ├── run_case_gen.sh         # Stage 1: generate candidate test cases
        │   ├── run_case_verification.sh # Stage 2: verify with LLM judge
        │   ├── run_iterative_gen.py    # Full iterative generation loop
        │   ├── verify_tests.py         # Vertex AI LLM verifier
        │   └── read_llm_gen_tests.py   # Inspection / stats utility
        └── lcb_runner/                 # Bundled LCB evaluation code
            ├── lm_styles.py
            ├── benchmarks/
            │   └── code_generation.py  # CodeGenerationProblem dataclass
            ├── evaluation/
            │   ├── testing_util.py     # run_test (subprocess-isolated execution)
            │   ├── pass_k_utils.py     # pass@k estimators
            │   └── compute_code_generation_metrics.py  # check_correctness
            └── utils/
                └── extraction_utils.py
```

---

## Two pipeline modes

### Mode 1 — `llm_test_cases` (generation + iterative refinement)

The main experiment loop.  For each problem the model generates `n_samples` code candidates, evaluates them against test cases, and refines failing ones for up to `n_turns` using an emotion or reflection strategy.

```
Dataset ──► Turn 1 generation ──► Evaluate ──► FINAL_ATTEMPT.json
                                        │
                                  (if n_turns > 1)
                                        │
                               ──► Turn 2 refinement ──► Evaluate
                                        │
                               ──► Turn 3 ... (up to n_turns)
```

**When to use:**
- Running new experiments where code still needs to be generated.
- Iterating on refinement strategies (emotion, COT, reflect).
- Chaining multiple turn files with `refine_from_file`.

**Key config fields:**

| Field | Default | Description |
|---|---|---|
| `mode` | `"llm_test_cases"` | Selects Mode 1 |
| `model` | — | Model name (Gemini API) or HF path (vLLM) |
| `backend` | `"api"` | `"api"` for Gemini/Vertex, `"vllm"` for local models |
| `n_samples` | `1` | Candidate codes generated per problem per turn |
| `n_turns` | `1` | Total turns (1 = baseline only, 2+ = refinement) |
| `refine_strategy` | `"none"` | Strategy used for Turn 2+ prompts (see below) |
| `temperature` | `0.7` | Sampling temperature |
| `timeout` | `10` | Per-test execution timeout in seconds |
| `max_workers` | `8` | Thread pool size (API backend only) |
| `custom_test_cases_path` | `null` | Path to LLM-generated test cases JSON (optional, see below) |
| `refine_from_file` | `null` | Path to a prior `FINAL_ATTEMPT.json` to seed Turn 2 from |
| `synthesize_candidates` | `false` | Merge n_samples codes into one synthesised solution (vLLM only) |

**Output files** (written to `output_dir`):

| File | Description |
|---|---|
| `{id}_turn1_{timestamp}.json` | Raw per-problem results after Turn 1 |
| `{id}_turn2_{timestamp}.json` | Raw per-problem results after Turn 2 (if n_turns ≥ 2) |
| `{id}_{timestamp}.json` | Full results across all turns, all problems |
| `{id}_FINAL_ATTEMPT.json` | **One best code per problem** — feed this into Mode 2 |
| `{id}_summary.json` | Aggregate pass rates and per-difficulty breakdown |

**`FINAL_ATTEMPT.json` format:**
```json
[
  {
    "question_id": "...",
    "question_title": "...",
    "platform": "leetcode",
    "difficulty": "hard",
    "extracted_code": "def solution(...):\n    ...",
    "model_full_response": "...",
    "contest_id": "...",
    "contest_date": "2024-..."
  },
  ...
]
```

**`_summary.json` format:**
```json
{
  "experiment_id": "my_exp",
  "model": "gemini-3-flash",
  "strategy": "disgust_anger_fixed",
  "n_turns": 2,
  "n_samples": 10,
  "total_problems": 150,
  "turn1_pass_count": 82,
  "turn1_pass_rate": 0.5467,
  "turn1_avg_pass_at_1": 0.5467,
  "turn1_hard_pass_rate": 0.5467,
  "combined_pass_count": 95,
  "combined_pass_rate": 0.6333
}
```

---

### Mode 2 — `final_submission` (evaluate saved code against real test cases)

Loads a `FINAL_ATTEMPT.json` written by Mode 1 and runs each problem's extracted code against the **real LCB private test cases** from the HuggingFace dataset.  No LLM is called.

```
FINAL_ATTEMPT.json ──► Load real test cases from HF ──► check_correctness ──► evaluation results
```

**When to use:**
- Getting the official pass rate on real LCB test cases after Mode 1 finishes.
- Comparing different refinement strategies on the same final-submission code.
- Verifying that results from Mode 1 (which may have used custom test cases) hold up on the real suite.

**Key config fields:**

| Field | Default | Description |
|---|---|---|
| `mode` | — | Must be `"final_submission"` |
| `final_attempt_path` | — | Path to `FINAL_ATTEMPT.json` from Mode 1 (required) |
| `release_version` | `"release_v6"` | Must match the dataset used in Mode 1 |
| `difficulty` | — | Must match the difficulty used in Mode 1 |
| `max_problems` | `null` | Must match Mode 1 (leave null for full run) |
| `timeout` | `30` | Per-test timeout; can be higher than Mode 1 |

**Output files** (written to `output_dir`):

| File | Description |
|---|---|
| `{id}_final_eval_{timestamp}.json` | Per-problem `passed` and `pass@1` against real test cases |
| `{id}_final_summary.json` | Aggregate pass rate, per-difficulty breakdown |

**`_final_summary.json` format:**
```json
{
  "experiment_id": "my_exp_final_eval",
  "mode": "final_submission",
  "total_problems": 150,
  "pass_count": 89,
  "pass_rate": 0.5933,
  "hard_pass_rate": 0.5933,
  "final_attempt_path": "results/livecode/my_exp/my_exp_FINAL_ATTEMPT.json"
}
```

---

## Refinement strategies

Set via `refine_strategy` in the config.  Used for Turn 2+ prompts appended after presenting the faulty Turn-1 code.

| Category | Strategies |
|---|---|
| **Baseline** | `none`, `wait`, `cot` |
| **Reflection** | `reflect`, `reflect_1` … `reflect_10` |
| **Emotion pairs (fixed order)** | `disgust_anger_fixed`, `fear_disgust_fixed`, `fear_anger_fixed`, `happy_surprise_fixed`, `sad_fear_fixed`, `sad_anger_fixed`, `sad_disgust_fixed`, `fear_surprise_fixed`, `anger_happy_fixed`, `anger_surprise_fixed`, `fear_happy_fixed` |
| **Emotion pairs (random pool)** | `disgust_anger`, `fear_disgust`, `emotion_all` |

`*_fixed` strategies assign a deterministic prompt per sample index (sample 1 gets prompt 1, sample 2 gets prompt 2, etc.), making results reproducible.  Pool strategies sample randomly.

---

## Backends

### `"api"` — Gemini / Vertex AI (default)
Uses HEART's existing `ApiClient`.  Parallel requests via `ThreadPoolExecutor`.

```json
{
  "backend": "api",
  "model": "gemini-3-flash",
  "max_workers": 8
}
```

### `"vllm"` — Local models via vLLM
Loads the model once and processes all problems in a single batched forward pass, which is much more efficient than one call at a time.

```json
{
  "backend": "vllm",
  "model": "google/gemma-3-12b-it",
  "local_model_path": "google/gemma-3-12b-it",
  "tensor_parallel_size": 8,
  "temperature": 0.5
}
```

Requires the `ttd` conda environment (`/opt/conda/envs/ttd/bin/python`).

---

## Custom test cases (optional, Mode 1)

Instead of evaluating against the real LCB private test cases during Mode 1, you can supply a
JSON file of LLM-verified test cases.  This is useful for fast iteration — custom test cases run
instantly, while the real private test cases require loading the full LCB dataset.

Pre-generated test case files are bundled in `test_cases/`:

| File | Difficulty | Problems |
|---|---|---|
| `HARD_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json` | hard | 350 |
| `HARD_GEMINI-2.5-PRO_gen_verified_test_cases.json` | hard | 350 |
| `MEDIUM_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json` | medium | 383 |
| `MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json` | medium + hard | 25 |

Set `custom_test_cases_path` in the config to point at one of these files (or any compatible file).
The loader filters automatically to `is_valid=True` test cases.

For the `MASTER_*` file (which contains mixed difficulties), also set
`custom_test_cases_difficulty_filter` to load only the relevant subset:

```json
{
  "custom_test_cases_path": "code/livecode/test_cases/HARD_GEMINI-2.5-PRO_gen_verified_test_cases.json",
  "difficulty": "hard"
}
```

```json
{
  "custom_test_cases_path": "code/livecode/test_cases/MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json",
  "custom_test_cases_difficulty_filter": "medium",
  "difficulty": "medium"
}
```

**Primary format** (used by all 4 bundled files):
```json
[
  {
    "question_id": "abc123",
    "difficulty": "hard",
    "verified_test_cases": [
      {
        "input": "5\n1 2 3 4 5",
        "expected_output": "15",
        "verification_status": "success",
        "is_valid": true,
        "verified_source": "llm_judge"
      }
    ]
  }
]
```

The loader also accepts simpler hand-crafted formats (list with `tests`/`test_cases` key, or a
dict keyed by `question_id`).  If a problem's `question_id` is not in the file, the runner falls
back to the real test cases automatically.

See `test_case_generation/README.md` for instructions on reproducing these files from scratch.

---

## Chaining turns with `refine_from_file`

To run Turn 2 as a separate job using the results of a prior Turn-1 run:

```json
{
  "id": "my_exp_t2_cot",
  "mode": "llm_test_cases",
  "n_turns": 2,
  "refine_strategy": "cot",
  "refine_from_file": "results/livecode/my_exp_t1/my_exp_t1_FINAL_ATTEMPT.json"
}
```

The runner loads the Turn-1 codes from the file, marks all problems as needing refinement if they failed, and proceeds directly to Turn 2 generation.  This lets you try different strategies on the same Turn-1 baseline without re-generating.

---

## Cross-strategy tournament (`run_livecode_tournament.sh`)

The tournament script runs multiple refinement strategies over several rounds, using the best-performing strategy from each round as the seed for the next. This exposes every strategy to progressively stronger prior attempts without locking in any one strategy early.

### How it works

```
Baseline (n_samples=1):  generate 1 code per problem → FINAL_ATTEMPT (seed)

Round 1:
  disgust_anger_fixed  (n_samples=10, seeded from baseline)
  fear_disgust_fixed   (n_samples=10, seeded from baseline)
  ...all 11 strategies run...
  → find best strategy by combined_pass_rate
  → write seed: 1 best code per problem (from winner)

Round 2:
  all 11 strategies re-run, now seeded from Round 1 winner
  → find best strategy → write seed

... (repeat for N rounds)

Mode 2: evaluate final seed on real LCB private test cases
```

**Key design points:**
- Baseline generates **1 candidate** per problem (no strategy applied).
- Each strategy turn generates **10 candidates** per problem — all 10 start from the same single seed code.
- After each round the **single best-performing strategy** (by `combined_pass_rate` across all problems) wins; its best code per problem becomes the seed.
- Problems already solved are skipped automatically in subsequent rounds (`turn1_passed=True`).
- vLLM strategies run **sequentially** (one model load holds all GPUs); Gemini API strategies run **in parallel**.

### Quick start

Run from the HEART root directory:

```bash
# DeepSeek-R1-Distill-Qwen-7B — all defaults (hard, 5 rounds, emotion strategies)
bash scripts/run_livecode_tournament.sh \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tp 2

# Gemini-3-Flash — parallel API backend
bash scripts/run_livecode_tournament.sh \
    --backend api \
    --model gemini-3-flash
```

### All flags and defaults

| Flag | Default | Description |
|---|---|---|
| `--backend` | `api` | `api` (Gemini/Vertex) or `vllm` (local) |
| `--model` | `gemini-3-flash` | Model name or HuggingFace path |
| `--tp` | `2` | Tensor parallel GPU count (vLLM only) |
| `--difficulty` | `hard` | `hard` or `medium` |
| `--n-rounds` | `5` | Number of tournament rounds |
| `--n-samples` | `10` | Candidates per strategy turn per problem |
| `--n-samples-t1` | `1` | Candidates for the baseline turn |
| `--n-problems` | `0` (all) | Cap problem count for smoke tests |
| `--strategy-type` | `emotion` | `emotion` \| `reflect` \| `cot` \| `wait` \| `all` |
| `--resume-from` | — | Path to a prior round seed to skip earlier rounds |

### Override examples

```bash
# Medium difficulty, 3 rounds, smoke test on 5 problems
bash scripts/run_livecode_tournament.sh \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tp 2 \
    --difficulty medium \
    --n-rounds 3 \
    --n-problems 5

# Use reflection strategies instead of emotion
bash scripts/run_livecode_tournament.sh \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tp 2 \
    --strategy-type reflect

# Resume from round 2 (rounds 1–2 already finished)
bash scripts/run_livecode_tournament.sh \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tp 2 \
    --resume-from results/livecode/tournament/tourney_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_hard_round2_best.json
```

### Monitoring progress (vLLM)

Since vLLM holds all GPUs, strategy jobs run sequentially. Watch the active log:

```bash
# Tail the currently-running strategy
tail -f results/livecode/lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_disgust_anger_fixed_hard/run.log

# See which round-1 dirs have appeared so far
ls -lth results/livecode/ | grep tourney_r1
```

### Output structure

For `--model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --difficulty hard --n-rounds 5 --strategy-type emotion`:

```
results/livecode/
  tournament/                          ← winning-strategy seeds, one per round
    tourney_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_hard_round1_best.json
    tourney_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_hard_round2_best.json
    tourney_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_hard_round3_best.json
    tourney_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_hard_round4_best.json
    tourney_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_hard_round5_best.json

  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_baseline_hard/
    run.log  *_FINAL_ATTEMPT.json  *_summary.json

  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_disgust_anger_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_fear_disgust_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_fear_anger_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_happy_surprise_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_sad_fear_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_sad_anger_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_sad_disgust_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_fear_surprise_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_anger_happy_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_anger_surprise_fixed_hard/
  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_r1_fear_happy_fixed_hard/
  ...  (same 11 dirs repeated for r2, r3, r4, r5 → 55 strategy dirs total)

  lcb_deepseek_ai_DeepSeek_R1_Distill_Qwen_7B_tourney_hard_final_eval/
    run.log  *_final_summary.json   ← Mode 2 pass rate on real LCB test cases
```

Each strategy directory contains: `run.log`, raw generation files, `*_FINAL_ATTEMPT.json`, and `*_summary.json`.

---

## Running experiments

**Standard (single strategy):**
```
1. Run Mode 1 (baseline, n_turns=1)          → lcb_my_exp_FINAL_ATTEMPT.json
2. Run Mode 1 with refine_from_file           → lcb_my_exp_t2_FINAL_ATTEMPT.json
   (strategy applied to failing problems)
3. Run Mode 2 on the best FINAL_ATTEMPT.json  → lcb_my_exp_final_summary.json
```

**Tournament (cross-strategy, multi-round):**
```
bash scripts/run_livecode_tournament.sh \
    --backend vllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tp 2
→ baseline + 5 rounds × 11 emotion strategies (sequential, vLLM)
→ best strategy each round seeds the next
→ final Mode 2 result in lcb_..._tourney_hard_final_eval/
```

**Using the Python entry point directly:**
```bash
/opt/conda/envs/ttd/bin/python main_lcb.py --config configs/lcb_gemini_example.json

# Single named experiment
/opt/conda/envs/ttd/bin/python main_lcb.py \
    --config configs/lcb_gemini_example.json \
    --exp_id lcb_smoke_test
```
