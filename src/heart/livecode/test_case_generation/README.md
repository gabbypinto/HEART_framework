# Test Case Generation Pipeline

This directory contains the code used to generate and verify the custom LLM-based test cases
stored in `../test_cases/`.  These test cases are used by HEART's Mode 1 (`llm_test_cases`)
as a fast proxy for the real LCB private test cases during iterative refinement.

---

## Output files (in `../test_cases/`)

| File | Difficulty | Problems | Generator model |
|---|---|---|---|
| `HARD_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json` | hard | 350 | gemini-3-pro-preview |
| `HARD_GEMINI-2.5-PRO_gen_verified_test_cases.json` | hard | 350 | gemini-2.5-pro |
| `MEDIUM_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json` | medium | 383 | gemini-3-pro-preview |
| `MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json` | medium+hard | 25 (18 med + 7 hard) | gemini-2.5-pro |

All files share the same JSON format:

```json
[
  {
    "question_id": "...",
    "question_title": "...",
    "question_content": "...",
    "difficulty": "hard",
    "verified_test_cases": [
      {
        "input": "...",
        "expected_output": "...",
        "verification_status": "success",
        "is_valid": true,
        "verified_source": "llm_judge"
      }
    ]
  }
]
```

HEART's `_load_custom_test_cases()` loader reads these files and filters to `is_valid=True`
entries automatically.  When using `MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json`,
set `"custom_test_cases_difficulty_filter": "medium"` (or `"hard"`) in your experiment config
to restrict to a single difficulty.

---

## Generation pipeline

Test case generation runs in two stages:

### Stage 1 — Generate candidate test cases (`run_case_gen.sh`)

Uses `lcb_runner.runner.main` with `--generate_test_cases_output_file` to prompt an LLM
to produce input/output pairs for each coding problem.

```bash
python -m lcb_runner.runner.main \
    --model gemini-2.5-pro \
    --scenario codegeneration \
    --codegen_n 1 \
    --multiprocess 4 \
    --timeout 20 \
    --temperature 0.2 \
    --evaluate \
    --difficulty medium \
    --generate_test_cases_output_file "medium_livecodebench_test_cases.json"
```

Output: a JSON list of `{question_id, generated_test_cases_parsed: [{input, expected_output}]}`.

### Stage 2 — Verify test cases (`verify_tests.py`)

Sends each generated input/output pair back to an LLM judge (via Vertex AI) which checks
whether the expected output is correct.  Produces a JSON with `is_valid: true/false` per
test case.

```bash
python -m lcb_runner.runner.verify_tests \
    --model_name gemini-2.5-pro \
    --project_id "YOUR-PROJECT" \
    --location "us-central1" \
    --input_file medium_livecodebench_test_cases.json \
    --output_file MEDIUM_GEMINI-2.5-PRO_gen_verified_test_cases.json \
    --max_workers 20
```

### Iterative pipeline (`run_iterative_gen.py`)

For large-scale generation that keeps retrying unsolved problems across multiple iterations
until all problems have at least one valid test case.  Merges results into a single master
file (e.g. `MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json`).

```bash
python -m lcb_runner.runner.run_iterative_gen \
    --model gemini-2.5-pro \
    --verifier_model gemini-2.5-pro \
    --project_id "YOUR-PROJECT" \
    --location "us-central1" \
    --difficulty "medium,hard" \
    --max_iterations 10 \
    --multiprocess 15
```

The script:
1. Discovers all problem IDs for the target difficulty from the LCB dataset.
2. Checks which ones already have valid test cases in the master file.
3. Runs generation → verification → merge in chunks of 5 problems.
4. Repeats until all problems are solved or `max_iterations` is reached.

---

## Using these test cases in HEART experiments

In your HEART experiment config, set:

```json
{
  "custom_test_cases_path": "code/livecode/test_cases/HARD_GEMINI-2.5-PRO_gen_verified_test_cases.json",
  "difficulty": "hard"
}
```

For the MASTER file (mixed difficulty):

```json
{
  "custom_test_cases_path": "code/livecode/test_cases/MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json",
  "custom_test_cases_difficulty_filter": "medium",
  "difficulty": "medium"
}
```

See `../README.md` for the full Mode 1 / Mode 2 experiment workflow.
