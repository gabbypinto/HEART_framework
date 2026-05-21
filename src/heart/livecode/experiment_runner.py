# livecode/experiment_runner.py
#
# Two pipeline modes for LiveCodeBench code-generation experiments.
#
# Mode 1 — "llm_test_cases" (generation + iterative refinement)
# -------------------------------------------------------------
#   • Generates n_samples code candidates per problem (Turn 1 baseline).
#   • Evaluates against real LCB test cases OR custom LLM-generated test cases
#     (custom_test_cases_path).
#   • Optionally refines failing problems for up to n_turns using emotion/reflect
#     strategies; each subsequent turn can be seeded from a prior results file
#     (refine_from_file).
#   • After all turns, writes a FINAL_ATTEMPT.json — one best-code entry per
#     problem — suitable as input to Mode 2.
#
# Mode 2 — "final_submission" (evaluate saved results against real test cases)
# ---------------------------------------------------------------------------
#   • Loads a FINAL_ATTEMPT.json produced by Mode 1 (or assembled externally).
#   • Re-loads the original LCB dataset to obtain the real private test cases.
#   • Runs check_correctness on the saved extracted_code per problem.
#   • Writes an evaluation summary + detailed graded results.
#
# Backend
# -------
#   • "api"  (default) — uses HEART's ApiClient (Gemini, Vertex, etc.)
#   • "vllm" — uses VLLMBackend for local models (google/gemma-3-12b-it, etc.)
#
# Config keys (all optional unless noted)
# ----------------------------------------
#   id                   str    experiment id (used in filenames)          [required]
#   model                str    model name / HF path                        [required]
#   backend              str    "api" | "vllm"                             (default "api")
#   mode                 str    "llm_test_cases" | "final_submission"      (default "llm_test_cases")
#
#   --- dataset ---
#   release_version      str    HF version tag                             (default "release_v6")
#   difficulty           str    "easy"|"medium"|"hard"|"medium,hard"|"all" (default "all")
#   start_date           str    "YYYY-MM-DD" lower bound                   (optional)
#   end_date             str    "YYYY-MM-DD" upper bound                   (optional)
#   max_problems         int    cap for smoke tests                        (optional)
#
#   --- generation (Mode 1) ---
#   n_samples            int    candidates per problem per turn            (default 1)
#   n_turns              int    total turns (1 = baseline only)            (default 1)
#   refine_strategy      str    key from REFINE_STRATEGIES                 (default "none")
#   temperature          float  sampling temperature                       (default 0.7)
#   max_tokens           int    max tokens per generation                  (default 2048)
#   timeout              int    test execution timeout (s)                 (default 10)
#   max_workers          int    ThreadPool size for API backend            (default 8)
#   custom_test_cases_path str  JSON file with custom test cases           (optional)
#   refine_from_file     str    prior results JSON to seed turn 2+         (optional)
#   synthesize_candidates bool  merge multi-candidate outputs after T1     (default False)
#   output_dir           str    directory for results / summaries          [required]
#
#   --- vLLM (backend="vllm") ---
#   local_model_path     str    HF id or absolute path                     (default = model)
#   tensor_parallel_size int    GPU count for tensor parallelism           (default 1)
#
#   --- Mode 2 only ---
#   final_attempt_path   str    path to FINAL_ATTEMPT.json                 [required for Mode 2]

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

# livecode sub-modules
from heart.livecode.data_loader import load_problems
from heart.livecode.emotion_prompts import get_refinement_prompts
from heart.livecode.judge import extract_code, grade_outputs, grade_outputs_custom, compute_pass_at_1, compute_pass_at_k
from heart.livecode.prompts import build_turn1_prompt, build_turn2_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def _build_backend(cfg: dict):
    """Return either an ApiClient or VLLMBackend depending on cfg['backend']."""
    backend_type = cfg.get("backend", "api").lower()
    model_name = cfg["model"]

    if backend_type == "vllm":
        from heart.livecode.vllm_backend import VLLMBackend
        local_path = cfg.get("local_model_path", model_name)
        return VLLMBackend(
            model_path=local_path,
            tensor_parallel_size=int(cfg.get("tensor_parallel_size", 1)),
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.95)),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            dtype=cfg.get("dtype", "auto"),
        )
    else:
        from heart.api_client import ApiClient
        return ApiClient(model_name=model_name)


# ---------------------------------------------------------------------------
# Custom test cases loading
# ---------------------------------------------------------------------------

def _load_custom_test_cases(
    path: str,
    difficulty_filter: str | None = None,
) -> dict[str, list]:
    """
    Load LLM-verified or LLM-generated custom test cases from a JSON file.

    Supported formats
    -----------------
    A. Verified test cases format (primary — used by all 4 real LCB files):
       [{"question_id": "...",
         "difficulty": "hard",
         "verified_test_cases": [
           {"input": "...", "expected_output": "...",
            "verification_status": "success", "is_valid": true, ...}
         ]}, ...]
       Only entries with is_valid=True are included.
       Files: HARD_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json
              HARD_GEMINI-2.5-PRO_gen_verified_test_cases.json
              MEDIUM_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json
              MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json  (mixed difficulty)

    B. LCB generated format (legacy):
       [{"question_id": "...",
         "generated_test_cases_parsed": [{"input": ..., "expected_output": ...}],
         "parsing_error": null}, ...]

    C. Simple list format:
       [{"question_id": "...",
         "tests": [{"input": ..., "output": ...}]}, ...]

    D. Dict keyed by question_id:
       {"question_id_1": [{"input": ..., "output": ...}], ...}

    Parameters
    ----------
    difficulty_filter : str or None
        If provided (e.g. "medium" or "hard"), only include entries whose
        "difficulty" field matches.  Useful for MASTER_* files that contain
        mixed difficulties.  Case-insensitive.

    Returns {question_id: [{"input": ..., "output": ...}, ...]}
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        # Format D — already keyed by question_id; normalise output key
        return {qid: _normalise_tests(tests) for qid, tests in raw.items()}

    result: dict[str, list] = {}
    skipped = 0

    for entry in raw:
        qid = entry.get("question_id") or entry.get("id")
        if not qid:
            continue

        # Optional difficulty filter (e.g. extract "medium" from MASTER file)
        if difficulty_filter:
            entry_diff = entry.get("difficulty", "")
            if entry_diff.lower() != difficulty_filter.lower():
                continue

        # Format A — verified test cases (primary format for all 4 real files)
        if "verified_test_cases" in entry:
            all_tests = entry.get("verified_test_cases") or []
            # Only use tests that passed verification
            valid = [t for t in all_tests if t.get("is_valid", True)]
            if not valid:
                skipped += 1
                continue
            result[qid] = _normalise_tests(valid)

        # Format B — LCB generated test cases (legacy)
        elif "generated_test_cases_parsed" in entry:
            tests = entry.get("generated_test_cases_parsed")
            if not tests:
                skipped += 1
                continue
            result[qid] = _normalise_tests(tests)

        else:
            # Format C — simple list format
            tests = (
                entry.get("tests")
                or entry.get("test_cases")
                or entry.get("custom_tests")
                or []
            )
            result[qid] = _normalise_tests(tests)

    if skipped:
        logger.warning(
            "Skipped %d problems with no valid custom test cases in %s",
            skipped, path,
        )
    return result


def _normalise_tests(tests: list) -> list:
    """
    Ensure every test dict has an "input" and "output" key.

    LCB generated files use "expected_output"; we rename it to "output" so
    CodeGenerationProblem.get_custom_evaluation_sample() has a uniform interface.
    """
    normalised = []
    for t in tests:
        if not isinstance(t, dict):
            continue
        normalised.append({
            "input":  t.get("input", ""),
            "output": t.get("output") or t.get("expected_output") or "",
        })
    return normalised


# ===========================================================================
# Mode 1: LCBExperimentRunner
# ===========================================================================

class LCBExperimentRunner:
    """
    Run a full LiveCodeBench code-generation experiment (Mode 1).

    Supports:
    - Turn 1 baseline generation (n_samples candidates per problem)
    - Multi-turn refinement (n_turns > 1) with emotion/reflect strategies
    - Custom LLM-generated test cases for fast evaluation (custom_test_cases_path)
    - Seeding from a prior results file (refine_from_file) to chain turns
    - Candidate synthesis (synthesize_candidates=True, vLLM only)
    - Both API and vLLM backends
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg           = cfg
        self.exp_id        = cfg["id"]
        self.model_name    = cfg["model"]
        self.n_samples     = int(cfg.get("n_samples", 1))
        self.n_turns       = int(cfg.get("n_turns", 1))
        self.strategy      = cfg.get("refine_strategy", "none")
        self.timeout       = int(cfg.get("timeout", 10))
        self.output_dir    = cfg["output_dir"]
        self.max_workers   = int(cfg.get("max_workers", 8))
        self.temperature   = float(cfg.get("temperature", 0.7))
        self.max_tokens    = int(cfg.get("max_tokens", 2048))
        self.synthesize    = bool(cfg.get("synthesize_candidates", False))
        self.refine_from   = cfg.get("refine_from_file")          # optional seed file
        self.custom_tc_path = cfg.get("custom_test_cases_path")   # optional custom tests
        # When loading a mixed-difficulty file (e.g. MASTER_GEMINI-2.5-PRO),
        # set this to "medium" or "hard" to keep only the relevant subset.
        self.custom_tc_difficulty_filter = cfg.get("custom_test_cases_difficulty_filter")

        os.makedirs(self.output_dir, exist_ok=True)

        self.backend = _build_backend(cfg)

        # Custom test cases: {question_id: [test_dict, ...]} or None
        self.custom_tests: dict[str, list] | None = None
        if self.custom_tc_path:
            logger.info("Loading custom test cases from %s", self.custom_tc_path)
            self.custom_tests = _load_custom_test_cases(
                self.custom_tc_path,
                difficulty_filter=self.custom_tc_difficulty_filter,
            )
            logger.info("Custom test cases loaded for %d problems", len(self.custom_tests))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full experiment and return a summary dict."""
        logger.info("=== LCB Experiment [Mode 1]: %s | model: %s ===",
                    self.exp_id, self.model_name)

        # 1. Load dataset
        problems = load_problems(self.cfg)
        logger.info("Loaded %d problems.", len(problems))

        # 2. Either load prior Turn-1 results (refine_from_file) or generate fresh
        if self.refine_from:
            logger.info("Seeding from prior results file: %s", self.refine_from)
            all_records = self._load_records_from_file(self.refine_from, problems)
            # Detect the highest turn already present so we skip re-running
            # completed turns (e.g. resume Turn 3 from a Turn-2 FINAL_ATTEMPT.json).
            _max_loaded = max(
                (
                    t
                    for t in range(1, self.n_turns + 2)
                    if any(
                        r.get(f"turn{t}_codes")
                        for r in all_records.values()
                    )
                ),
                default=1,
            )
            start_turn = _max_loaded + 1
            if _max_loaded > 1:
                logger.info(
                    "Detected loaded turns up to Turn %d — resuming from Turn %d.",
                    _max_loaded, start_turn,
                )
        else:
            logger.info("--- Turn 1: generating %d candidate(s) per problem ---",
                        self.n_samples)
            all_records = self._run_generation_turn(problems, turn=1, prev_results=None)
            all_records = self._evaluate_turn(all_records, problems, turn_key="turn1")
            self._save_turn_results(all_records, problems, turn=1)
            start_turn = 2

        # 3. Optional multi-turn refinement
        for turn_id in range(start_turn, self.n_turns + 1):
            failed = [
                p for p in problems
                if not all_records[p.question_id].get("turn1_passed", False)
                and not any(
                    all_records[p.question_id].get(f"turn{t}_passed", False)
                    for t in range(2, turn_id)
                )
            ]
            logger.info(
                "--- Turn %d: refining %d unsolved problems (strategy=%s) ---",
                turn_id, len(failed), self.strategy,
            )
            if not failed:
                logger.info("All problems solved — stopping refinement early.")
                break

            t_key = f"turn{turn_id}"
            t_records = self._run_generation_turn(
                problems=failed,
                turn=turn_id,
                prev_results=all_records,
            )
            t_records = self._evaluate_turn(t_records, failed, turn_key=t_key)

            # Merge back — copy all turn-keyed fields (codes, outputs, metrics,
            # graded, passed, pass_at_1) into the main records dict.
            # Since _run_generation_turn now uses "turnN_codes" / "turnN_outputs",
            # all relevant keys start with t_key and are captured here.
            for qid, t_data in t_records.items():
                for k, v in t_data.items():
                    if k.startswith(t_key) or k in ("errors",):
                        all_records[qid][k] = v

            self._save_turn_results(all_records, problems, turn=turn_id)

        # 4. Synthesise candidates (vLLM only)
        if self.synthesize and hasattr(self.backend, "synthesize_candidates"):
            logger.info("--- Synthesising candidates ---")
            all_records = self._run_synthesis(all_records, problems)

        # 5. Write FINAL_ATTEMPT.json — one best code per problem
        final_path = self._save_final_attempt(all_records, problems)
        logger.info("FINAL_ATTEMPT written → %s", final_path)

        # 6. Save full results JSON + summary
        self._save_results(all_records, problems)
        summary = self._compute_summary(all_records)
        self._save_summary(summary)
        self._log_summary(summary)

        return summary

    # ------------------------------------------------------------------
    # Generation turn
    # ------------------------------------------------------------------

    def _run_generation_turn(
        self,
        problems: list,
        turn: int,
        prev_results: dict | None,
    ) -> dict:
        """Call the backend for every problem × sample; return records dict."""
        refinement_suffixes: list[str] = []
        if turn >= 2:
            refinement_suffixes = get_refinement_prompts(self.strategy, self.n_samples)

        # Keys for outputs/codes use the actual turn number so Turn-2 codes are
        # stored as "turn2_codes" (not "turn1_codes"), allowing _pick_best_code
        # to find them after they are merged back into all_records.
        codes_key   = f"turn{turn}_codes"
        outputs_key = f"turn{turn}_outputs"
        metrics_key = f"turn{turn}_metrics"

        tasks: list[tuple] = []
        records: dict[str, dict] = {}

        for problem in problems:
            if turn == 1:
                sys_instr, user_prompt = build_turn1_prompt(problem)
                for s_idx in range(self.n_samples):
                    tasks.append((problem, s_idx, sys_instr, user_prompt))
            else:
                prev = prev_results.get(problem.question_id, {})
                # Prefer the most recent prior turn's codes as the "faulty attempt"
                prior_turn = turn - 1
                prev_codes = prev.get(f"turn{prior_turn}_codes", prev.get("turn1_codes", [""]))
                for s_idx in range(self.n_samples):
                    # If T1 had fewer samples than T2 (e.g. baseline n=1, strategy n=10),
                    # repeat the last available code rather than handing an empty string.
                    prev_code = prev_codes[s_idx] if s_idx < len(prev_codes) else (prev_codes[-1] if prev_codes else "")
                    suffix = (
                        refinement_suffixes[s_idx]
                        if s_idx < len(refinement_suffixes)
                        else ""
                    )
                    sys_instr, user_prompt = build_turn2_prompt(problem, prev_code, suffix)
                    tasks.append((problem, s_idx, sys_instr, user_prompt))

        # Initialise skeletons using the correct turn-keyed field names
        for problem in problems:
            records[problem.question_id] = {
                "question_id":    problem.question_id,
                "question_title": problem.question_title,
                "platform":       problem.platform.value,
                "difficulty":     problem.difficulty.value,
                outputs_key:  [""] * self.n_samples,
                codes_key:    [""] * self.n_samples,
                metrics_key:  {},
            }

        backend = self.backend

        # Use batch generation for vLLM (much more efficient)
        if hasattr(backend, "generate_batch"):
            prompt_pairs = [(t[2], t[3]) for t in tasks]
            all_texts = backend.generate_batch(
                prompt_pairs,
                n=1,
                temperature=self.temperature,
            )
            for (problem, s_idx, _, _), text_list in zip(tasks, all_texts):
                text = text_list[0] if text_list else ""
                rec = records[problem.question_id]
                rec[outputs_key][s_idx] = text
                rec[codes_key][s_idx]   = extract_code(text)
        else:
            # API backend: use ThreadPoolExecutor
            def _call(task):
                prob, s_idx, sys_instr, user_prompt = task
                try:
                    text, metrics = backend.generate_response_with_metrics(
                        model_id=self.model_name,
                        system_instruction=sys_instr,
                        prompt=user_prompt,
                    )
                    return prob.question_id, s_idx, text, metrics, None
                except Exception as exc:
                    logger.error(
                        "API call failed for %s sample %d: %s",
                        prob.question_id, s_idx, exc,
                    )
                    return prob.question_id, s_idx, "", {}, str(exc)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(_call, t): t for t in tasks}
                for future in as_completed(futures):
                    qid, s_idx, text, metrics, err = future.result()
                    rec = records[qid]
                    rec[outputs_key][s_idx] = text
                    rec[codes_key][s_idx]   = extract_code(text)
                    if not rec[metrics_key]:
                        rec[metrics_key] = (
                            metrics.to_row_dict()
                            if hasattr(metrics, "to_row_dict")
                            else (metrics if isinstance(metrics, dict) else {})
                        )
                    if err:
                        rec.setdefault("errors", []).append(f"sample_{s_idx}: {err}")

        return records

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_turn(
        self,
        records: dict,
        problems: list,
        turn_key: str = "turn1",
    ) -> dict:
        """Run test-case evaluation for every problem's code candidates."""
        problem_map = {p.question_id: p for p in problems}

        for qid, rec in records.items():
            problem = problem_map.get(qid)
            if problem is None:
                continue

            codes = rec.get(f"{turn_key}_codes", [])

            # Use custom test cases if provided and available for this problem
            if self.custom_tests and qid in self.custom_tests:
                custom = self.custom_tests[qid]
                graded = grade_outputs_custom(problem, codes, custom, timeout=self.timeout)
            else:
                graded = grade_outputs(problem, codes, timeout=self.timeout)

            pass_1 = compute_pass_at_1(graded)

            rec[f"{turn_key}_graded"]    = graded
            rec[f"{turn_key}_passed"]    = any(graded)
            rec[f"{turn_key}_pass_at_1"] = pass_1

            if self.n_samples >= 5:
                rec[f"{turn_key}_pass_at_5"] = compute_pass_at_k(graded, k=5)
            if self.n_samples >= 10:
                rec[f"{turn_key}_pass_at_10"] = compute_pass_at_k(graded, k=10)

        return records

    # ------------------------------------------------------------------
    # Candidate synthesis (vLLM)
    # ------------------------------------------------------------------

    def _run_synthesis(self, records: dict, problems: list) -> dict:
        """Call VLLMBackend.synthesize_candidates on the n_samples codes."""
        by_qid: dict[str, list[str]] = {
            qid: rec.get("turn1_codes", [])
            for qid, rec in records.items()
        }
        synth = self.backend.synthesize_candidates(by_qid)
        problem_map = {p.question_id: p for p in problems}

        for qid, (raw, code) in synth.items():
            rec = records[qid]
            rec["synthesized_output"] = raw
            rec["synthesized_code"]   = code
            problem = problem_map.get(qid)
            if problem and code:
                graded = grade_outputs(problem, [code], timeout=self.timeout)
                rec["synthesized_passed"] = bool(graded[0]) if graded else False

        return records

    # ------------------------------------------------------------------
    # File loading (seed from prior run)
    # ------------------------------------------------------------------

    def _load_records_from_file(self, path: str, problems: list) -> dict:
        """
        Load Turn-1 results from a prior run JSON file.

        Accepts both the full per-turn results JSON and the leaner
        FINAL_ATTEMPT.json produced by _save_final_attempt().  The
        FINAL_ATTEMPT format includes turn1_passed, turn1_pass_at_1, and
        turn1_codes fields added specifically to support correct Turn-2
        chaining (problems that already passed T1 must be skipped in T2).
        """
        with open(path, "r", encoding="utf-8") as fh:
            saved: list[dict] = json.load(fh)

        problem_map = {p.question_id: p for p in problems}
        records: dict[str, dict] = {}

        for entry in saved:
            qid = entry.get("question_id")
            if qid and qid in problem_map:
                # Resolve codes: full list preferred, fall back to single
                # extracted_code stored in FINAL_ATTEMPT.json
                codes = (
                    entry.get("turn1_codes")
                    or entry.get("code_list")
                    or ([entry.get("extracted_code")] if entry.get("extracted_code") else None)
                    or [""]
                )
                rec = {
                    "question_id":    qid,
                    "question_title": entry.get("question_title", ""),
                    "platform":       entry.get("platform", ""),
                    "difficulty":     entry.get("difficulty", ""),
                    "turn1_outputs":  entry.get("turn1_outputs", entry.get("output_list", [])),
                    "turn1_codes":    codes,
                    "turn1_graded":   entry.get("turn1_graded", entry.get("graded_list", [])),
                    # turn1_passed: read from FINAL_ATTEMPT.json (primary) or compute
                    # from legacy formats.  This is the critical field — when True,
                    # the refinement loop will skip this problem in T2+.
                    "turn1_passed":   (
                        entry.get("turn1_passed", False)
                        or entry.get("pass@1", 0) > 0
                        or any(entry.get("graded_list", []))
                        or any(entry.get("turn1_graded", []))
                    ),
                    "turn1_pass_at_1": entry.get("turn1_pass_at_1", entry.get("pass@1", 0.0)),
                    "turn1_metrics":  entry.get("turn1_metrics", {}),
                }
                # If the saved file recorded a latest_turn > 1, populate that
                # turn's codes/graded so _run_generation_turn finds the right prior
                # codes via prev.get("turn{prior_turn}_codes", ...) — without this,
                # resuming at Turn 3 would fall back to turn1_codes (T1 codes) instead
                # of the T2 codes that were used to produce the T2 FINAL_ATTEMPT.
                lt = entry.get("latest_turn", 1)
                if lt > 1:
                    lt_codes  = entry.get("latest_codes") or codes
                    lt_graded = entry.get("latest_graded", [])
                    rec[f"turn{lt}_codes"]  = lt_codes
                    rec[f"turn{lt}_graded"] = lt_graded
                    rec[f"turn{lt}_passed"] = any(lt_graded) if lt_graded else False

                records[qid] = rec

        # Add skeleton entries for problems not present in the file
        for problem in problems:
            if problem.question_id not in records:
                logger.warning(
                    "Problem %s not found in seed file; treating as unsolved.",
                    problem.question_id,
                )
                records[problem.question_id] = {
                    "question_id":    problem.question_id,
                    "question_title": problem.question_title,
                    "platform":       problem.platform.value,
                    "difficulty":     problem.difficulty.value,
                    "turn1_outputs":  [""] * self.n_samples,
                    "turn1_codes":    [""] * self.n_samples,
                    "turn1_graded":   [False] * self.n_samples,
                    "turn1_passed":   False,
                    "turn1_pass_at_1": 0.0,
                    "turn1_metrics":  {},
                }

        return records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_turn_results(self, records: dict, problems: list, turn: int) -> str:
        """Save per-turn intermediate results."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.exp_id}_turn{turn}_{ts}.json"
        path = os.path.join(self.output_dir, filename)
        self._write_json(list(records.values()), path)
        logger.info("Turn %d results → %s", turn, path)
        return path

    def _save_results(self, records: dict, problems: list) -> str:
        """Serialise all per-problem results to a timestamped JSON file."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"{self.exp_id}_{ts}.json")
        problem_map = {p.question_id: p for p in problems}
        output: list[dict] = []
        for qid, rec in records.items():
            entry = dict(rec)
            problem = problem_map.get(qid)
            if problem is not None:
                entry["contest_id"]   = problem.contest_id
                entry["contest_date"] = (
                    problem.contest_date.isoformat() if problem.contest_date else None
                )
                entry["starter_code"] = problem.starter_code
            output.append(entry)
        self._write_json(output, path)
        logger.info("Full results → %s", path)
        return path

    def _save_final_attempt(self, records: dict, problems: list) -> str:
        """
        Write FINAL_ATTEMPT.json — one entry per problem with the best code
        candidate (first passing, else the first candidate).
        """
        path = os.path.join(self.output_dir, f"{self.exp_id}_FINAL_ATTEMPT.json")
        problem_map = {p.question_id: p for p in problems}
        output: list[dict] = []

        for qid, rec in records.items():
            # Prefer synthesized code if available
            if rec.get("synthesized_code"):
                best_code = rec["synthesized_code"]
                best_raw  = rec.get("synthesized_output", best_code)
            else:
                # Walk turns to find the first passing code
                best_code, best_raw = self._pick_best_code(rec)

            problem = problem_map.get(qid)
            # Determine whether this problem was solved in any turn — used by
            # _load_records_from_file when this file is later used as
            # refine_from_file.  A True value here tells the refinement loop
            # to skip this problem rather than sending it to Turn 2+.
            any_turn_passed = any(
                rec.get(f"turn{t}_passed", False)
                for t in range(1, self.n_turns + 1)
            )
            entry: dict[str, Any] = {
                "question_id":    qid,
                "question_title": rec.get("question_title", ""),
                "platform":       rec.get("platform", ""),
                "difficulty":     rec.get("difficulty", ""),
                "extracted_code":    best_code,
                "model_full_response": best_raw,
                # --- Chaining fields (used when this file is refine_from_file) ---
                # turn1_passed: skip this problem in the next Turn 2 run if True
                "turn1_passed":   any_turn_passed,
                "turn1_pass_at_1": rec.get("turn1_pass_at_1", 0.0),
                # turn1_codes: codes for the next turn to refine from.
                # Use the latest completed turn's codes so T3+ refines from T2
                # codes (not the original T1 codes), enabling proper chaining.
                # When T1 had n=1 and T2 has n=10, the 10 T2 codes are preserved.
                "turn1_codes":    next(
                    (rec[f"turn{t}_codes"] for t in range(self.n_turns, 0, -1)
                     if rec.get(f"turn{t}_codes")),
                    [best_code] if best_code else []
                ),
                # turn1_graded: per-sample pass/fail needed by _pick_best_code when
                # this file is loaded as refine_from_file in a chained run — without
                # it the fallback silently picks codes[0] which may be the wrong sample
                "turn1_graded":   rec.get("turn1_graded", []),
            }
            # latest_*: the most-recently completed turn's codes/graded, used when
            # this FINAL_ATTEMPT.json is used as refine_from_file for a Turn N+1 resume.
            # _load_records_from_file reads these to populate turn{latest_turn}_codes so
            # that _run_generation_turn finds the right prior codes via turn{prior_turn}_codes.
            _latest_t = 1
            for _t in range(self.n_turns, 0, -1):
                if rec.get(f"turn{_t}_codes"):
                    _latest_t = _t
                    break
            entry["latest_turn"]   = _latest_t
            entry["latest_codes"]  = rec.get(f"turn{_latest_t}_codes", [])
            entry["latest_graded"] = rec.get(f"turn{_latest_t}_graded", [])
            if problem:
                entry["contest_id"]   = problem.contest_id
                entry["contest_date"] = (
                    problem.contest_date.isoformat() if problem.contest_date else None
                )
            output.append(entry)

        self._write_json(output, path)
        return path

    def _pick_best_code(self, rec: dict) -> tuple[str, str]:
        """Return (extracted_code, raw_response) — first passing candidate.

        Uses index-safe access instead of zip so that a missing or truncated
        outputs list (e.g. when loaded from a chained FINAL_ATTEMPT.json that
        does not store turn1_outputs) does not silently skip all candidates.
        """
        # Pass 1: find the first passing code across all turns
        for turn in range(1, 11):
            codes   = rec.get(f"turn{turn}_codes", [])
            outputs = rec.get(f"turn{turn}_outputs", [])
            graded  = rec.get(f"turn{turn}_graded", [])
            if not codes:
                break
            for i, code in enumerate(codes):
                passed = graded[i] if i < len(graded) else False
                if passed and code:
                    return code, (outputs[i] if i < len(outputs) else "")

        # Pass 2: no passing code found — return first non-empty code from any turn
        for turn in range(1, 11):
            codes   = rec.get(f"turn{turn}_codes", [])
            outputs = rec.get(f"turn{turn}_outputs", [])
            if not codes:
                break
            for i, code in enumerate(codes):
                if code:
                    return code, (outputs[i] if i < len(outputs) else "")

        return "", ""

    def _save_summary(self, summary: dict) -> None:
        path = os.path.join(self.output_dir, f"{self.exp_id}_summary.json")
        self._write_json(summary, path)
        logger.info("Summary → %s", path)

    @staticmethod
    def _write_json(obj: Any, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Summary metrics
    # ------------------------------------------------------------------

    def _compute_summary(self, records: dict) -> dict:
        total = len(records)
        t1_pass = sum(1 for r in records.values() if r.get("turn1_passed", False))

        summary: dict[str, Any] = {
            "experiment_id":    self.exp_id,
            "model":            self.model_name,
            "strategy":         self.strategy,
            "n_turns":          self.n_turns,
            "n_samples":        self.n_samples,
            "total_problems":   total,
            "turn1_pass_count": t1_pass,
            "turn1_pass_rate":  round(t1_pass / total, 4) if total else 0.0,
        }

        all_pass1 = [r.get("turn1_pass_at_1", 0.0) for r in records.values()]
        summary["turn1_avg_pass_at_1"] = (
            round(sum(all_pass1) / len(all_pass1), 4) if all_pass1 else 0.0
        )

        for diff in ("easy", "medium", "hard"):
            diff_recs = [r for r in records.values() if r.get("difficulty") == diff]
            if diff_recs:
                d_pass = sum(1 for r in diff_recs if r.get("turn1_passed", False))
                summary[f"turn1_{diff}_pass_rate"] = round(d_pass / len(diff_recs), 4)

        # Multi-turn combined pass
        combined = sum(
            1 for r in records.values()
            if any(
                r.get(f"turn{t}_passed", False)
                for t in range(1, self.n_turns + 1)
            )
        )
        if self.n_turns > 1:
            summary["combined_pass_count"] = combined
            summary["combined_pass_rate"]  = round(combined / total, 4) if total else 0.0

        return summary

    def _log_summary(self, summary: dict) -> None:
        logger.info("=== RESULTS: %s ===", self.exp_id)
        for k, v in summary.items():
            logger.info("  %-38s %s", k, v)


# ===========================================================================
# Mode 2: LCBFinalSubmissionRunner
# ===========================================================================

class LCBFinalSubmissionRunner:
    """
    Evaluate saved code (FINAL_ATTEMPT.json) against the real LCB test cases.

    This runner does NOT call any LLM — it just loads the code produced by
    Mode 1 and checks it against the real private test cases from the dataset.

    Required config keys
    --------------------
    id                  str   experiment id
    final_attempt_path  str   path to FINAL_ATTEMPT.json from Mode 1
    output_dir          str   where to write evaluation results

    Optional
    --------
    (All dataset filter keys apply: release_version, difficulty, start_date,
     end_date — used to load the matching slice of the real dataset.)
    timeout             int   per-test execution timeout in seconds (default 30)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg              = cfg
        self.exp_id           = cfg["id"]
        self.final_path       = cfg["final_attempt_path"]
        self.output_dir       = cfg["output_dir"]
        self.timeout          = int(cfg.get("timeout", 30))

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> dict:
        """Load final attempts, evaluate against real test cases, save results."""
        logger.info("=== LCB Final Submission [Mode 2]: %s ===", self.exp_id)

        # 1. Load final attempts
        if not os.path.exists(self.final_path):
            raise FileNotFoundError(
                f"final_attempt_path not found: {self.final_path}"
            )
        with open(self.final_path, "r", encoding="utf-8") as fh:
            attempts: list[dict] = json.load(fh)

        code_map: dict[str, dict] = {}
        for rec in attempts:
            qid  = rec.get("question_id")
            code = rec.get("extracted_code", "")
            if qid and code:
                code_map[qid] = {
                    "code":    code,
                    "raw":     rec.get("model_full_response", code),
                    "difficulty": rec.get("difficulty", ""),
                    "question_title": rec.get("question_title", ""),
                }
            else:
                logger.warning("Skipping entry with missing question_id or code.")

        logger.info("Loaded %d coded attempts from %s", len(code_map), self.final_path)

        # 2. Load the real dataset (for test cases)
        problems = load_problems(self.cfg)
        problem_map = {p.question_id: p for p in problems}
        logger.info("Loaded %d real-test-case problems from dataset.", len(problems))

        # 3. Evaluate
        eval_records: list[dict] = []
        pass_count = 0

        for qid, info in code_map.items():
            problem = problem_map.get(qid)
            if problem is None:
                logger.warning("Problem %s not found in real dataset — skipping.", qid)
                continue

            graded = grade_outputs(problem, [info["code"]], timeout=self.timeout)
            passed = bool(graded[0]) if graded else False
            if passed:
                pass_count += 1

            eval_records.append({
                "question_id":    qid,
                "question_title": info.get("question_title", ""),
                "difficulty":     info.get("difficulty", ""),
                "extracted_code": info["code"],
                "passed":         passed,
                "pass@1":         1.0 if passed else 0.0,
            })

        total = len(eval_records)
        pass_rate = round(pass_count / total, 4) if total else 0.0

        # 4. Save results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            self.output_dir, f"{self.exp_id}_final_eval_{ts}.json"
        )
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(eval_records, fh, indent=2, default=str)
        logger.info("Evaluation results → %s", results_path)

        summary: dict[str, Any] = {
            "experiment_id":         self.exp_id,
            "mode":                  "final_submission",
            "total_problems":        total,
            "pass_count":            pass_count,
            "pass_rate":             pass_rate,
            "final_attempt_path":    self.final_path,
        }

        # Per-difficulty breakdown
        for diff in ("easy", "medium", "hard"):
            diff_recs = [r for r in eval_records if r.get("difficulty") == diff]
            if diff_recs:
                d_pass = sum(1 for r in diff_recs if r.get("passed", False))
                summary[f"{diff}_pass_rate"] = round(d_pass / len(diff_recs), 4)

        summary_path = os.path.join(self.output_dir, f"{self.exp_id}_final_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)

        logger.info("=== FINAL SUBMISSION RESULTS: %s ===", self.exp_id)
        for k, v in summary.items():
            logger.info("  %-38s %s", k, v)

        return summary


# ===========================================================================
# Factory
# ===========================================================================

def build_runner(cfg: dict) -> LCBExperimentRunner | LCBFinalSubmissionRunner:
    """Return the appropriate runner based on cfg['mode']."""
    mode = cfg.get("mode", "llm_test_cases").lower()
    if mode == "final_submission":
        return LCBFinalSubmissionRunner(cfg)
    return LCBExperimentRunner(cfg)
