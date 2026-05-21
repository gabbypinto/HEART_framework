#!/usr/bin/env python3
# main_lcb.py  —  LiveCodeBench experiment entry point for HEART
#
# Usage
# -----
#   python main_lcb.py --config configs/lcb_gemini_example.json
#   python main_lcb.py --config configs/lcb_gemma_example.json --exp_id lcb_gemma_t1_cot
#
# Modes
# -----
#   Mode 1 "llm_test_cases"  (default):
#     Generates code candidates, evaluates against LCB test cases (or custom
#     LLM-generated ones if custom_test_cases_path is provided), and iterates
#     with refinement strategies.  Writes a FINAL_ATTEMPT.json.
#
#   Mode 2 "final_submission":
#     Loads a FINAL_ATTEMPT.json produced by Mode 1 and evaluates the saved
#     code against the real LCB private test cases.  No generation happens.
#
# Config format
# -------------
# {
#   "config": [
#     {
#       "id":               "my_exp",
#       "model":            "gemini-3-flash",          // or "google/gemma-3-12b-it"
#       "backend":          "api",                     // "api" | "vllm"
#       "mode":             "llm_test_cases",          // "llm_test_cases" | "final_submission"
#       "release_version":  "release_v6",
#       "difficulty":       "hard",
#       "n_samples":        10,
#       "n_turns":          2,
#       "refine_strategy":  "disgust_anger_fixed",
#       "timeout":          10,
#       "output_dir":       "results/livecode/my_exp",
#       "max_workers":      8,
#       // Mode 1 optional:
#       "custom_test_cases_path": null,
#       "refine_from_file":       null,
#       "synthesize_candidates":  false,
#       // Mode 2 only:
#       "final_attempt_path": "results/livecode/my_exp/my_exp_FINAL_ATTEMPT.json",
#       // vLLM only:
#       "tensor_parallel_size": 8
#     }
#   ]
# }

import argparse
import json
import logging
import os
import sys

from heart import config as heart_config

_PROJECT_ROOT = os.path.abspath(heart_config.BASE_DIR)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main_lcb")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LiveCodeBench code-generation experiments using HEART.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to a HEART-style JSON config file.",
    )
    p.add_argument(
        "--exp_id",
        default=None,
        help="Run only the experiment whose 'id' matches this value.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as fh:
        config_data = json.load(fh)

    experiments: list[dict] = config_data.get("config", [])
    if not experiments:
        logger.error("No experiments found in config file.")
        sys.exit(1)

    if args.exp_id:
        experiments = [e for e in experiments if e.get("id") == args.exp_id]
        if not experiments:
            logger.error("No experiment with id='%s' found.", args.exp_id)
            sys.exit(1)

    from heart.livecode.experiment_runner import build_runner

    logger.info("Running %d experiment(s) from %s", len(experiments), config_path)

    all_summaries: list[dict] = []

    for exp_cfg in experiments:
        exp_id = exp_cfg.get("id", "<unnamed>")
        logger.info("Starting experiment: %s  (mode=%s)", exp_id, exp_cfg.get("mode", "llm_test_cases"))

        # Resolve relative output_dir against the script directory
        if "output_dir" not in exp_cfg:
            exp_cfg["output_dir"] = os.path.join(_PROJECT_ROOT, "results", "livecode", exp_id)
        elif not os.path.isabs(exp_cfg["output_dir"]):
            exp_cfg["output_dir"] = os.path.join(_PROJECT_ROOT, exp_cfg["output_dir"])

        # Resolve relative path keys
        for path_key in ("custom_test_cases_path", "refine_from_file", "final_attempt_path"):
            val = exp_cfg.get(path_key)
            if val and not os.path.isabs(val):
                exp_cfg[path_key] = os.path.join(_PROJECT_ROOT, val)

        try:
            runner  = build_runner(exp_cfg)
            summary = runner.run()
            all_summaries.append(summary)

            mode = exp_cfg.get("mode", "llm_test_cases")
            if mode == "final_submission":
                logger.info(
                    "Experiment '%s' finished.  Pass rate = %.4f",
                    exp_id,
                    summary.get("pass_rate", 0.0),
                )
            else:
                logger.info(
                    "Experiment '%s' finished.  T1 Pass@1 = %.4f  combined = %.4f",
                    exp_id,
                    summary.get("turn1_avg_pass_at_1", 0.0),
                    summary.get("combined_pass_rate", summary.get("turn1_avg_pass_at_1", 0.0)),
                )
        except Exception as exc:
            logger.exception("Experiment '%s' failed: %s", exp_id, exc)

    # Print multi-experiment table
    if len(all_summaries) > 1:
        logger.info("\n--- Multi-experiment summary ---")
        for s in all_summaries:
            mode = s.get("mode", "llm_test_cases")
            if mode == "final_submission":
                logger.info(
                    "  %-40s  pass_rate=%.4f",
                    s.get("experiment_id", "?"),
                    s.get("pass_rate", 0.0),
                )
            else:
                logger.info(
                    "  %-40s  T1_pass@1=%.4f  combined=%.4f",
                    s.get("experiment_id", "?"),
                    s.get("turn1_avg_pass_at_1", 0.0),
                    s.get("combined_pass_rate", s.get("turn1_avg_pass_at_1", 0.0)),
                )


if __name__ == "__main__":
    main()
