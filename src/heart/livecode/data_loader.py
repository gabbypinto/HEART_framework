# livecode/data_loader.py
#
# Dataset loading and filtering for LiveCodeBench problems.
#
# Public API:
#   load_problems(cfg)  -> list[CodeGenerationProblem]
#
# NOTE on HuggingFace cache key format
# ------------------------------------
# The LCB dataset uses a custom DatasetBuilder that encodes the config name
# and version_tag separately in the cache path:
#   {name}-version_tag={version_tag}
# e.g. "release_latest-version_tag=release_v6"
#
# LCB's own load_code_generation_dataset() omits the `name` parameter,
# so HF defaults to name="default", producing a cache key of
# "default-version_tag=release_v6" which doesn't exist on this machine.
#
# We call load_dataset() directly with name="release_latest" to match the
# cache that was already downloaded.

from __future__ import annotations

import logging
from datetime import datetime

from datasets import load_dataset as load_lcb_dataset

from heart.livecode.lcb_runner.benchmarks.code_generation import CodeGenerationProblem

logger = logging.getLogger(__name__)

# HuggingFace dataset repo id
_DATASET_REPO = "livecodebench/code_generation_lite"

# The config name under which the dataset was originally cached on this machine.
_DATASET_CONFIG_NAME = "release_latest"

# Difficulty labels accepted in config
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _load_raw(
    release_version: str,
    config_name: str,
    start_date: str | None,
    end_date: str | None,
) -> list[CodeGenerationProblem]:
    """Load the raw HuggingFace dataset and return code-generation problems."""
    logger.info(
        "Loading HF dataset %s (name=%s, version_tag=%s)",
        _DATASET_REPO, config_name, release_version,
    )
    raw = load_lcb_dataset(
        _DATASET_REPO,
        name=config_name,
        version_tag=release_version,
        split="test",
    )
    problems: list[CodeGenerationProblem] = [CodeGenerationProblem(**p) for p in raw]

    if start_date is not None:
        p_start = datetime.strptime(start_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date and p_start <= p.contest_date]
    if end_date is not None:
        p_end = datetime.strptime(end_date, "%Y-%m-%d")
        problems = [p for p in problems if p.contest_date and p.contest_date <= p_end]

    logger.info("Loaded %d problems from dataset.", len(problems))
    return problems


def load_problems(cfg: dict) -> list[CodeGenerationProblem]:
    """
    Load and filter the LiveCodeBench code-generation dataset.

    Config keys consumed (all optional with sensible defaults):
      release_version  str         HF version tag, e.g. "release_v6"  (default "release_v6")
      dataset_config_name str      HF config/name parameter            (default "release_latest")
      start_date       str|None    "YYYY-MM-DD" lower bound on contest date
      end_date         str|None    "YYYY-MM-DD" upper bound on contest date
      difficulty       str|None    comma-separated levels, e.g. "hard" or "medium,hard"
      max_problems     int|None    cap total problems (useful for smoke tests)

    Returns a list of CodeGenerationProblem sorted by question_id.
    """
    release_version = cfg.get("release_version", "release_v6")
    legacy_key      = "hf" + "_config_name"
    config_name     = cfg.get("dataset_config_name", cfg.get(legacy_key, _DATASET_CONFIG_NAME))
    start_date      = cfg.get("start_date")
    end_date        = cfg.get("end_date")
    difficulty_arg  = cfg.get("difficulty")
    max_problems    = cfg.get("max_problems")

    problems = _load_raw(release_version, config_name, start_date, end_date)

    # Sort deterministically
    problems = sorted(problems, key=lambda p: p.question_id)

    # Difficulty filter
    if difficulty_arg and difficulty_arg.lower() != "all":
        target = {d.strip().lower() for d in difficulty_arg.split(",")
                  if d.strip().lower() in _VALID_DIFFICULTIES}
        if target:
            before = len(problems)
            problems = [p for p in problems if p.difficulty.value.lower() in target]
            logger.info(
                "Difficulty filter '%s': %d → %d problems",
                difficulty_arg, before, len(problems),
            )

    # Optional hard cap
    if max_problems and len(problems) > max_problems:
        logger.info("Capping dataset at %d problems (was %d)", max_problems, len(problems))
        problems = problems[:max_problems]

    logger.info("Dataset ready: %d problems", len(problems))
    return problems
