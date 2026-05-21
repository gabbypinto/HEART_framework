# livecode/judge.py
#
# Code extraction and test-case execution for LiveCodeBench problems.
#
# Public API:
#   extract_code(model_output)                          -> str
#   evaluate_code(problem, code, timeout)               -> EvalResult
#   grade_outputs(problem, code_list, timeout)          -> list[bool]
#   grade_outputs_custom(problem, code_list, custom_tests, timeout) -> list[bool]
#   compute_pass_at_1(graded_list)                      -> float
#   compute_pass_at_k(graded_list, k)                   -> float

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from heart.livecode.lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from heart.livecode.lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of running one code submission against all test cases."""
    passed: bool
    per_test_results: list
    metadata: dict = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(model_output: str) -> str:
    """
    Extract the last fenced Python code block from a model response.

    Returns "" if no complete fenced block is found.
    """
    if not model_output:
        return ""

    lines = model_output.split("\n")
    fence_indices = [i for i, ln in enumerate(lines) if ln.strip().startswith("```")]

    if len(fence_indices) < 2:
        return ""

    start = fence_indices[-2]
    end   = fence_indices[-1]
    return "\n".join(lines[start + 1 : end]).strip()


# ---------------------------------------------------------------------------
# Single-problem evaluation
# ---------------------------------------------------------------------------

def evaluate_code(
    problem: CodeGenerationProblem,
    code: str,
    timeout: int = 10,
    custom_tests: list | None = None,
) -> EvalResult:
    """
    Run *code* against all test cases for *problem*.

    If *custom_tests* is provided (list of {"input": ..., "output": ...} dicts),
    those are used instead of the real LCB private test cases.  This supports
    Mode 1 evaluation against LLM-generated test cases.

    Uses LCB's check_correctness which forks a subprocess per test run.
    """
    if not code or not code.strip():
        return EvalResult(passed=False, per_test_results=[], error="empty code")

    if custom_tests is not None:
        sample = problem.get_custom_evaluation_sample(custom_tests)
    else:
        sample = problem.get_evaluation_sample()

    try:
        result, metadata = check_correctness(sample, code, timeout=timeout, debug=False)
    except Exception as exc:
        logger.warning("check_correctness raised: %s", exc)
        return EvalResult(passed=False, per_test_results=[], metadata={}, error=str(exc))

    if not isinstance(result, list):
        result = []

    passed = bool(result) and all(
        (r.item(0) if isinstance(r, np.ndarray) else bool(r)) for r in result
    )
    return EvalResult(passed=passed, per_test_results=result, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def grade_outputs(
    problem: CodeGenerationProblem,
    code_list: list[str],
    timeout: int = 10,
) -> list[bool]:
    """Evaluate each candidate against real LCB test cases."""
    return [evaluate_code(problem, code, timeout=timeout).passed for code in code_list]


def grade_outputs_custom(
    problem: CodeGenerationProblem,
    code_list: list[str],
    custom_tests: list,
    timeout: int = 10,
) -> list[bool]:
    """Evaluate each candidate against LLM-generated custom test cases (Mode 1)."""
    return [
        evaluate_code(problem, code, timeout=timeout, custom_tests=custom_tests).passed
        for code in code_list
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_pass_at_1(graded_list: list[bool]) -> float:
    """Pass@1: fraction of candidates that passed."""
    if not graded_list:
        return 0.0
    return sum(graded_list) / len(graded_list)


def compute_pass_at_k(graded_list: list[bool], k: int) -> float:
    """
    Unbiased estimator for pass@k.
    Uses: 1 - C(n-c, k) / C(n, k)
    """
    n = len(graded_list)
    c = sum(graded_list)
    if n == 0 or k > n:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
