# livecode/prompts.py
#
# Builds (system_instruction, user_prompt) tuples that HEART's ApiClient
# expects — one pair per CodeGenerationProblem per turn.
#
# Turn 1: plain code-generation prompt.
# Turn 2: same structure with a reflection/repair block injected before the
#         final "### Answer:" delimiter, plus the Turn-1 code as context.
#
# The prompt style is "generic" (markdown fenced blocks), matching LCB's
# get_geminithinking_question_template_answer / get_generic_question_template_answer.
# No LMStyle enum is needed because HEART's ApiClient abstracts model details.

from __future__ import annotations

from heart.livecode.lcb_runner.benchmarks.code_generation import CodeGenerationProblem

# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are an expert Python programmer. "
    "You will be given a question (problem specification) and will generate "
    "a correct Python program that matches the specification and passes all tests."
)

_FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)

_FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). "
    "Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm "
    "and writes output to STDOUT."
)


# ---------------------------------------------------------------------------
# Turn 1 — initial generation
# ---------------------------------------------------------------------------

def build_turn1_prompt(problem: CodeGenerationProblem) -> tuple[str, str]:
    """
    Return (system_instruction, user_prompt) for Turn 1 (initial code generation).
    """
    user_prompt = _question_body(problem)
    user_prompt += "### Answer: (use the provided format with backticks)\n\n"
    return SYSTEM_INSTRUCTION, user_prompt


# ---------------------------------------------------------------------------
# Turn 2 — reflection / repair
# ---------------------------------------------------------------------------

def build_turn2_prompt(
    problem: CodeGenerationProblem,
    previous_code: str,
    strategy_suffix: str,
) -> tuple[str, str]:
    """
    Return (system_instruction, user_prompt) for Turn 2 (refinement).

    The Turn-1 code is presented first, followed by the correction instruction
    (strategy_suffix), then the standard answer delimiter.
    """
    body = _question_body(problem)

    reflection_block = (
        "\n\n--- REFLECTION AND REPAIR (Turn 2) ---\n"
        "### Faulty Previous Attempt (Turn 1 Code):\n"
        "```python\n"
        f"{previous_code}\n"
        "```\n\n"
        "### Correction Instruction:\n"
        "Your previous code generation attempt was **INCORRECT**.\n"
        f"{strategy_suffix.strip()}\n"
        "Carefully analyze the faulty attempt and the original problem. "
        "Your goal is to generate a **fully corrected and complete** solution "
        "in the required format.\n"
        "--- END REFLECTION ---\n"
    )

    user_prompt = body + reflection_block
    user_prompt += "\n\n### Answer: (use the provided format with backticks)\n\n"
    return SYSTEM_INSTRUCTION, user_prompt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _question_body(problem: CodeGenerationProblem) -> str:
    """Build the question + format section (shared by Turn 1 and Turn 2)."""
    body = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        body += f"### Format: {_FORMATTING_WITH_STARTER}\n"
        body += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        body += f"### Format: {_FORMATTING_WITHOUT_STARTER}\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    return body
