# livecode/lcb_runner/benchmarks/code_generation.py
#
# CodeGenerationProblem and related data classes.
# Copied from LiveCodeBench and trimmed: dataset-loading functions removed;
# all class definitions kept intact.

import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, field


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list
    private_test_cases: list
    metadata: dict
    full_iteration_history: list = field(default_factory=list)

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        if isinstance(self.contest_date, str):
            try:
                self.contest_date = datetime.fromisoformat(self.contest_date)
            except ValueError:
                self.contest_date = None
        else:
            self.contest_date = None

        self.public_test_cases = json.loads(self.public_test_cases)
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)
        except Exception:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))
                    )
                )
            )
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)

    def insert_output(self, output_list: list, code_list: list) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat() if self.contest_date else None,
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list,
        code_list: list,
        graded_list: list,
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list) if graded_list else 0.0
        if self.full_iteration_history:
            for i, is_correct in enumerate(graded_list):
                if i < len(self.full_iteration_history) and isinstance(
                    self.full_iteration_history[i], dict
                ):
                    self.full_iteration_history[i]["evaluation_status"] = bool(is_correct)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self) -> dict:
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }

    def get_custom_evaluation_sample(self, custom_tests: list) -> dict:
        """Build an evaluation sample from caller-supplied test cases.

        custom_tests: list of dicts with keys 'input', 'output', and optionally
        'testtype'.  Useful for evaluating against LLM-generated test cases
        instead of the real private suite (Mode 1 / fast iteration).
        """
        return {
            "input_output": json.dumps(
                {
                    "inputs":  [t.get("input", t.get("inputs", "")) for t in custom_tests],
                    "outputs": [t.get("output", t.get("outputs", "")) for t in custom_tests],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }
