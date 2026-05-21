"""No-network smoke checks for a publishable HEART install."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from heart import config

ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|-)?\s*\(?\b([ABCD])\b\)?",
    re.IGNORECASE,
)


def extract_gpqa_answer_letter(text: str | None) -> str:
    """Extract a GPQA multiple-choice answer letter from a model response."""
    if not text:
        return ""

    matches = ANSWER_RE.findall(str(text))
    if matches:
        return matches[-1].upper()

    stripped = str(text).strip().upper()
    return stripped if stripped in {"A", "B", "C", "D"} else ""


def judge_gpqa_rows(rows: list[dict[str, str]], response_col: str) -> list[dict[str, str]]:
    judged: list[dict[str, str]] = []
    for row in rows:
        expected = str(row.get("shuffled_correct_key", "")).strip().upper()
        predicted = extract_gpqa_answer_letter(row.get(response_col, ""))
        judged_row = dict(row)
        judged_row["predicted_letter"] = predicted
        judged_row["rule_based_correct"] = str(bool(predicted and predicted == expected))
        judged.append(judged_row)
    return judged


def run_smoke(input_file: Path, output_dir: Path, response_col: str) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Sample input not found: {input_file}")

    with input_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Sample input has no rows: {input_file}")

    judged = judge_gpqa_rows(rows, response_col)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gpqa_smoke_judged.csv"

    fieldnames = list(judged[0].keys())
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(judged)

    correct = sum(row["rule_based_correct"] == "True" for row in judged)
    print(f"HEART smoke: judged {len(judged)} rows, {correct} correct.")
    print(f"Wrote {output_file}")
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a no-network GPQA rule-based judging smoke test."
    )
    parser.add_argument(
        "--input",
        default=str(Path(config.DATA_DIR) / "gpqa_sample_predictions.csv"),
        help="CSV file with GPQA-style columns.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(config.BASE_RESULTS_DIR) / "smoke"),
        help="Directory for the judged smoke output.",
    )
    parser.add_argument(
        "--response-col",
        default="clean_model_response_it_0",
        help="Column containing model responses to judge.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_smoke(Path(args.input), Path(args.output_dir), args.response_col)


if __name__ == "__main__":
    main()
