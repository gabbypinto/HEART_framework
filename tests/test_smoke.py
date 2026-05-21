from pathlib import Path

from heart.smoke import extract_gpqa_answer_letter, judge_gpqa_rows, run_smoke


def test_extract_gpqa_answer_letter():
    assert extract_gpqa_answer_letter("The final answer is B.") == "B"
    assert extract_gpqa_answer_letter("Answer: (c)") == "C"
    assert extract_gpqa_answer_letter("D") == "D"
    assert extract_gpqa_answer_letter("No answer here") == ""


def test_rule_based_gpqa_judging():
    rows = [
        {"shuffled_correct_key": "A", "clean_model_response_it_0": "Answer: A"},
        {"shuffled_correct_key": "D", "clean_model_response_it_0": "Answer: B"},
    ]
    judged = judge_gpqa_rows(rows, "clean_model_response_it_0")
    assert [row["predicted_letter"] for row in judged] == ["A", "B"]
    assert [row["rule_based_correct"] for row in judged] == ["True", "False"]


def test_run_smoke_writes_output(tmp_path):
    input_file = Path(__file__).parents[1] / "examples" / "data" / "gpqa_sample_predictions.csv"
    output_file = run_smoke(input_file, tmp_path, "clean_model_response_it_0")
    assert output_file.exists()
    text = output_file.read_text(encoding="utf-8")
    assert "predicted_letter" in text
    assert "rule_based_correct" in text
