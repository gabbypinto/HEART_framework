import re
import ast

import pandas as pd

from heart.math_judger import MathJudger

def _extract_boxed_last(text: str) -> str | None:
    """Extract content of the last \\boxed{} in text with proper nested-brace handling."""
    positions = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    if not positions:
        return None
    # Walk from the last \\boxed{ to its matching }
    start = positions[-1] + len(r'\boxed{')
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def extract_answer(is_chinese, model_output, is_deepseek=False):
    """Extract the final answer from a model response.

    Priority order (applied for all models):
      1. Model-specific phrase patterns (existing logic)
      2. \\boxed{...}  — AIME / MATH style (last occurrence, nested-brace-safe)
      3. **Answer:** / **Final Answer:** N  — DeepSeek-style bold answer
      4. ^Answer: N  — plain 'Answer:' line
      5. 'The [final] answer is N'
      6. Returns "" rather than the full response so MathJudger gets a clean input.
    """
    if not isinstance(model_output, str):
        return ""

    # ── 1. Model-specific phrase patterns ─────────────────────────────────────
    if is_deepseek:
        patterns = ['## 解题答案(.*)'] if is_chinese else ['The answer is: (.*)']
    else:
        patterns = ['所以最终答案是(.*)'] if is_chinese else ['So the final answer is (.*)']

    for pat in patterns:
        matches = re.findall(pat, model_output)
        if matches:
            return matches[-1].strip()

    # ── 2. \\boxed{...} ───────────────────────────────────────────────────────
    boxed = _extract_boxed_last(model_output)
    if boxed is not None:
        return boxed

    def _clean(s: str) -> str:
        """Strip surrounding bold markers, whitespace, and trailing punctuation."""
        return s.strip().strip('*').strip().rstrip('.,;: ').strip()

    # ── 3. Bold-answer formats (DeepSeek style, two variants) ─────────────────
    # Variant A: **Answer:** 540   (value sits OUTSIDE the bold markers)
    m = re.findall(r'\*\*(?:[Ff]inal\s+)?[Aa]nswer\s*:\s*\*\*\s*(.*?)(?:\n|$)', model_output)
    if m:
        ans = _clean(m[-1])
        if ans:
            return ans

    # Variant B: **Answer: 16**   (value sits INSIDE the bold markers)
    m = re.findall(r'\*\*(?:[Ff]inal\s+)?[Aa]nswer:\s*(.*?)\*\*', model_output)
    if m:
        ans = _clean(m[-1])
        if ans:
            return ans

    # ── 4. ^Answer: N  (plain label, last occurrence) ─────────────────────────
    m = re.findall(r'(?:^|\n)(?:[Ff]inal\s+)?[Aa]nswer:\s*(.*?)(?:\n|$)', model_output)
    if m:
        ans = _clean(m[-1])
        if ans:
            return ans

    # ── 5. 'The [final] answer is N' ──────────────────────────────────────────
    m = re.findall(r'[Tt]he\s+(?:final\s+)?answer\s+is\s+(.*?)(?:\.|,|\n|$)', model_output)
    if m:
        ans = _clean(m[-1])
        if ans:
            return ans

    # ── 6. Fallback: return empty string (do NOT return the full response) ────
    return ""

def judge_result(extracted_answer_response,goal_answer,answer_type,answer_error,subject):
        judger = MathJudger()
        is_chinese = False 
        is_deepseek = False 
        count = 0
        output_data = []

        # clean_model_answer = extract_answer(is_chinese,extracted_answer_response[0])
        # Pass the extracted_answer_from_model_response directly to extract_answer
        # because it's already a single string (or None) from main code's extract_model_answer.
        # Handle the case where it might be None.
        if extracted_answer_response is None:
            # If baseline_prompt.py's extract_model_answer returned None,
            # it means no answer could be extracted by its rules.
            # You might want to assign a default string or handle this as an unjudgable case.
            # For judging, an empty string is often safer than None for comparison.
            clean_model_answer = ""
        else:
            # If it's not None, pass it to extract_answer (from this judge.py file).
            # This extract_answer function might still refine it or return the whole string if no match.
            clean_model_answer = extract_answer(is_chinese, extracted_answer_response)

        # --- Type conversion for goal_answer ---
        # The traceback indicates goal_answer is a list (e.g., goal_answer[0]).
        # Ensure goal_answer is correctly accessed as a single scalar.
        # If goal_answer is always a list containing one element, access it like this:
        if isinstance(goal_answer, list) and len(goal_answer) > 0:
            actual_goal_answer = goal_answer[0]
        else:
            actual_goal_answer = goal_answer # Assume it's already scalar or handle appropriately


        if 'Tuple' in answer_type:
            judge_result = judger.judge(clean_model_answer,goal_answer[0]) 
        else:
            if pd.notna(answer_error): #check if this var is not NaN 
                if isinstance(answer_error,str) and ',' in answer_error:
                    precisions = answer_error.split(',')
                    precisions = [float(p) if p else 1e-8 for p in precisions]
                    judge_result = judger.judge(clean_model_answer,goal_answer[0],precisions)
                elif isinstance(answer_error,(int,float)):
                    precision = float(answer_error)
                    judge_result = judger.judge(clean_model_answer,goal_answer[0],precision)
                else:
                    precision = float(answer_error)
                    if subject.lower() != 'physics':
                        goal_answer = ast.literal_eval(goal_answer)  ###physics
                    
                    judge_result = judger.judge(clean_model_answer, str(goal_answer[0]),precision)  ###changed for physics
            else: ## error does not exist 
                if subject.lower() == 'math':
                    pass
                elif subject.lower() == 'physics':
                    pass
                    # goal_answer = ast.literal_eval(goal_answer)

                if isinstance(goal_answer,str):
                    judge_result = judger.judge(clean_model_answer, goal_answer)
                else:
                    if isinstance(goal_answer,list):
                        judge_result = False
                        for goal in goal_answer:
                            if judger.judge(clean_model_answer, goal):
                                judge_result = True  # Assign to the correct variable
                                break 
                    else:
                        new_goal_answer = str(goal_answer[0])
                        judge_result = judger.judge(clean_model_answer, new_goal_answer) 
            # print(judge_result)
            return judge_result
