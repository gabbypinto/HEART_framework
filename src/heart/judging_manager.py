# judging_manager.py

import pandas as pd
import re
import ast
import logging
import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np


from typing import Literal
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset

# --- Imports for OlympiadBench ---
from heart.math_judger import MathJudger
from heart.judge import judge_result 


# --- Imports for LiveCodeBench ---
# # from lcb_runner.evaluation import extract_instance_results
# from lcb_evaluation import extract_instance_results
# from lcb_runner.scenario_router import get_metrics, combine_results
# from lcb_benchmarks.code_generation import CodeGenerationProblem
# from livecode_lm_styles import LanguageModelStore

# from heart.api_client import ApiClient


logger = logging.getLogger(__name__)

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability


class JudgingManager:
    """Handles the extraction and judging of model answers against ground truths."""

    def __init__(self, subject: str, benchmark_name: str, lcb_args=None):
        """
        Initializes the JudgingManager.

        Args:
            subject (str): The subject of the experiment (e.g., 'Math', 'codegeneration').
            benchmark_name (str): The name of the benchmark ('OlympiadBench' or 'LiveCodeBench').
            lcb_args: Parsed arguments containing config needed for LCB evaluation.
        """
        if type(subject) == "str":
            self.subject = subject.lower()
        else:
            self.subject=  subject
        self.benchmark_name = benchmark_name
        self.lcb_args = lcb_args
        
        if self.benchmark_name == "OlympiadBench":
            self.judger = MathJudger()
            # pass

        # logger.info(f"JudgingManager initialized for benchmark: {self.benchmark_name}")


    def _extract_answer(self, model_output: str) -> str:
        """Extracts the final answer from a model's full response text."""
        if not isinstance(model_output, str):
            return ""

        matches_final_answer_phrase = re.findall(r'The final answer is\s*(.*)', model_output, re.IGNORECASE)
        if matches_final_answer_phrase:
            # If the phrase is found, return the first captured segment.
            return matches_final_answer_phrase[0].strip()
        # If "The final answer is" phrase is NOT found,
        # Look for content within \boxed{} anywhere in the text.
        matches_boxed = re.findall(r'\\boxed\{(.*?)\}', model_output)
        if matches_boxed:
            # If \boxed{} matches are found, return the FIRST one, formatted as LaTeX.
            # This ensures a single string like '$answer$' is returned.
            return f"${matches_boxed[0].strip()}$"
        logger.warning("Answer pattern not found in response.")
        return ""   


    def _judge_single_result(self, extracted_answer: str, goal_answer, answer_type: str, error_tolerance) -> bool:
        """
        This method now acts as a wrapper that calls the more comprehensive
        judge_result function from judge.py.
        """

        # Call the imported function, passing the arguments it needs.
        return judge_result(
            extracted_answer_response=extracted_answer,
            goal_answer=goal_answer,
            answer_type=answer_type,
            answer_error=error_tolerance,
            subject=self.subject
        )

    def judge_dataframe(self, df: pd.DataFrame, iteration: int, prompt_options: dict = None) -> pd.DataFrame:
        """Iterates over a dataframe and judges the correctness of answers for a given iteration."""
        print(f"Starting judgment for iteration {iteration}...")


        if self.benchmark_name == "OlympiadBench":
            # This is your existing logic, which works correctly for OlympiadBench.
            # It handles iteration 0 and subsequent iterations with row-by-row judging.
            if iteration == 0:
                extracted_col = "extracted_answer_it_0"
                is_correct_col = "is_correct_it_0"
                df[is_correct_col] = df.apply(
                    lambda row: self._judge_single_result(
                        row.get(extracted_col, ""), row['final_answer'], row['answer_type'], row['error']
                    ), axis=1
                )
            else:
                if not prompt_options:
                    print("Warning: Judging iteration > 0 but no prompt options were provided.")
                    return df
                for prompt_id in prompt_options.keys():
                    extracted_col = f"extracted_answer_prompt_{prompt_id}_it_{iteration}"
                    is_correct_col = f"is_correct_prompt_{prompt_id}_it_{iteration}"
                    
                    if extracted_col not in df.columns:
                        print(f"Warning: Column '{extracted_col}' not found. Skipping.")
                        continue

                    df[is_correct_col] = df.apply(
                        lambda row: self._judge_single_result(
                            row.get(extracted_col, ""), row['final_answer'], row['answer_type'], row['error']
                        ), axis=1
                    )
        # elif self.benchmark_name == "hle":
            



        ### LiveCodeBench, the below section is not completed for LiveCodeBench
        elif self.benchmark_name == "LiveCodeBench":
            # --- New batch-based judging logic for LiveCodeBench ---
            # LCB evaluation is done once per batch, not row-by-row.
            model_style = LanguageModelStore[self.lcb_args.model]

            for prompt_id in prompt_options.keys():
                response_col = f"clean_model_response_prompt_{prompt_id}_it_{iteration}"
                extracted_col = f"extracted_answer_prompt_{prompt_id}_it_{iteration}"
                is_correct_col = f"is_correct_prompt_{prompt_id}_it_{iteration}"

                if extracted_col not in df.columns:
                    print(f"Warning: Column '{extracted_col}' not found. Skipping judging for prompt {prompt_id}.")
                    continue

                # 1. Transform DataFrame into the list format LCB's evaluator expects.
                benchmark_objects = [CodeGenerationProblem(**row.to_dict()) for _, row in df.iterrows()]
                results_for_lcb = [[resp] for resp in df[response_col].tolist()]
                combined_results = list(zip(results_for_lcb, df[extracted_col].apply(lambda x: [x]).tolist()))

                # 2. Call LCB's batch evaluation function.
                print(f"Running LCB batch evaluation for {len(df)} items...")
                metrics = get_metrics(self.lcb_args.scenario, self.lcb_args, benchmark_objects, combined_results)
                graded_results = extract_instance_results(metrics[1])

                # 3. Map the batch results back to the DataFrame.
                is_correct_series = [res[0] if res else False for res in graded_results]
                df[is_correct_col] = is_correct_series
                
                print(f"Judged {df[is_correct_col].sum()} correct answers for prompt {prompt_id}.")

        print("Judgment complete.")
        return df
