import sys
import os
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
except ImportError:
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

class BatchTestCaseVerifier:
    def __init__(self, model_name, project_id, location, max_workers=20):
        self.model_name = model_name
        self.max_workers = max_workers
        try:
            vertexai.init(project=project_id, location=location)
            self.model = GenerativeModel(self.model_name)
        except Exception as e:
            log.error(f"Vertex Init Error: {e}")
            sys.exit(1)

        self.config = GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            response_mime_type="application/json"
        )

    def verify_batch(self, problem_data):
        q_id = problem_data.get('question_id')
        content = problem_data.get('question_content') or problem_data.get('question/problem')
        
        # 1. GET SOURCE TESTS
        # We verify whatever came from generation
        candidates = problem_data.get('generated_test_cases_parsed', [])
        
        # 2. PREPARE OUTPUT LIST
        # We explicitly copy them to 'verified_test_cases' so we don't overwrite raw data
        # We Default is_valid to False.
        final_tests = []
        for tc in candidates:
            new_tc = tc.copy()
            new_tc['is_valid'] = False 
            final_tests.append(new_tc)
        
        # Attach list to problem object immediately
        problem_data['verified_test_cases'] = final_tests

        if not content or not candidates:
            return problem_data

        # 3. BUILD PROMPT WITH IDS
        # The ID here corresponds exactly to the list index
        tests_block_str = ""
        indices_to_verify = []
        
        for i, tc in enumerate(final_tests):
            inp = str(tc.get('input', ''))
            out = str(tc.get('expected_output', ''))
            
            # Truncate if massive to prevent token overflow errors
            if len(inp) > 1000: inp = inp[:1000] + "...(trunc)"
            
            tests_block_str += f"Test Case ID {i}:\nINPUT: {inp}\nEXPECTED OUTPUT: {out}\n\n"
            indices_to_verify.append(i)

        if not indices_to_verify:
            return problem_data

        prompt = f"""
        You are a smart software QA engineer. Verify these test cases.

        --- PROBLEM ---
        {content[:4000]}
        --- END PROBLEM ---

        --- TEST CASES ---
        {tests_block_str}
        --- END TEST CASES ---

        INSTRUCTIONS:
        1. Check if the EXPECTED OUTPUT is correct for the INPUT based on the problem.
        2. BE LENIENT: Accept Python list formats (e.g. `[1,2]`) even if the problem implies standard input `1 2`.
        3. Return JSON: [{{ "id": 0, "is_valid": true }}, {{ "id": 1, "is_valid": false }}]
        """

        # 4. CALL API & MAP RESULTS
        validity_map = {}
        for _ in range(3):
            try:
                response = self.model.generate_content(
                    prompt, generation_config=self.config, safety_settings=SAFETY_SETTINGS
                )
                results = json.loads(response.text)
                if isinstance(results, list):
                    for r in results:
                        # CRITICAL: We map by ID (index)
                        idx = r.get('id')
                        is_valid = r.get('is_valid', False)
                        if isinstance(idx, int):
                            validity_map[idx] = is_valid
                    break
            except Exception:
                time.sleep(1)

        # 5. UPDATE THE OUTPUT LIST
        for i in indices_to_verify:
            # Check map using the index
            is_valid = validity_map.get(i, False)
            
            # Update the record in place
            problem_data['verified_test_cases'][i]['is_valid'] = is_valid
            problem_data['verified_test_cases'][i]['verified_source'] = 'llm_judge'
            
            # Update status string for clarity
            if is_valid:
                problem_data['verified_test_cases'][i]['verification_status'] = 'success'
            else:
                problem_data['verified_test_cases'][i]['verification_status'] = 'rejected_by_llm'

        return problem_data

    def run(self, input_file, output_file):
        with open(input_file, 'r') as f: data = json.load(f)
        
        log.info(f"Verifying {len(data)} problems...")
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.verify_batch, p) for p in data]
            for f in tqdm(as_completed(futures), total=len(data)):
                results.append(f.result())

        # Save result
        with open(output_file, 'w') as f: json.dump(results, f, indent=4)
        
        # Log stats
        valid_count = 0
        for p in results:
            for t in p['verified_test_cases']:
                if t.get('is_valid') is True:
                    valid_count += 1
                    
        log.info(f"Summary: {valid_count} valid tests found. See {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_workers", type=int, default=20)
    # Ignore extra args
    parser.add_argument("--source_file", default=None) 
    parser.add_argument("--temperature", default=0)
    parser.add_argument("--top_p", default=0)
    parser.add_argument("--max_tokens", default=0)
    parser.add_argument("--max_regeneration_attempts", default=0)
    
    args = parser.parse_args()
    infile = args.input_file
    if not os.path.exists(infile) and args.source_file: infile = args.source_file

    verifier = BatchTestCaseVerifier(args.model_name, args.project_id, args.location, args.max_workers)
    verifier.run(infile, args.output_file)



# import sys
# import os
# import json
# import re
# import time
# import argparse
# import threading
# import logging
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import List, Dict, Any, Optional
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import List, Dict, Any, Optional
# from tqdm import tqdm

# # --- Dependency Check ---
# try:
#     import pandas as pd
#     import vertexai
#     from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold
# except ImportError:
#     print("Error: Required libraries not found. Please run:")
#     print("pip install google-cloud-aiplatform pandas tqdm")
#     sys.exit(1)

# # --- Configuration ---
# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# log = logging.getLogger(__name__)

# # Constants
# RETRY_DELAY_SECONDS = 30
# MAX_RETRIES = 5
# SAFETY_SETTINGS = {
#     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
# }

# class TestCaseVerifier:
#     """
#     This class verifies test cases generated by a model by sending them back
#     to another model (or the same) for validation.
#     """

#     def __init__(self, model_name: str, project_id: str, location: str, max_workers: int = 10, temperature: float = 0.0, top_p: float = 1.0, max_tokens: int = 50):
#         self.model_name = model_name
#         self.project_id = project_id
#         self.location = location
#         self.max_workers = max_workers
#         self.semaphore = threading.Semaphore(max_workers) # Limit concurrent requests

#         # Initialize Vertex AI
#         try:
#             vertexai.init(project=self.project_id, location=self.location)
#             self.model = GenerativeModel(self.model_name)
#             log.info(f"Initialized Vertex AI client for model: {self.model_name}")
#         except Exception as e:
#             log.error(f"Failed to initialize Vertex AI: {e}")
#             log.error("Please ensure you are authenticated (e.g., `gcloud auth application-default login`)")
#             sys.exit(1)

#         # Configure generation settings for verification (low temp, low output)
#         self.generation_config = GenerationConfig(
#             temperature=temperature,
#             top_p=top_p,
#             max_output_tokens=max_tokens,
#             candidate_count=1,
#         )

#     def format_verification_prompt(self, problem_content: str, test_input: str, test_output: str) -> tuple[str, str]:
#         """
#         Creates the system and user prompts for verifying a single test case.
#         """
#         system_prompt = """You are an expert programmer and meticulous software tester. Your task is to verify if a given test case (consisting of an 'input' and an 'expected_output') is correct for the provided programming problem.

#             The "input" represents the data that would be fed into the program (e.g., via standard input).
#             The "expected_output" represents the exact string the program should print to standard output given that input.

#             Carefully analyze the problem description, the input, and the expected output.
#             - If the expected output is precisely what a correct solution to the problem should produce for the given input, respond with ONLY the word "YES".
#             - If the expected output is incorrect, or if the input is malformed according to the problem constraints, respond with ONLY the word "NO".

#             Do not provide any explanation, reasoning, or any other text. Your entire response must be "YES" or "NO"."""

#         user_prompt = f"""
#             Here is the programming problem:
#             --- PROBLEM START ---
#             {problem_content}
#             --- PROBLEM END ---

#             Here is the test case to verify:

#             INPUT:
#             {test_input}

#             EXPECTED OUTPUT:
#             {test_output}

#             Is this test case correct? (YES/NO)
#         """
#         return system_prompt, user_prompt.strip()

#     def _run_single(self, prompt_data: dict) -> dict:
#         """
#         Handles a single API call for one test case.
#         'prompt_data' is a dictionary containing all necessary info.
#         """
        
#         # system_instruction, user_prompt = self.format_verification_prompt(
#         #     prompt_data['question_content'],
#         #     prompt_data['input'],
#         #     prompt_data['expected_output']
#         # )
#         # Access the nested 'test_case' dictionary
#         tc = prompt_data.get('test_case', {})
#         system_instruction, user_prompt = self.format_verification_prompt(
#             prompt_data.get('question_content', ''),
#             tc.get('input', ''),
#             tc.get('expected_output', '')
#         )
        
#         raw_response = "ERROR: NO_ATTEMPT"
#         is_valid = None
#         error_message = None

#         with self.semaphore:
#             for attempt in range(MAX_RETRIES):
#                 try:
#                     response = self.model.generate_content(
#                         [user_prompt],
#                         generation_config=self.generation_config,
#                         safety_settings=SAFETY_SETTINGS,
#                         # system_instruction=system_instruction
#                     )
                    
#                     if not response.candidates:
#                         if response.prompt_feedback and response.prompt_feedback.block_reason:
#                             error_message = f"Prompt blocked: {response.prompt_feedback.block_reason}"
#                         else:
#                             error_message = "No candidates received from Gemini."
#                         raw_response = f"ERROR: {error_message}"
#                         continue # Retry if possible
                    
#                     text_response = "".join(part.text for part in response.candidates[0].content.parts).strip().upper()
#                     raw_response = text_response
                    
#                     if "YES" in text_response:
#                         is_valid = True
#                     elif "NO" in text_response:
#                         is_valid = False
#                     else:
#                         is_valid = False # Treat ambiguous answers as invalid
#                         error_message = f"Ambiguous response: {text_response}"

#                     break # Successful call, exit retry loop

#                 except Exception as e:
#                     error_message = f"Exception in _run_single: {repr(e)}"
#                     log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for QID {prompt_data['question_id']}: {error_message}")
#                     if attempt < MAX_RETRIES - 1:
#                         time.sleep(RETRY_DELAY_SECONDS) # Wait before retrying
#                     else:
#                         raw_response = f"ERROR: {error_message}"
                        
        
#         # return {
#         #     "question_id": prompt_data['question_id'],
#         #     "input": prompt_data['test_input'],
#         #     "expected_output": prompt_data['test_output'],
#         #     "verification_response": raw_response,
#         #     "is_valid": is_valid,
#         #     "error": error_message
#         # }
#         tc = prompt_data.get('test_case', {})
#         return {
#             "question_id": prompt_data['question_id'],
#             "input": tc.get('input'),
#             "expected_output": tc.get('expected_output'),
#             "verification_response": raw_response,
#             "is_valid": is_valid,
#             "error": error_message
#         }

#     def run_verification(self, input_file_path: str, output_file_path: str):
#         """
#         Loads generated test cases, verifies them in parallel, and saves the results.
#         """
        
#         # 1. Load the generated test cases JSON file
#         try:
#             with open(input_file_path, 'r', encoding='utf-8') as f:
#                 problems = json.load(f)
#             print(f"Loaded {len(problems)} problems from {input_file_path}")
#         except FileNotFoundError:
#             print(f"❌ Error: Input file not found at {input_file_path}. Aborting.")
#             return
#         except json.JSONDecodeError as e:
#             print(f"❌ Error: Could not parse JSON from {input_file_path}. Aborting.")
#             return

#         # 2. Create a flat list of all test cases to verify
#         tasks_to_run = []
#         for problem in problems:
#             question_id = problem.get('question_id')
#             # Use 'question/problem' key based on your example, fallback to 'question_content'
#             question_content = problem.get('question_content') 
            
#             if not question_id or not question_content:
#                 print(f"Skipping record due to missing 'question_id' or 'question_content': {problem}")
#                 continue

#             if problem.get('parsing_error') or not problem.get('generated_test_cases_parsed'):
#                 print(f"Skipping problem {question_id}: No valid test cases found (parsing_error or empty list).")
#                 continue

#             for test_case in problem['generated_test_cases_parsed']:
#                 task_data = {
#                     "question_id": question_id,
#                     "question_content": question_content,
#                     "test_case": test_case # This is the {"input": "...", "expected_output": "..."} dict
#                 }
#                 tasks_to_run.append(task_data)

#         if not tasks_to_run:
#             print("No valid test cases found to verify.")
#             return

#         print(f"Found {len(tasks_to_run)} total test cases to verify...")

#         # 3. Run the verification in parallel
#         all_results = []
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             futures = {executor.submit(self._run_single, task): task for task in tasks_to_run}
            
#             for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying Test Cases"):
#                 try:
#                     result = future.result()
#                     all_results.append(result)
#                 except Exception as exc:
#                     task = futures[future]
#                     qid = task.get('question_id', 'Unknown')
#                     print(f'Task for QID {qid} generated an exception: {exc}')
#                     all_results.append({
#                         "question_id": qid,
#                         "input": task['test_case'].get('input'),
#                         "expected_output": task['test_case'].get('expected_output'),
#                         "verification_response": "ERROR: Future failed",
#                         "is_valid": False,
#                         "error": str(exc)
#                     })

#         # 4. Save the verified test cases to a new file
#         try:
#             output_dir = os.path.dirname(output_file_path)
#             if output_dir: os.makedirs(output_dir, exist_ok=True)
                
#             with open(output_file_path, "w", encoding='utf-8') as f:
#                 # Ensure 'problems' is indeed a list before dumping
#                 if isinstance(problems, list):
#                     json.dump(problems, f, indent=4)
#                 else:
#                     print("⚠️ Warning: Data to save is not a list!")
#                     json.dump(problems, f, indent=4)
            
#             print(f"\n✅ Successfully verified and saved {len(all_results)} test cases to: {output_file_path}")
            
#             # Print a summary
#             valid_count = sum(1 for item in all_results if item.get('is_valid') is True)
#             invalid_count = len(all_results) - valid_count
#             print(f"Summary: {valid_count} valid cases, {invalid_count} invalid cases.")

#         except Exception as e:
#             print(f"❌ Error saving results to JSON file '{output_file_path}': {e}")


# def main():
#     parser = argparse.ArgumentParser(description="Verify generated test cases using an LLM.")
    
#     parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use (e.g., 'gemini-1.0-pro')")
#     parser.add_argument("--project_id", type=str, required=True, help="Google Cloud Project ID")
#     parser.add_argument("--location", type=str, required=True, help="Google Cloud Location (e.g., 'us-central1')")
#     parser.add_argument("--input_file", type=str, required=True, help="Path to the generated test cases JSON file (output from previous step).")
#     parser.add_argument("--output_file", type=str, required=True, help="Path to save the verified test cases JSON file.")    
#     parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel threads for API calls.")
#     parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
#     parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling.")
#     parser.add_argument("--max_tokens", type=int, default=50, help="Max output tokens for verification.")
    
#     args = parser.parse_args()

#     # --- Initialize and Run ---
#     verifier = TestCaseVerifier(
#         model_name=args.model_name,
#         project_id=args.project_id,
#         location=args.location,
#         max_workers=args.max_workers,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         max_tokens=args.max_tokens
#     )
    
#     verifier.run_verification(
#         input_file_path=args.input_file,
#         output_file_path=args.output_file
#     )

# if __name__ == "__main__":
#     main()