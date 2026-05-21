import json
from collections import defaultdict
import math # Used for median calculation

def calculate_median(counts_list):
    """Helper to calculate median from a list of numbers."""
    if not counts_list:
        return 0
    sorted_list = sorted(counts_list)
    n = len(sorted_list)
    mid = n // 2
    if n % 2 == 0: # Even number of items
        return (sorted_list[mid - 1] + sorted_list[mid]) / 2
    else: # Odd number of items
        return sorted_list[mid]

def analyze_json_stats_detailed(filepath):
    """
    Analyzes a JSON file for detailed statistics on valid test cases
    per question_id.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        return

    if not isinstance(data, list) or not data:
        print("Error: JSON content is not a list of records or is empty.")
        return

    # --- Data Processing ---
    all_question_ids = set()
    valid_case_counts = defaultdict(int)
    total_valid = 0
    total_invalid = 0
    # New: Collect error types if 'error' is not null
    error_counts = defaultdict(int)

    for record in data:
        qid = record.get("question_id")
        is_valid = record.get("is_valid", False)
        error = record.get("error")

        if qid:
            all_question_ids.add(qid)
            
            if is_valid:
                valid_case_counts[qid] += 1
                total_valid += 1
            else:
                total_invalid += 1
                # If it's invalid, let's see why (if info is present)
                if error:
                    error_counts[str(error)] += 1 # Use str as key

    # --- Report Statistics ---
    
    print(f"--- Overall Summary ---")
    print(f"Total records processed: {len(data)}")
    print(f"Total unique question IDs: {len(all_question_ids)}")
    print(f"Total VALID cases: {total_valid}")
    print(f"Total QIDs of VALID cases: {len(valid_case_counts)}")
    print(f"Total INVALID cases: {total_invalid}")
    if len(data) > 0:
        valid_rate = (total_valid / len(data)) * 100
        print(f"Overall validity rate: {valid_rate:.2f}%")

    # --- Statistics on Valid Case Distribution ---
    # Get a list of the counts of valid cases for all unique IDs
    # This includes 0s for IDs that had no valid cases
    counts_per_id = [valid_case_counts[qid] for qid in all_question_ids]

    if counts_per_id:
        print(f"\n" + f"--- Stats for 'Valid Cases per ID' ---")
        avg_cases = sum(counts_per_id) / len(counts_per_id)
        median_cases = calculate_median(counts_per_id)
        max_cases = max(counts_per_id)
        min_cases = min(counts_per_id) # Will be 0 if any ID has 0 valid cases

        print(f"Average valid cases per ID: {avg_cases:.2f}")
        print(f"Median valid cases per ID: {median_cases}")
        print(f"Max valid cases for a single ID: {max_cases}")
        print(f"Min valid cases for a single ID: {min_cases}")

        # Find which IDs correspond to the max/min
        ids_with_max = [qid for qid in all_question_ids if valid_case_counts[qid] == max_cases]
        ids_with_min = [qid for qid in all_question_ids if valid_case_counts[qid] == min_cases]

        print(f"ID(s) with max cases ({max_cases}): {', '.join(ids_with_max)}")
        print(f"ID(s) with min cases ({min_cases}): {len(ids_with_min)} (listed below)")
        # Only list IDs with 0 cases if it's manageable
        if min_cases == 0:
             print(f"  > Question IDs with 0 valid cases: {len(ids_with_min)}")
             if len(ids_with_min) < 20: # Avoid huge printout
                 print(f"    {', '.join(ids_with_min)}")
        

    print(len(all_question_ids))
    # --- Valid Cases per ID (The original request) ---
    # print(f"\n" + f"--- Valid Case Count per Question ID (Sorted) ---")
    # for qid in sorted(all_question_ids):
    #     print(f"{qid}: {valid_case_counts[qid]}")

    # --- (Suggested) Error Analysis ---
    # if error_counts:
    #     print(f"\n" + f"--- Invalid Case Error Breakdown ---")
    #     for error_type, count in error_counts.items():
    #         print(f"{error_type}: {count}")

# --- Main execution ---
if __name__ == "__main__":
    json_file_path = "code/livecode/test_cases/MEDIUM_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json"
    analyze_json_stats_detailed(json_file_path)

# df=pd.read_json("code/livecode/test_cases/MEDIUM_GEMINI-3-PRO-PREVIEW_gen_verified_test_cases.json")

# print(df.columns.to_list())
