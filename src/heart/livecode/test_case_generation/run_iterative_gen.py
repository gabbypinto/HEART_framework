import subprocess
import json
import os
import argparse
import sys
import math

# --- Utility Functions ---
def load_json(filepath):
    if not os.path.exists(filepath): 
        # Silent return for empty file checks
        return []
    try:
        with open(filepath, 'r') as f: return json.load(f)
    except Exception as e:
        print(f"⚠️ Warning: Could not read {filepath}: {e}")
        return []

def save_json(data, filepath):
    with open(filepath, 'w') as f: json.dump(data, f, indent=4)

# def get_all_problem_ids(difficulty):
#     """
#     Discovers problem IDs by loading the dataset directly via a subprocess.
#     """
#     print(f"   > Querying dataset for difficulties: {difficulty}")
    
#     python_code = f"""
# import sys
# try:
#     from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    
#     # Load all problems (our modified loader returns everything)
#     problems = load_code_generation_dataset()
    
#     # Filter
#     targets = set('{difficulty}'.lower().split(','))
    
#     # p.difficulty is an Enum, so we use .value
#     selected_ids = [
#         p.question_id for p in problems 
#         if p.difficulty.value.lower() in targets
#     ]
    
#     print(','.join(selected_ids))
# except Exception as e:
#     print(f"ERROR: {{e}}", file=sys.stderr)
#     sys.exit(1)
# """
#     cmd = [sys.executable, "-c", python_code]
    
#     try:
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         ids = result.stdout.strip()
#         if not ids:
#             return []
#         return ids.split(',')
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Failed to get ID list.")
#         print(f"   Error Output: {e.stderr}")
#         return []





def get_all_problem_ids(difficulty):
    print(f"   > Querying dataset for difficulties: {difficulty}")
    
    python_code = f"""
import sys
try:
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    problems = load_code_generation_dataset()
    targets = set('{difficulty}'.lower().split(','))
    selected_ids = [p.question_id for p in problems if p.difficulty.value.lower() in targets]
    print("START_IDS" + ",".join(selected_ids) + "END_IDS")
except Exception as e:
    sys.exit(1)
"""
    cmd = [sys.executable, "-c", python_code]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw = result.stdout.strip()
        # Use markers to ignore any "RELEASE VERSION" or "Loaded X problems" logs
        # if "START_IDS" in raw and "END_IDS" in raw:
        #     id_part = raw.split("START_IDS")[1].split("END_IDS")[0]
        #     return [i for i in id_part.split(',') if i]
        if "START_IDS" in raw and "END_IDS" in raw:
            id_part = raw.split("START_IDS")[1].split("END_IDS")[0]
            # .strip() each individual ID to remove \n or spaces
            return [i.strip() for i in id_part.split(',') if i and " " not in i.strip()]
        
        return []
    except Exception:
        return []
    


def merge_results(master_path, new_path):
    """
    Merges newly verified test cases into the master file.
    Includes robust checks for the 'is_valid' flag.
    """
    # print(f"   > Loading new verification data from {new_path}...")
    new_data = load_json(new_path)
    
    if not new_data:
        print("   ⚠️ New verification file is empty or invalid.")
        return 0

    # Initialize Master if it doesn't exist
    if not os.path.exists(master_path): 
        save_json([], master_path)
    
    master_list = load_json(master_path)
    master_data = {str(p.get('question_id')): p for p in master_list}
    
    count_new_valid_total = 0

    for item in new_data:
        qid = str(item.get('question_id'))
        
        # 1. Get the list of tests (Prioritize verified list)
        raw_tests = item.get('verified_test_cases', [])
        if not raw_tests:
            raw_tests = item.get('generated_test_cases_parsed', [])

        valid_tests = []
        for t in raw_tests:
            # 2. Robust Validity Check
            val = t.get('is_valid')
            is_valid_bool = False
            
            if val is True: 
                is_valid_bool = True
            elif isinstance(val, str) and val.lower() == 'true': 
                is_valid_bool = True
            
            if is_valid_bool:
                t['is_valid'] = True 
                valid_tests.append(t)
        
        # 3. Merge Logic
        if valid_tests: 
            count_new_valid_total += len(valid_tests)
            
            if qid not in master_data:
                master_data[qid] = {
                    "question_id": qid,
                    "question_title": item.get('question_title'),
                    "question_content": item.get('question_content'),
                    "difficulty": item.get('difficulty'),
                    "verified_test_cases": valid_tests
                }
            else:
                existing = master_data[qid].get('verified_test_cases', [])
                existing_sigs = set(str(t.get('input')).strip() for t in existing)
                
                for t in valid_tests:
                    sig = str(t.get('input')).strip()
                    if sig not in existing_sigs:
                        existing.append(t)
                        existing_sigs.add(sig)
                
                master_data[qid]['verified_test_cases'] = existing

    # 4. Save Updates
    final_list = list(master_data.values())
    save_json(final_list, master_path)
    
    print(f"   > Merged {count_new_valid_total} valid tests.")
    return len(final_list)

def run_loop(args):
    MASTER_FILE = "MASTER_GEMINI-2.5-PRO_gen_verified_test_cases.json"

    # 1. Get Universe (Total scope)
    print("🔍 Discovering total problem set...")
    all_ids = get_all_problem_ids(args.difficulty)
    if not all_ids or all_ids == ['']:
        print("❌ Could not find any problems. Check difficulty arg.")
        return
    
    print(f"🎯 Total Universe: {len(all_ids)} problems.")
    
    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'='*40}\n🚀 ITERATION {iteration}/{args.max_iterations}\n{'='*40}")
        
        # 2. Determine Unsolved based on Master File (Resumes work)
        master_data = load_json(MASTER_FILE)
        solved_ids = set()
        for p in master_data:
            v = [t for t in p.get('verified_test_cases', []) 
                 if t.get('is_valid') is True or str(t.get('is_valid')).lower() == 'true']
            if v: solved_ids.add(str(p['question_id']))
            
        unsolved_ids = sorted(list(set(all_ids) - solved_ids))
        
        print(f"📊 Status: {len(solved_ids)}/{len(all_ids)} Solved.")
        print(f"🔄 Remaining Unsolved: {len(unsolved_ids)}")
        
        if not unsolved_ids:
            print("🎉 SUCCESS: All problems solved!")
            break

        # 3. Dynamic Chunking Strategy
        # If we have many problems, go fast (50). If we have few (likely the hard ones), go slow (5).
        # Also, if we are in later iterations, force small chunks.
        if len(unsolved_ids) < 50 or iteration > 1:
            CHUNK_SIZE = 5
            print(f"⚠️  Processing difficult/tail-end problems. Using SMALL Batch Size: {CHUNK_SIZE}")
        else:
            CHUNK_SIZE = 5
            print(f"⚡ Processing large batch. Using NORMAL Batch Size: {CHUNK_SIZE}")

        num_chunks = math.ceil(len(unsolved_ids) / CHUNK_SIZE)
        
        for i in range(num_chunks):
            chunk_ids = unsolved_ids[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE]
            print(f"\n📦 Processing Chunk {i+1}/{num_chunks} ({len(chunk_ids)} problems)...")
            
            gen_file = f"iter_{iteration}_chunk_{i}_gen.json"
            ver_file = f"iter_{iteration}_chunk_{i}_ver.json"
            
            # A. GENERATE
            # Increase temp slightly every iteration to break stuck loops
            current_temp = 0.2 + ((iteration - 1) * 0.1)
            
            cmd_gen = [
                sys.executable, "-m", "lcb_runner.runner.main",
                "--model", args.model,
                "--scenario", "codegeneration",
                "--codegen_n", "1",
                "--evaluate",
                "--generate_test_cases_output_file", gen_file,
                "--temperature", str(current_temp),
                "--multiprocess", str(args.multiprocess),
                "--question_ids_subset", ",".join(chunk_ids),
                "--difficulty", args.difficulty 
            ]
            try:
                # 20 min timeout for generation batch
                subprocess.run(cmd_gen, check=True, timeout=1200) 
                
            except subprocess.TimeoutExpired:
                print("⏰ Generation Timed Out. Skipping chunk.")
                continue
            except subprocess.CalledProcessError:
                print("❌ Generation Failed. Skipping chunk.")
                continue


            if not os.path.exists(gen_file):
                print(f"⚠️  Generation failed for {chunk_ids}. Skipping verification.")
                # print(f"❌ Error: {gen_file} was not created. Skipping verification for this chunk.")
                continue


            # B. VERIFY
            cmd_ver = [
                sys.executable, "-m", "lcb_runner.runner.verify_tests",
                "--model_name", args.verifier_model,
                "--project_id", args.project_id,
                "--location", args.location,
                "--input_file", gen_file,
                "--output_file", ver_file,
                "--max_workers", "10" # Keep verification stable
            ]
            try:
                subprocess.run(cmd_ver, check=True)
            except Exception as e:
                print(f"❌ Verification Failed: {e}")
                continue

            # C. MERGE
            merge_results(MASTER_FILE, ver_file)
            
            # Cleanup temp files
            try:
                if os.path.exists(gen_file): os.remove(gen_file)
                if os.path.exists(ver_file): os.remove(ver_file)
            except: pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--verifier_model", default="gemini-2.5-flash")
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--max_iterations", type=int, default=3)
    parser.add_argument("--multiprocess", type=int, default=10)
    args = parser.parse_args()
    
    run_loop(args)

