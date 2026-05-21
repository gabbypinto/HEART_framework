# data_manager.py

import pandas as pd
import os
import sys
import logging
from heart import config 
import glob
logger = logging.getLogger(__name__)

"""
Do after simpleqa verified
MATH500:
https://huggingface.co/datasets/di-zhang-fdu/MATH500
https://huggingface.co/datasets/HuggingFaceH4/MATH-500

MATH:
https://huggingface.co/datasets/HuggingFaceH4/MATH


Run the benchmarks below is time allows:
OpenMathReasoning:  <---- really big 
https://huggingface.co/datasets/nvidia/OpenMathReasoning

AIMO:
https://huggingface.co/datasets/AI-MO/aimo-validation-aime



#### integrated 
AIME2024:
https://huggingface.co/datasets/HuggingFaceH4/aime_2024

AIME2025:
dataset = load_dataset("math-ai/aime25")
https://huggingface.co/datasets/math-ai/aime25



"""

# if MODEL == "claude-sonnet-4@20250514":
#     available_regions = ["us-east5", "europe-west4", "GLOBAL"]
# elif MODEL == "claude-opus-4@20250514":
#     available_regions = ["us-east5", "europe-west4"]

class DataManager:
    """Manages loading data, saving results, and preparing inputs for the next iteration."""

    def __init__(self, experiment_id: str, subject: str, model_name: str, prompt_strategy:str, data_split:str, benchmark_name:str):
        """
        Initializes the DataManager.

        Args:
            experiment_id (str): The unique ID for the experiment (e.g., from experiments.json).
            subject (str): The subject of the experiment (e.g., 'Math').
            model_n
            ame (str): The name of the model being used.
        """
        self.experiment_id = experiment_id
        self.subject = subject.lower()
        self.model_name = model_name
        self.prompt_strategy = prompt_strategy
        self.data_split = data_split.lower()
        self.benchmark_name = benchmark_name

        # Construct the experiment directory path from the experiment_id

        if self.data_split == "test":
            base_dir = config.TEST_DATA_EXPERIMENTS_RESULTS_DIR
            print("base_dir (config):",base_dir)
            print("self.experiment_id: ",self.experiment_id)
            # os.makedirs(self.experiment_id,exist_ok=True)
            chunkfolder = f"{base_dir}/{self.experiment_id}/{self.prompt_strategy}/"
            os.makedirs(chunkfolder,exist_ok=True)
            print("chunkfolder:",chunkfolder)
            self.experiment_dir = os.path.join(base_dir,chunkfolder)
            print("self.experiment_dir:",self.experiment_dir)
            os.makedirs(self.experiment_dir, exist_ok=True)
            print(f"DataManager initialized. Path set to: {self.experiment_dir}")
            logger.info(f"DataManager initialized. Path set to: {self.experiment_dir}")

        elif self.data_split == "validation":
            base_dir = config.VALIDATION_DATA_EXPERIMENTS_RESULTS_DIR
            print("base_dir (config):",base_dir)
            print("self.experiment_id: ",self.experiment_id)
            # os.makedirs(self.experiment_id,exist_ok=True)
            chunkfolder = f"{base_dir}/{self.experiment_id}/{self.prompt_strategy}/"
            os.makedirs(chunkfolder,exist_ok=True)
            print("chunkfolder:",chunkfolder)
            self.experiment_dir = os.path.join(base_dir,chunkfolder)
            print("self.experiment_dir:",self.experiment_dir)
            os.makedirs(self.experiment_dir, exist_ok=True)
            print(f"DataManager initialized. Path set to: {self.experiment_dir}")
            logger.info(f"DataManager initialized. Path set to: {self.experiment_dir}")


    def _get_filepath(self, iteration: int, stage: str) -> str:
        """Constructs a standardized file path for different stages."""
        # stages: 'input', 'raw_output', 'judged_output'
        
        filename = f"{self.subject.capitalize()}_{self.model_name}_it{iteration}_{stage}.parquet"
        print("filename in get_filepath:",filename) 
        print("****EXPERIMENT DIR:****",self.experiment_dir)
        return os.path.join(self.experiment_dir, filename)


    def simpleqa_combine_chunked_files(self, iteration: int, file_type: str) -> pd.DataFrame:
        """
        Finds and combines all chunked output files for a given iteration and file type.
        
        Args:
            iteration (int): The iteration number.
            file_type (str): The type of file to combine (e.g., "raw_output").
            
        Returns:
            pd.DataFrame: A single DataFrame containing all combined data.
                        Returns an empty DataFrame if no files are found.
        """
        # First, find the correct directory
        # This path is where your generation stage saves the files
        chunk_output_dir = os.path.join(self.experiment_dir, "chunks")
        iteration = int(iteration)
        # --- THIS IS THE CRITICAL FIX ---
        # Construct a robust pattern to match all raw output chunk files
        # The wildcard '*' matches any part of the filename
        print(self.benchmark_name)
        if int(iteration) >= 0:
            search_pattern = os.path.join(
                chunk_output_dir,
                f"{self.benchmark_name}_{self.subject}_{self.model_name}_{self.data_split}_chunk_*_of_*_input_it_{iteration}_raw_output_it_{iteration}.parquet"
            )
            chunk_files = glob.glob(search_pattern)

        if not chunk_files:
            print(f"Warning: No chunk files found with pattern: '{search_pattern}'")
            return pd.DataFrame()

        print(f"Found {len(chunk_files)} chunk files to combine.")
        combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        print(f"Successfully combined into a DataFrame with {len(combined_df)} rows.")

        return combined_df

    def raw_output_combine_chunked_files_for_ensemble(self, iteration: int, file_type: str) -> pd.DataFrame:
        """
        Finds and combines all chunked output files for a given iteration and file type.
        
        Args:
            iteration (int): The iteration number.
            file_type (str): The type of file to combine (e.g., "raw_output").
            
        Returns:
            pd.DataFrame: A single DataFrame containing all combined data.
                        Returns an empty DataFrame if no files are found.
        """
        chunk_output_dir = os.path.join(self.experiment_dir, "raw_outputs", f"it_{iteration}")
        
        iteration = int(iteration)

        search_pattern = os.path.join(
            chunk_output_dir,
            f"raw_output_it_{iteration}_*.parquet"
        )
        
        print(f"Searching for files with pattern: '{search_pattern}'")
        chunk_files = glob.glob(search_pattern)

        if not chunk_files:
            print(f"Warning: No chunk files found with the specified pattern.")
            return pd.DataFrame()

        print(f"Found {len(chunk_files)} chunk files to combine.")
        
        try:
            combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
            print(f"Successfully combined into a DataFrame with {len(combined_df)} rows.")
            return combined_df
        except Exception as e:
            print(f"An error occurred while combining files: {e}")
            return pd.DataFrame()


    # def olympiad_combine_chunked_files(self, iteration: int, file_type: str) -> pd.DataFrame:
    #     """
    #     Finds and combines all chunked output files for a given iteration and file type.
        
    #     Args:
    #         iteration (int): The iteration number.
    #         file_type (str): The type of file to combine (e.g., "raw_output").
            
    #     Returns:
    #         pd.DataFrame: A single DataFrame containing all combined data.
    #                     Returns an empty DataFrame if no files are found.
    #     """
    #     # First, find the correct directory
    #     # This path is where your generation stage saves the files
    #     chunk_output_dir = os.path.join(self.experiment_dir, "raw_outputs")
    #     chunk_output_dir = os.path.join(chunk_output_dir,f"it_{iteration}")

    #     iteration = int(iteration)
    #     if int(iteration) >= 0:
    #         search_pattern = os.path.join(
    #             chunk_output_dir,
    #             f"{self.benchmark_name}_{self.subject.capitalize()}_{self.model_name}_{self.data_split}_chunk_*_of_*_input_it_{iteration}_raw_output_it_{iteration}.parquet"
    #         )
    #         chunk_files = glob.glob(search_pattern)

    #     if not chunk_files:
    #         print(f"Warning: No chunk files found with pattern: '{search_pattern}'")
    #         return pd.DataFrame()

    #     print(f"Found {len(chunk_files)} chunk files to combine.")
    #     combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    #     print(f"Successfully combined into a DataFrame with {len(combined_df)} rows.")

    #     return combined_df

    def olympiad_combine_chunked_files(self, iteration: int, file_type: str) -> pd.DataFrame:
        """
        Finds and combines all chunked output files, handling GPQA's unique
        (non-subject) naming convention.
        
        Args:
            iteration (int): The iteration number.
            file_type (str): The type of file to combine (e.g., "raw_output").
            
        Returns:
            pd.DataFrame: A single DataFrame containing all combined data.
                        Returns an empty DataFrame if no files are found.
        """
        chunk_output_dir = os.path.join(self.experiment_dir, "raw_outputs", f"it_{iteration}")

        iteration = int(iteration)
        
        # --- NEW: Dynamically determine search pattern based on benchmark name ---
        
        # 1. Determine the base filename components
        if self.benchmark_name.lower() == "gpqa":
            # GPQA has no subject field in its chunk name
            search_pattern_base = f"{self.benchmark_name}_{self.model_name}_{self.data_split}"
        elif self.benchmark_name.lower() == "simpleqa":
             # SimpleQA (also often non-subject specific)
             search_pattern_base = f"{self.benchmark_name}_{self.model_name}_{self.data_split}"
        elif self.benchmark_name.lower() == "aime2025":
             # SimpleQA (also often non-subject specific)
             search_pattern_base = f"{self.benchmark_name}_{self.model_name}_{self.data_split}"
        else:
            # Default to OlympiadBench/HLE style (Subject is included)
            search_pattern_base = f"{self.benchmark_name}_{self.subject.capitalize()}_{self.model_name}_{self.data_split}"

        # 2. Construct the full search pattern
        search_pattern = os.path.join(
            chunk_output_dir,
            f"{search_pattern_base}_chunk_*_of_*_input_it_{iteration}_raw_output_it_{iteration}.parquet"
        )
        # --- END NEW LOGIC ---
        
        print(f"Searching for files with pattern: '{search_pattern}'")
        chunk_files = glob.glob(search_pattern)

        if not chunk_files:
            print(f"Warning: No chunk files found with pattern: '{search_pattern}'")
            return pd.DataFrame()

        print(f"Found {len(chunk_files)} chunk files to combine.")
        combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        print(f"Successfully combined into a DataFrame with {len(combined_df)} rows.")

        return combined_df
    
    def get_raw_output_chunk_path(self, it_num: int, chunk_filename: str) -> str:
        """
        Constructs the path for a raw output chunk based on an input chunk's filename
        by precisely replacing the input suffix with the raw_output suffix.
        """
        print("fname:",chunk_filename)
        basename = chunk_filename.removesuffix('.parquet')
        raw_output_fn = f"{basename}_raw_output_it_{it_num}.parquet"

        # # The raw output chunks are saved inside the 'chunks' subdirectory
        chunk_dir = os.path.join(self.experiment_dir, "chunks")
        print(raw_output_fn)
        print(os.path.join(chunk_dir, raw_output_fn))
        return os.path.join(chunk_dir, raw_output_fn)

    def hle_combine_chunked_files(self, iteration: int, file_type: str) -> pd.DataFrame:
        """
        Finds and combines all chunked output files for a given iteration and file type.
        
        Args:
            iteration (int): The iteration number.
            file_type (str): The type of file to combine (e.g., "raw_output").
            
        Returns:
            pd.DataFrame: A single DataFrame containing all combined data.
                        Returns an empty DataFrame if no files are found.
        """
        chunk_output_dir = os.path.join(self.experiment_dir, "chunks")
        iteration = int(iteration)
        print(self.benchmark_name)
        if int(iteration) >= 0:
            search_pattern = os.path.join(
                chunk_output_dir,
                f"{self.subject.capitalize()}_{self.model_name}_{self.data_split}_chunk_*_of_*_input_it_{iteration}_raw_output_it_{iteration}.parquet"
            )
            chunk_files = glob.glob(search_pattern)

        if not chunk_files:
            print(f"Warning: No chunk files found with pattern: '{search_pattern}'")
            return pd.DataFrame()

        print(f"Found {len(chunk_files)} chunk files to combine.")
        combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        print(f"Successfully combined into a DataFrame with {len(combined_df)} rows.")

        return combined_df

    def simpleqa_combine_chunked_files(self, iteration: int, file_type: str) -> pd.DataFrame:
        """
        Finds and combines all chunked output files for a given iteration and file type.
        
        Args:
            iteration (int): The iteration number.
            file_type (str): The type of file to combine (e.g., "raw_output").
            
        Returns:
            pd.DataFrame: A single DataFrame containing all combined data.
                        Returns an empty DataFrame if no files are found.
        """

        chunk_output_dir = os.path.join(self.experiment_dir, "chunks")
        iteration = int(iteration)
        print(self.benchmark_name)
        if int(iteration) >= 0:
            search_pattern = os.path.join(
                chunk_output_dir,
                f"{self.benchmark_name}_{self.model_name}_{self.data_split}_chunk_*_of_*_input_it_{iteration}_raw_output_it_{iteration}.parquet"
            )
            chunk_files = glob.glob(search_pattern)

        if not chunk_files:
            print(f"Warning: No chunk files found with pattern: '{search_pattern}'")
            return pd.DataFrame()

        print(f"Found {len(chunk_files)} chunk files to combine.")
        combined_df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        print(f"Successfully combined into a DataFrame with {len(combined_df)} rows.")

        return combined_df



    def load_judged_output(self, iteration: int) -> pd.DataFrame:
        """
        Loads the judged output DataFrame for a given iteration.

        Args:
            iteration (int): The iteration number to load the judged outputs for.

        Returns:
            pd.DataFrame: The loaded DataFrame. Returns an empty DataFrame if the file doesn't exist.
        """
        file_path = self._get_filepath(iteration, "judged_output")
        
        if os.path.exists(file_path):
            print(f"Loading judged outputs from: {file_path}")
            return pd.read_parquet(file_path)
        else:
            print(f"Warning: Judged output file not found at {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()

    def get_raw_output_path(self, it_num: int) -> str:
            """
            Returns the directory path for raw outputs of a specific iteration.
            """
            path = os.path.join(self.experiment_dir, "raw_outputs", f"it_{int(it_num)}")
            os.makedirs(path, exist_ok=True) # Ensure the directory exists
            return path

    def load_gt_judged_output(self, iteration: int) -> pd.DataFrame:
        """
        Loads the judged output DataFrame for a given iteration.

        Args:
            iteration (int): The iteration number to load the judged outputs for.

        Returns:
            pd.DataFrame: The loaded DataFrame. Returns an empty DataFrame if the file doesn't exist.
        """
        file_path = self._get_filepath(iteration, "gt_judged_output")
        
        if os.path.exists(file_path):
            print(f"Loading judged outputs from: {file_path}")
            return pd.read_parquet(file_path)
        else:
            print(f"Warning: Judged output file not found at {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()



    def load_input_data(self, iteration: int, initial_file: str) -> pd.DataFrame:
        """Loads the input DataFrame for a given iteration."""
        if iteration == 0:
            input_path = initial_file
            print(f"Loading initial data from: {input_path}")

            if self.benchmark_name.lower() == "gpqa":
                # GPQA is loaded as a CSV and requires specific encoding handling
                try:
                    return pd.read_csv(input_path, dtype=str, encoding='cp1252', errors='ignore')
                except Exception as e:
                    sys.exit(f"❌ FATAL: Failed to load GPQA CSV/Shuffled data: {e}")
            elif input_path.endswith('.json'):
                return pd.read_json(input_path)
            elif input_path.endswith('.csv'):
                return pd.read_csv(input_path)
            elif input_path.endswith('.parquet'):
                return pd.read_parquet(input_path)
            else:
                sys.exit(f"Unsupported file format: {input_path}")
        elif iteration == 1:
            return pd.read_parquet(initial_file)
        else:
            input_path = self._get_filepath(iteration, "input")
            print(f"*****Loading data for iteration {iteration} from: {input_path}")
            if not os.path.exists(input_path):
                 print(f"Input file for iteration {iteration} not found. Concluding experiment.")
                 return pd.DataFrame()
            return pd.read_parquet(input_path)



    def save_ensembled_outputs(self, df: pd.DataFrame, iteration: int) -> str:
        """Saves the dataframe with the final ensembled answers."""
        output_path = self._get_filepath(iteration, "ensembled_output")
        df.to_parquet(output_path, index=False)
        print(f"Ensembled outputs for iteration {iteration} saved to: {output_path}")
        return output_path

    def save_raw_outputs_chunk(self, df: pd.DataFrame, iteration: int, chunk_name: str):
        """
        Saves the raw outputs for a single chunk to the correct directory.
        
        Args:
            df (pd.DataFrame): The DataFrame to save.
            iteration (int): The current iteration number.
            chunk_name (str): The base filename of the chunk (e.g., 'hle_test_chunk_1_of_4.parquet').
        """
        # output_dir = os.path.join(self.experiment_dir, self.prompt_strategy)
        # os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(self.experiment_dir, "chunks")
        
        # Get the part of the filename before the iteration number
        # This assumes the format is always '..._it_X...'
        filename_without_iteration = chunk_name.replace(f'_it{iteration}', '')
        
        # The filename is constructed by adding the raw_output_it_ suffix
        base_name = os.path.splitext(filename_without_iteration)[0]
        output_filename = f"{base_name}_raw_output_it_{iteration}.parquet"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Saving raw outputs for chunk to: {output_path}")
        df.to_parquet(output_path, index=False)

            
    def save_raw_outputs(self, df: pd.DataFrame, iteration: int) -> str:
        """Saves the dataframe with raw model outputs."""
        output_path = self._get_filepath(iteration, "raw_output")

        df.to_parquet(output_path, index=False)
        print(f"Raw model outputs for iteration {iteration} saved to: {output_path}")
        return output_path

    # Part of the DataManager class

    def load_gt_judged_output(self, iteration: int) -> pd.DataFrame:
        """
        Loads the DataFrame containing both blind and ground truth judgments for a given iteration.

        Args:
            iteration (int): The iteration number to load the file for.

        Returns:
            pd.DataFrame: The loaded DataFrame. Returns an empty DataFrame if the file doesn't exist.
        """
        file_path = self._get_filepath(iteration, "gt_judged_output")
        
        if os.path.exists(file_path):
            print(f"Loading ground truth judged outputs from: {file_path}")
            return pd.read_parquet(file_path)
        else:
            print(f"Warning: Ground truth judged output file not found at {file_path}. Returning empty DataFrame.")
            return pd.DataFrame()

    def save_judged_outputs(self, df: pd.DataFrame, iteration: int) -> str:
        """Saves the dataframe after judging is complete."""
        print("Saves the dataframe after judging is complete")
        output_path = self._get_filepath(iteration, "judged_output")
        print("output_path",output_path)
        df.to_parquet(output_path, index=False)
        print(f"Judged outputs for iteration {iteration} saved to: {output_path}")
        return output_path

    def prepare_next_iteration_inputs(self, judged_df: pd.DataFrame, current_iteration: int):
        """Filters incorrectly answered questions and saves them as input for the next iteration."""
        # Find all 'is_correct' columns for the current iteration
        is_correct_cols = [col for col in judged_df.columns if 'is_correct' in col and col.endswith(f"_{current_iteration}")]

        if not is_correct_cols:
            print(f"No 'is_correct' columns found for iteration {current_iteration}. Cannot prepare next input.")
            return

        solved_mask = judged_df[is_correct_cols].any(axis=1)
        next_iteration_df = judged_df[~solved_mask].copy()
        
        if next_iteration_df.empty:
            print("All questions have been answered correctly. No input for the next iteration.")
            return

        next_input_path = self._get_filepath(current_iteration + 1, "input")
        next_iteration_df.to_parquet(next_input_path, index=False)
        print(f"Input for iteration {current_iteration + 1} saved to: {next_input_path}")

