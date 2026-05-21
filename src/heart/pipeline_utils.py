# pipeline_utils.py

from __future__ import annotations

import glob
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from heart import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRuntime:
    """
    Shared runtime context so pipeline stages can reuse a single
    DataManager / experiment directory / chunk directory / logger setup.
    """

    data_manager: Any
    experiment_dir: str
    chunk_dir: str
    log_file: str


def setup_logging(instance_id: str, *, force: bool = True) -> str:
    """
    Configure logging to file and console.

    Returns the log file path. `force=True` prevents duplicate handlers when
    the initializer is called more than once in the same process.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(config.LOG_DIR, f"{instance_id}_experiment.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
        force=force,
    )
    logger.info("Logging initialized. Log file at: %s", log_file)
    return log_file


def initialize_runtime(
    *,
    instance_id: str,
    experiment_id: str,
    subject: str,
    model_name: str,
    prompt_strategy: str,
    data_split: str,
    benchmark_name: str,
) -> PipelineRuntime:
    """
    One shared initializer for the pipeline.

    This creates the DataManager once, sets up logging once, and prepares the
    standard experiment/chunk directories for the stage helpers.
    """
    from heart.data_manager import DataManager  # local import to avoid import cycles

    log_file = setup_logging(instance_id)

    data_manager = DataManager(
        experiment_id=experiment_id,
        subject=subject,
        model_name=model_name,
        prompt_strategy=prompt_strategy,
        data_split=data_split,
        benchmark_name=benchmark_name,
    )

    experiment_dir = data_manager.experiment_dir
    chunk_dir = os.path.join(experiment_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    logger.info("Runtime initialized.")
    logger.info("  experiment_dir: %s", experiment_dir)
    logger.info("  chunk_dir: %s", chunk_dir)

    return PipelineRuntime(
        data_manager=data_manager,
        experiment_dir=experiment_dir,
        chunk_dir=chunk_dir,
        log_file=log_file,
    )


def ensure_object_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    fill_value: str = "",
) -> pd.DataFrame:
    """
    Pre-create text columns as object dtype to avoid pandas FutureWarning when
    assigning strings into float-initialized columns.

    If a column already exists, it is coerced to object dtype.
    If it does not exist, it is created with object dtype.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series([fill_value] * len(df), dtype="object")
        else:
            if df[col].dtype != "object":
                df[col] = df[col].astype("object")
    return df


def _load_dataframe(input_file: str) -> pd.DataFrame:
    """Load a dataframe from parquet/json/csv."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}")

    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == ".parquet":
        return pd.read_parquet(input_file)
    if file_ext == ".json":
        return pd.read_json(input_file)
    if file_ext == ".csv":
        return pd.read_csv(input_file)

    raise ValueError(f"Unsupported file type '{file_ext}'. Use .parquet, .json, or .csv.")


def split_datafile(
    input_file: str,
    num_chunks: int,
    output_dir: str,
    iteration_number: int,
    exp_id: str,
    ensemble_suffix: bool = False,
) -> List[str]:
    """
    Split a Parquet/JSON/CSV file into the specified number of chunks
    and save each chunk as parquet.
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be > 0, got {num_chunks}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at {input_file}")

    logger.info("Loading data from %s", input_file)
    df = _load_dataframe(input_file)
    logger.info("Loaded %d total rows.", len(df))

    # Ensure every row carries a stable identity column that survives index=False saves
    # and ignore_index=True merges throughout the pipeline.
    # SimpleQA has no named ID column, so we fall back to the integer position in the
    # original file (e.g. "0", "1", ...).  For benchmarks that already have an id
    # column the first matching name is used so the value is human-readable.
    _Q_ID_COL = "_question_id"
    if _Q_ID_COL not in df.columns:
        _SOURCE_ID_COLS = ("id", "Record ID", "original_index")
        for _src in _SOURCE_ID_COLS:
            if _src in df.columns:
                df = df.copy()
                df[_Q_ID_COL] = df[_src].astype(str)
                logger.info("Assigned '%s' from existing column '%s'.", _Q_ID_COL, _src)
                break
        else:
            df = df.copy()
            df[_Q_ID_COL] = df.index.astype(str)
            logger.info(
                "No named ID column found — assigned '%s' from integer index (SimpleQA / unknown benchmark).",
                _Q_ID_COL,
            )

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Splitting data into %d chunks and saving to '%s'.", num_chunks, output_dir)

    # df_chunks = np.array_split(df, num_chunks)
    df_chunks = [df.iloc[idx].copy() for idx in np.array_split(np.arange(len(df)), num_chunks)]
    original_base_name = os.path.splitext(os.path.basename(input_file))[0]

    chunk_paths: List[str] = []
    for i, chunk in enumerate(df_chunks):
        if not ensemble_suffix:
            chunk_filename = f"{exp_id}_chunk_{i + 1}_of_{num_chunks}_input_it_{iteration_number}.parquet"
        else:
            chunk_filename = f"{exp_id}_chunk_{i + 1}_of_{num_chunks}_input_it_{iteration_number}_ensemble.parquet"

        chunk_filepath = os.path.join(output_dir, chunk_filename)
        chunk.to_parquet(chunk_filepath, index=False)
        chunk_paths.append(chunk_filepath)

        logger.info(
            "Saved chunk %d/%d (%d rows) from '%s' to %s",
            i + 1,
            num_chunks,
            len(chunk),
            original_base_name,
            chunk_filepath,
        )

    logger.info("Split data file complete.")
    return chunk_paths


def merge_raw_output_chunks(
    chunk_dir: str,
    final_output_path: str,
    it_num: int,
    is_selected: bool = False,
) -> None:
    """
    Find and merge raw_output_ chunks for a specific iteration into a single parquet file.
    """
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    base_pattern = f"raw_output_it_{it_num}_"
    if is_selected:
        base_pattern += "selected_"

    file_pattern = os.path.join(chunk_dir, f"{base_pattern}*.parquet")
    chunk_files = sorted(glob.glob(file_pattern))

    if not chunk_files:
        logger.warning("No raw_output chunk files found for iteration %s in %s", it_num, chunk_dir)
        return

    logger.info("Merging %d raw output chunks for iteration %s.", len(chunk_files), it_num)

    all_chunks: List[pd.DataFrame] = []
    for file_path in chunk_files:
        try:
            df_chunk = pd.read_parquet(file_path)
            all_chunks.append(df_chunk)
            logger.info("Loaded chunk %s with %d rows.", os.path.basename(file_path), len(df_chunk))
        except Exception as e:
            logger.exception("Error loading %s: %s", os.path.basename(file_path), e)

    if not all_chunks:
        logger.warning("No raw output chunks were successfully loaded. Skipping merge.")
        return

    merged_df = pd.concat(all_chunks, ignore_index=True)
    merged_df.to_parquet(final_output_path, index=False)
    logger.info(
        "Merged raw output saved to %s (%d rows).",
        final_output_path,
        len(merged_df),
    )


def _merge_prefixed_chunks(
    chunk_dir: str,
    final_output_path: str,
    it_num: int,
    prefix: str,
) -> None:
    """
    Generic helper for merging prefixed chunk files like gt_judged_ or blind_judged_.
    """
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    target_it_str = f"_it_{it_num}"

    chunk_files = sorted(
        f for f in os.listdir(chunk_dir)
        if f.startswith(prefix) and target_it_str in f and f.endswith(".parquet")
    )

    if not chunk_files:
        logger.warning("No %s chunk files found for iteration %s in %s", prefix, it_num, chunk_dir)
        return

    logger.info("Merging %d %s chunks for iteration %s.", len(chunk_files), prefix, it_num)

    all_chunks: List[pd.DataFrame] = []
    for file_name in chunk_files:
        file_path = os.path.join(chunk_dir, file_name)
        try:
            df_chunk = pd.read_parquet(file_path)
            all_chunks.append(df_chunk)
            logger.info("Loaded chunk %s with %d rows.", file_name, len(df_chunk))
        except Exception as e:
            logger.exception("Error loading %s: %s", file_name, e)

    if not all_chunks:
        logger.warning("No %s chunks were successfully loaded. Skipping merge.", prefix)
        return

    merged_df = pd.concat(all_chunks, ignore_index=True)
    merged_df.to_parquet(final_output_path, index=False)
    logger.info("Merged output saved to %s (%d rows).", final_output_path, len(merged_df))


def merge_gt_judged_chunks(chunk_dir: str, final_output_path: str, it_num: int) -> None:
    """Merge gt_judged_ output chunks for a specific iteration."""
    _merge_prefixed_chunks(
        chunk_dir=chunk_dir,
        final_output_path=final_output_path,
        it_num=it_num,
        prefix="gt_judged_",
    )


def merge_blind_judged_chunks(chunk_dir: str, final_output_path: str, it_num: int) -> None:
    """Merge blind_judged_ output chunks for a specific iteration."""
    _merge_prefixed_chunks(
        chunk_dir=chunk_dir,
        final_output_path=final_output_path,
        it_num=it_num,
        prefix="blind_judged_",
    )


def merge_chunks(
    chunk_dir: str,
    final_output_path: str,
    it_num: int,
    kind: str,
) -> None:
    """
    Convenience dispatcher for chunk merging.

    kind values:
      - 'raw'
      - 'gt_judged'
      - 'blind_judged'
    """
    kind = kind.lower().strip()

    if kind == "raw":
        merge_raw_output_chunks(chunk_dir, final_output_path, it_num)
    elif kind == "gt_judged":
        merge_gt_judged_chunks(chunk_dir, final_output_path, it_num)
    elif kind == "blind_judged":
        merge_blind_judged_chunks(chunk_dir, final_output_path, it_num)
    else:
        raise ValueError(f"Unsupported merge kind: {kind}")