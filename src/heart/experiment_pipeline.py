# experiment_pipeline.py

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import pandas as pd

from heart import config
from heart.data_manager import DataManager
from heart.experiment_runner import ExperimentRunner

from heart.pipeline_utils import (
    setup_logging,
    split_datafile,
    merge_raw_output_chunks,
    merge_gt_judged_chunks,
    merge_blind_judged_chunks,
)

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
else:
    LLM = Any
    SamplingParams = Any


def _load_vllm_classes():
    from vllm import LLM as VLLM, SamplingParams as VLLMSamplingParams

    return VLLM, VLLMSamplingParams


def _huggingface_token_kwargs() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"hf" + "_token": token} if token else {}


class ExperimentPipeline:
    """
    Thin orchestration layer for the HEART pipeline.

    This version removes duplicate helper implementations and uses
    pipeline_utils.py as the single source of truth for:
    - logging
    - splitting
    - merging
    """

    SUPPORTED_BENCHMARKS = {
        "OlympiadBench",
        "Livecodebench",
        "HLE",
        "SimpleQA",
        "SimpleQA_verified",
        "GPQA",
        "AIME2025",
        "AIME2024",
    }

    SUPPORTED_DATASETS = {"validation", "test"}

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ensembleBool = (
            self.args.workflow_type == "self_correcting_ensemble_gt"
            or self.args.workflow_type == "self_correcting_ensemble_gt_no_filter"
        )

    # ------------------------------------------------------------------
    # Validation / factories
    # ------------------------------------------------------------------

    def _validate_base_args(self) -> None:
        if self.args.benchmark not in self.SUPPORTED_BENCHMARKS:
            print("EXITING: INVALID BENCHMARK!")
            sys.exit(1)

        if self.args.dataset is None:
            sys.exit("ERROR: --dataset is required.")

        if self.args.dataset.lower() not in self.SUPPORTED_DATASETS:
            sys.exit(
                f"ERROR: dataset must be one of {sorted(self.SUPPORTED_DATASETS)}. "
                f"Got '{self.args.dataset}'."
            )

        if self.args.run_parallel and not self.args.stage:
            sys.exit("ERROR: --stage is required when using --run-parallel.")

        if self.args.id is None or self.args.model is None or self.args.prompt_strategy is None:
            sys.exit("ERROR: --id, --model, and --prompt_strategy are required.")

    def _validate_parallel_run_args(self) -> None:
        if not self.args.input:
            sys.exit("ERROR: --input must specify the original, unsplit data file when using --run-parallel.")

        if self.args.start_it is None:
            sys.exit("ERROR: --start_it is required for parallel runs.")

    def _new_data_manager(self) -> DataManager:
        return DataManager(
            experiment_id=self.args.id,
            subject=self.args.subject,
            model_name=self.args.model,
            prompt_strategy=self.args.prompt_strategy,
            data_split=self.args.dataset,
            benchmark_name=self.args.benchmark,
        )

    def _new_runner(
        self,
        *,
        data_manager: Optional[DataManager] = None,
        ensemble_bool: Optional[bool] = None,
        selector_bool: bool = False,
    ) -> ExperimentRunner:
        if ensemble_bool is None:
            ensemble_bool = self.ensembleBool

        if data_manager is None:
            data_manager = self._new_data_manager()

        return ExperimentRunner(
            benchmark_name=self.args.benchmark,
            subject=self.args.subject,
            model_name=self.args.model,
            prompt_strategy=self.args.prompt_strategy,
            data_manager=data_manager,
            dataset_split=self.args.dataset,
            prepend_str=self.args.prepend_string,
            base_completion_prefix=getattr(self.args, "base_completion_prefix", ""),
            ensembleBool=ensemble_bool,
            selectorBool=selector_bool,
        )

    # ------------------------------------------------------------------
    # vLLM helpers
    # ------------------------------------------------------------------

    def _build_gemma_vllm(self) -> Tuple[LLM, SamplingParams, int]:
        LLM, SamplingParams = _load_vllm_classes()
        model_name = self.args.model.lower()
        is_explicit_model_path = "/" in self.args.model
        is_base = (
            "base" in model_name
            or "-pt" in model_name
            or (is_explicit_model_path and not any(x in model_name for x in ["instruct", "-it", "r1-distill"]))
        )

        if "1b" in model_name:
            max_model_len = 4096
            vllm_init_kwargs = {
                "model": self.args.model if is_explicit_model_path else ("google/gemma-3-1b-pt" if is_base else "google/gemma-3-1b-it"),
                "trust_remote_code": True,
                **_huggingface_token_kwargs(),
                "max_model_len": 4096,
                "tensor_parallel_size": 4,
                "gpu_memory_utilization": 0.90,
                "dtype": "bfloat16",
            }

        elif "4b" in model_name:
            max_model_len = 32768
            vllm_init_kwargs = {
                "model": self.args.model if is_explicit_model_path else ("google/gemma-3-4b-pt" if is_base else "google/gemma-3-4b-it"),
                "trust_remote_code": True,
                **_huggingface_token_kwargs(),
                "max_model_len": 32768,
                "tensor_parallel_size": 4,
                "gpu_memory_utilization": 0.90,
                "dtype": "bfloat16",
            }

        elif "12b" in model_name:
            max_model_len = 16384
            vllm_init_kwargs = {
                "model": self.args.model if is_explicit_model_path else ("google/gemma-3-12b-pt" if is_base else "google/gemma-3-12b-it"),
                "trust_remote_code": True,
                **_huggingface_token_kwargs(),
                "max_model_len": 16384,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 512,
                "tensor_parallel_size": 4,
                "gpu_memory_utilization": 0.85,
                "dtype": "bfloat16",
                # Disable image encoder — text-only task, avoids ~80s multimodal init overhead
                "limit_mm_per_prompt": {"image": 0},
            }
        else:
            raise ValueError(f"Unsupported Gemma model variant: {self.args.model}")

        if getattr(self.args, 'get_logprobs', False):
            vllm_init_kwargs["max_logprobs"] = 50

        vllm_engine = LLM(**vllm_init_kwargs)

        max_output_tokens = int(os.environ.get("HEART_VLLM_MAX_OUTPUT_TOKENS", "8192"))
        max_input_tokens = max_model_len - max_output_tokens - 10
        print("MAX_INPUT_TOKENS:", max_input_tokens)

        # Base models don't use turn markers — use generic EOS tokens only.
        stop_tokens = ["<eos>", "<|endoftext|>"] if is_base else ["<end_of_turn>", "<eos>", "<|endoftext|>"]
        extra_stop_tokens = os.environ.get("HEART_VLLM_EXTRA_STOP_TOKENS", "").strip()
        if extra_stop_tokens:
            stop_tokens = stop_tokens + [tok for tok in extra_stop_tokens.split("|||") if tok]

        _logprobs = 50 if getattr(self.args, 'get_logprobs', False) else None
        vllm_sampling_params = SamplingParams(
            frequency_penalty=0,
            presence_penalty=0,
            temperature=self.args.temperature,
            top_p=float(os.environ.get("HEART_VLLM_TOP_P", "0.95")),
            repetition_penalty=float(os.environ.get("HEART_VLLM_REPETITION_PENALTY", "1.0")),
            max_tokens=max_output_tokens,
            truncate_prompt_tokens=max_input_tokens,
            logprobs=_logprobs,
            prompt_logprobs=_logprobs,
            stop=stop_tokens,
        )

        return vllm_engine, vllm_sampling_params, max_model_len

    def _build_qwen_vllm(self) -> Tuple[LLM, SamplingParams]:
        LLM, SamplingParams = _load_vllm_classes()
        # Use the actual model name from config — never hardcode the path.
        # args.model keeps the original '/' form (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
        # which is what vLLM / HuggingFace requires.
        model_path = self.args.model
        model_lower = model_path.lower()

        # Scale max_model_len and memory budget by parameter count.
        is_reasoning = "deepseek-r1-distill" in model_lower
        if "72b" in model_lower or "70b" in model_lower:
            # 72/70 B weights ~144 GB in bf16 across 4×48 GB GPUs.
            # max_model_len covers full prompt + output; keep utilization high.
            max_model_len = 16384
            extra_kwargs = {
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 8,
            }
        elif "14b" in model_lower:
            max_model_len = 32768
            extra_kwargs = {}
        elif "32b" in model_lower:
            max_model_len = 16384
            extra_kwargs = {}
        else:  # 7B default
            max_model_len = 32768
            extra_kwargs = {}

        vllm_engine = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=max_model_len,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            enable_prefix_caching=True,
            trust_remote_code=True,
            **extra_kwargs,
        )

        # A Qwen model is "base" if it has no instruct/reasoning marker in its path.
        is_base = not any(x in model_lower for x in ["instruct", "r1-distill"])
        if is_reasoning:
            stop_tokens = ["<｜end▁of▁sentence｜>"]
        elif is_base:
            # Base models: no chat-turn markers, stop on EOS only.
            stop_tokens = ["<|endoftext|>"]
        else:
            # Standard instruction models — let the chat template handle EOS.
            stop_tokens = ["<|im_end|>", "<|endoftext|>"]

        # Reserve half the context for output so responses are never truncated.
        # truncate_prompt_tokens clips over-long inputs rather than failing.
        max_output_tokens = max_model_len // 2          # e.g. 8192 for a 16384-token window
        max_input_tokens = max_model_len - max_output_tokens - 10

        vllm_sampling_params = SamplingParams(
            n=1,
            max_tokens=max_output_tokens,
            temperature=self.args.temperature,
            top_p=0.95,
            stop=stop_tokens,
            truncate_prompt_tokens=max_input_tokens,
        )

        return vllm_engine, vllm_sampling_params

    def _build_mistral_vllm(self) -> Tuple[LLM, SamplingParams]:
        LLM, SamplingParams = _load_vllm_classes()
        is_base = "base" in self.args.model.lower()
        model_path = (
            "mistralai/Mistral-Small-3.1-24B-Base-2503"
            if is_base
            else "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        )
        max_model_len = 16384
        vllm_engine = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            dtype="bfloat16",
            enforce_eager=True,
            **_huggingface_token_kwargs(),
        )
        max_output_tokens = 8192
        max_input_tokens = max_model_len - max_output_tokens - 10
        print("MAX_INPUT_TOKENS:", max_input_tokens)
        # Base models don't use instruction markers — use generic EOS only.
        stop_tokens = ["</s>", "<unk>"] if is_base else ["</s>", "[INST]", "[/INST]"]
        vllm_sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=0.95,
            max_tokens=max_output_tokens,
            truncate_prompt_tokens=max_input_tokens,
            stop=stop_tokens,
        )
        return vllm_engine, vllm_sampling_params

    def _build_llama_vllm(self) -> Tuple[LLM, SamplingParams]:
        LLM, SamplingParams = _load_vllm_classes()
        # Llama 3.x 70B: ~142 GB in bf16 across 4×48 GB GPUs.
        # Use the full HF path passed in args.model so any Llama variant works.
        model_path = self.args.model
        max_model_len = 16384
        vllm_engine = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            trust_remote_code=True,
            **_huggingface_token_kwargs(),
        )
        max_output_tokens = max_model_len // 2   # 8192 — reserves half the window for output
        max_input_tokens = max_model_len - max_output_tokens - 10
        print("MAX_INPUT_TOKENS:", max_input_tokens)
        # Base Llama models don't use the instruction end-of-turn token.
        is_base = "instruct" not in self.args.model.lower()
        stop_tokens = ["<|end_of_text|>"] if is_base else ["<|eot_id|>", "<|end_of_text|>"]
        vllm_sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=0.95,
            max_tokens=max_output_tokens,
            truncate_prompt_tokens=max_input_tokens,
            stop=stop_tokens,
        )
        return vllm_engine, vllm_sampling_params

    def _build_vllm_for_generation(self):
        model_lower = self.args.model.lower()

        if "gemma" in model_lower:
            print("✅ Gemma model detected. Running generation sequentially with a single vLLM engine.")
            engine, sampling_params, _ = self._build_gemma_vllm()
            return engine, sampling_params

        if "qwen" in model_lower or "deepseek-r1-distill" in model_lower:
            print(f"✅ Reasoning model detected ({self.args.model}). Optimizing context for long Chain-of-Thought.")
            engine, sampling_params = self._build_qwen_vllm()
            return engine, sampling_params

        if "mistral" in model_lower:
            print(f"✅ Mistral model detected ({self.args.model}). Loading via vLLM.")
            engine, sampling_params = self._build_mistral_vllm()
            return engine, sampling_params

        if "llama" in model_lower:
            print(f"✅ Llama model detected ({self.args.model}). Loading via vLLM.")
            engine, sampling_params = self._build_llama_vllm()
            return engine, sampling_params

        return None, None

    def _build_qwen_judge_vllm(self) -> Tuple[LLM, SamplingParams]:
        LLM, SamplingParams = _load_vllm_classes()
        """Load Qwen 2.5 72B Instruct locally as an LLM judge (no external API)."""
        judge_engine = LLM(
            model="Qwen/Qwen2.5-72B-Instruct",
            tokenizer="Qwen/Qwen2.5-72B-Instruct",
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=8192,
            enforce_eager=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            disable_custom_all_reduce=True,
        )
        # Greedy + short output: judge only needs "CORRECT" or "INCORRECT".
        judge_params = SamplingParams(
            temperature=0.0,
            max_tokens=64,
            stop=["<|im_end|>", "\n\n"],
        )
        return judge_engine, judge_params

    def _run_chunked_generation_with_vllm(
        self,
        runner: ExperimentRunner,
        chunk_paths: List[str],
        parallel_k: int,
    ) -> None:
        for i, path in enumerate(chunk_paths):
            print(f"\n--- Processing chunk {i + 1}/{len(chunk_paths)} ---")

            if parallel_k > 1:
                print(f"📡 Parallel Sampling Mode: Generating {parallel_k} samples per prompt.")
                runner.vllm_sampling_params.n = parallel_k
                if runner.vllm_sampling_params.temperature == 0:
                    runner.vllm_sampling_params.temperature = 0.7

                runner.run_parallel_generation_stage(
                    it_num=int(self.args.start_it),
                    input_file=path,
                    k=parallel_k,
                )
            else:
                runner.run_generation_stage(
                    it_num=int(self.args.start_it),
                    input_file=path,
                )

        print("\n✅ All chunks processed sequentially with vLLM.")
        if runner.use_vllm:
            runner.cleanup_vllm_resources()

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def run_generate(self) -> None:
        self._validate_parallel_run_args()

        if "no_think" in self.args.id and "claude" in self.args.id:
            config.CLAUDE_THINK_BOOL = False
            print("CLAUDE THINKING IS TURNED OFF")
        if "no_think" in self.args.id and "flash" in self.args.id:
            config.GEMINI_THINK_BOOL = False
            print("GEMINI 2.5 FLASH THINKING IS TURNED OFF")
        if "no_think" in self.args.id and "pro" in self.args.id:
            config.GEMINI_THINK_BOOL = False
            print("GEMINI 2.5 PRO THINKING IS TURNED OFF")

        data_manager = self._new_data_manager()
        runner = self._new_runner(data_manager=data_manager)
        
        if self.args.get_logprobs:
            print("✅ Log probability collection ENABLED.")
            runner.collectLogsBool = True

        print("🚀 Splitting input file into chunks for parallel processing...")
        chunk_output_dir = os.path.join(data_manager.experiment_dir, "chunks")
        print("Chunk output dir: ", chunk_output_dir)
        os.makedirs(chunk_output_dir, exist_ok=True)

        print("Number of chunks", self.args.num_chunks)
        chunk_paths = split_datafile(
            self.args.input,
            self.args.num_chunks,
            chunk_output_dir,
            int(self.args.start_it),
            self.args.id,
        )
        print("chunk_paths:", chunk_paths)
        print("✅ Splitting complete. Launching parallel generation jobs for chunks...")

        vllm_engine, vllm_sampling_params = self._build_vllm_for_generation()

        if vllm_engine is not None:
            runner.vllm_engine = vllm_engine
            runner.vllm_sampling_params = vllm_sampling_params
            runner.use_vllm = True
            self._run_chunked_generation_with_vllm(runner, chunk_paths, self.args.parallel_k)
            return

        # API model best-of-N: run the API k times per question and accumulate _sample_i columns.
        # For vLLM models this is handled above via SamplingParams(n=k); for API models we must
        # call the API k times sequentially (respects rate limits, mirrors "10 separate calls"
        # for models like Gemini that don't support n>1 natively).
        if self.args.parallel_k > 1:
            print(f"🔁 API Best-of-N mode: {self.args.parallel_k} samples per question via sequential API calls.")
            for path in chunk_paths:
                runner.run_api_best_of_n_generation_stage(
                    it_num=int(self.args.start_it),
                    input_file=path,
                    k=self.args.parallel_k,
                )
            print("✅ API Best-of-N generation complete.")
            return

        tasks_to_run = [(runner, int(self.args.start_it), path) for path in chunk_paths]
        with ThreadPoolExecutor(max_workers=self.args.num_chunks) as executor:
            futures = {
                executor.submit(runner.run_generation_stage, it_num=it, input_file=inp_file): (it, inp_file)
                for runner, it, inp_file in tasks_to_run
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ A parallel generation task failed with an error: {e}")
                    raise

        print("🎉 All parallel generation jobs have finished.")

    def run_gt_judging(self) -> None:
        self._validate_parallel_run_args()

        data_manager = self._new_data_manager()
        runner = self._new_runner(data_manager=data_manager)

        print("Experiment dir (data manager)", data_manager.experiment_dir)
        raw_output_chunk_dir = data_manager.get_raw_output_path(int(self.args.start_it))
        combined_raw_output_file = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{self.args.start_it}_raw_output.parquet"
        )

        print("\n🚀 Combining raw output chunks into a single file for judging...")
        merge_raw_output_chunks(raw_output_chunk_dir, combined_raw_output_file, int(self.args.start_it))
        print("✅ Raw output chunks successfully combined.")
        print(f"FULL OUTPUT PATH: {combined_raw_output_file}")

        print("\n🚀 Splitting combined raw output file for parallel GT Judging...")
        chunk_output_dir = os.path.join(data_manager.experiment_dir, "chunks")
        chunk_paths = split_datafile(
            combined_raw_output_file,
            self.args.num_chunks,
            chunk_output_dir,
            int(self.args.start_it),
            self.args.id,
        )

        use_rule_based = getattr(self.args, "rule_based_judging", False)
        # vllm_llm_as_judge: use Qwen 2.5 72B Instruct locally instead of Gemini API.
        # Ignored when rule_based_judging is active (rule-based takes priority).
        use_vllm_judge = getattr(self.args, "vllm_llm_as_judge", False) and not use_rule_based

        if use_rule_based:
            print("⚡ Rule-based judging enabled — skipping LLM API calls.")
        elif use_vllm_judge:
            print("🤖 vLLM LLM-as-judge enabled — loading Qwen/Qwen2.5-72B-Instruct locally.")

        # Load the local judge engine once; reuse across all chunks then clean up.
        judge_engine, judge_params = None, None
        if use_vllm_judge:
            judge_engine, judge_params = self._build_qwen_judge_vllm()

        try:
            for path in chunk_paths:
                try:
                    if use_rule_based:
                        runner.run_ground_truth_judging_stage_rule_based(
                            it_num=self.args.start_it,
                            input_file=path,
                            k=self.args.parallel_k,
                        )
                    elif use_vllm_judge:
                        runner.run_ground_truth_judging_stage_vllm(
                            it_num=self.args.start_it,
                            input_file=path,
                            k=self.args.parallel_k,
                            judge_engine=judge_engine,
                            judge_params=judge_params,
                        )
                    elif self.args.parallel_k > 1:
                        print(f"🔍 Parallel Judging Mode for HEART: Evaluating {self.args.parallel_k} samples.")
                        runner.run_ground_truth_judging_stage_parallel(
                            it_num=self.args.start_it,
                            input_file=path,
                            k=self.args.parallel_k,
                        )
                    else:
                        runner.run_ground_truth_judging_stage(
                            it_num=self.args.start_it,
                            input_file=path,
                        )
                except Exception as e:
                    print(f"❌ A GT Judging chunk failed with an error: {e}")
        finally:
            # Always free the judge engine regardless of errors.
            if judge_engine is not None:
                import gc
                del judge_engine
                gc.collect()
                print("🧹 vLLM judge engine cleaned up.")

        print("🎉 All parallel GT Judging jobs have finished.")

        final_judged_output_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{self.args.start_it}_gt_judged_output.parquet"
        )
        merge_gt_judged_chunks(chunk_output_dir, final_judged_output_path, int(self.args.start_it))
        print("🎉 All GT Judged chunks have been merged into a final output file.")

        print("\n🚀 Performing 'any-correct' evaluation on the merged results...")
        try:
            merged_judged_df = pd.read_parquet(final_judged_output_path)
            final_df_with_any_correct = runner.run_any_correct_evaluation(
                merged_judged_df,
                int(self.args.start_it),
            )
            final_df_with_any_correct.to_parquet(final_judged_output_path, index=False)
            print(f"✅ Successfully added 'any_correct' column and saved final results to: {final_judged_output_path}")
        except FileNotFoundError:
            print(
                f"Error: Could not find merged file at {final_judged_output_path} "
                f"to perform 'any-correct' evaluation."
            )
        except Exception as e:
            print(f"An error occurred during 'any-correct' evaluation: {e}")

    def run_filter(self) -> None:
        print("\n🚀 Starting Filter Stage: Identifying incorrect rows for the next iteration...")

        if not all([self.args.id, self.args.model, self.args.subject, self.args.prompt_strategy, self.args.start_it is not None]):
            sys.exit("ERROR: --id, --model, --subject, --prompt_strategy, and --start_it are required for filter stage.")

        data_manager = self._new_data_manager()
        current_iteration = int(self.args.start_it)
        next_iteration = current_iteration + 1

        input_file_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_gt_judged_output.parquet"
        )
        print("input file path:", input_file_path)

        output_file_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_incorrect_rows_from_it{current_iteration}_for_it{next_iteration}.parquet"
        )

        try:
            judged_df = pd.read_parquet(input_file_path)
            print(f"  - Successfully loaded {len(judged_df)} rows from: {input_file_path}")

            is_bestofn = self.args.prompt_strategy == "bestofn"

            if is_bestofn:
                print("  - Mode: Best-of-N")
                print(judged_df.columns.to_list())

                eval_cols = [
                    c
                    for c in judged_df.columns
                    if c.startswith("gt_judged_correct_eval_prompt_") and f"_it_{current_iteration}" in c
                ]

                if not eval_cols:
                    sys.exit(f"FATAL: No best-of-N judgment columns found for iteration {current_iteration}.")

                judged_df[f"any_correct_it_{current_iteration}"] = judged_df[eval_cols].fillna(False).any(axis=1)

                correctness_col = f"any_correct_it_{current_iteration}"
                filter_type = "Best-of-N Any-Correct"

                any_correct_count = judged_df[correctness_col].sum()
                print(f"  - [Stats] Pass@N (Any-Correct): {(any_correct_count / len(judged_df)) * 100:.2f}%")

                incorrect_rows_df = judged_df[judged_df[correctness_col] == False].copy()

            elif self.args.parallel_k > 1:
                print(f"  - Mode: Parallel Sampling (k={self.args.parallel_k})")

                eval_cols = [
                    c
                    for c in judged_df.columns
                    if c.startswith("gt_judged_correct_eval") and f"_it_{current_iteration}_" in c
                ]

                print(eval_cols)
                if not eval_cols:
                    sys.exit(f"FATAL: No sample judgment columns found for iteration {current_iteration}.")

                any_correct_count = judged_df[eval_cols].fillna(False).any(axis=1).sum()
                avg_consistency = judged_df[eval_cols].fillna(False).sum(axis=1).mean()

                print(
                    f"  - [Stats] Pass@{self.args.parallel_k} (Any-Correct) would be: "
                    f"{(any_correct_count / len(judged_df)) * 100:.2f}%"
                )
                print(
                    f"  - [Stats] Average samples correct per problem: "
                    f"{avg_consistency:.2f}/{self.args.parallel_k}"
                )

                any_correct_col = f"any_correct_it_{current_iteration}"

                # Always recompute from eval_cols so we're never relying on a stale
                # any_correct column that may have been written by an earlier broken run.
                any_correct_series = judged_df[eval_cols].fillna(False).any(axis=1)
                filter_type = f"Pass@{self.args.parallel_k} Refinement (recomputed from eval_cols)"
                incorrect_rows_df = judged_df[~any_correct_series].copy()

                total_rows = len(judged_df)
                incorrect_count = len(incorrect_rows_df)
                correct_count = total_rows - incorrect_count
                accuracy = (correct_count / total_rows) * 100 if total_rows > 0 else 0

                print("\n--- Filtering Summary ---")
                print(f"  - Filter Logic: {filter_type}")
                print(f"  - Iteration Evaluated: {current_iteration}")
                print(f"  - Total Rows: {total_rows}")
                print(f"  - Correct Rows: {correct_count} ({accuracy:.2f}%)")
                print(f"  - Incorrect Rows: {incorrect_count}")
                print("-------------------------")

                if incorrect_count > 0:
                    incorrect_rows_df.to_parquet(output_file_path, index=False)
                    print(f"\n✅ Saved {incorrect_count} incorrect rows for HEART Iteration {next_iteration}.")
                    print(f"  - Output file: {output_file_path}")
                else:
                    print("\n🎉 Congratulations! All rows solved. Terminating workflow.")
                    sys.exit(5)

            else:
                correctness_col = f"any_correct_it_{current_iteration}"
                if correctness_col not in judged_df.columns:
                    sys.exit(
                        f"FATAL: The required correctness column '{correctness_col}' was not found in the input file. "
                        f"Please run the 'gt_judging' stage first."
                    )

                incorrect_rows_df = judged_df[judged_df[correctness_col] == False].copy()

                total_rows = len(judged_df)
                incorrect_count = len(incorrect_rows_df)
                correct_count = total_rows - incorrect_count
                accuracy = (correct_count / total_rows) * 100 if total_rows > 0 else 0

                print("\n--- Filtering Summary ---")
                print(f"  - Iteration Evaluated: {current_iteration}")
                print(f"  - Total Rows: {total_rows}")
                print(f"  - Correct Rows: {correct_count} ({accuracy:.2f}%)")
                print(f"  - Incorrect Rows: {incorrect_count}")
                print("-------------------------")

                if incorrect_count > 0:
                    incorrect_rows_df.to_parquet(output_file_path, index=False)
                    print(f"\n✅ Saved {incorrect_count} incorrect rows to be used as input for iteration {next_iteration}.")
                    print(f"  - Output file: {output_file_path}")
                else:
                    print("\n🎉 Congratulations! All rows were answered correctly. Terminating workflow.")
                    sys.exit(5)

        except FileNotFoundError:
            print(
                f"Error: Input file not found at '{input_file_path}'. Please ensure the 'gt_judging' stage "
                f"for iteration {current_iteration} has been completed successfully."
            )
        except Exception as e:
            print(f"An unexpected error occurred during the filter stage: {e}")

    def run_select(self) -> None:
        data_manager = self._new_data_manager()
        runner = self._new_runner(data_manager=data_manager,selector_bool=True)

        current_iteration = int(self.args.start_it)

        raw_output_chunk_dir = data_manager.get_raw_output_path(current_iteration)
        combined_raw_output_file = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_raw_output.parquet"
        )
        print("\n🚀 Combining raw output chunks into a single file for selection...")
        merge_raw_output_chunks(raw_output_chunk_dir, combined_raw_output_file, current_iteration)
        print("✅ Raw output chunks successfully combined.")

        print("\n🚀 Splitting combined raw output file for parallel selection...")
        chunk_output_dir = os.path.join(data_manager.experiment_dir, "chunks")
        num_chunks = runner.semaphore_limit
        chunk_paths = split_datafile(
            combined_raw_output_file,
            num_chunks,
            chunk_output_dir,
            current_iteration,
            self.args.id,
        )
        print("✅ Splitting complete. Launching parallel selection jobs for chunks...")

        tasks_to_run = [(runner, current_iteration, path, self.args) for path in chunk_paths]
        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = {
                executor.submit(runner.run_selection_stage, it_num=it, input_file=inp_file, args=self.args): (it, inp_file)
                for runner, it, inp_file, args in tasks_to_run
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ A selection task failed with an error: {e}")
                    raise RuntimeError(f"Task failed: {e}")

        print("\n🎉 All selection jobs have finished.")

        final_selected_output_path = os.path.join(
            data_manager.experiment_dir,
            f"{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_selected_output.parquet",
        )

        selected_raw_chunk_dir = os.path.join(data_manager.experiment_dir, "raw_outputs", f"it_{current_iteration}")
        merge_raw_output_chunks(
            selected_raw_chunk_dir,
            final_selected_output_path,
            current_iteration,
            is_selected=True,
        )
        print("🎉 All selected chunks have been merged into a final output file.")
        sys.exit(1)

    def run_gt_judging_selected(self) -> None:
        print("\n🚀 Starting GT judging for selected answers...")

        if not all([self.args.id, self.args.model, self.args.prompt_strategy, self.args.start_it is not None]):
            sys.exit("ERROR: --id, --model, --prompt_strategy, and --start_it are required for this stage.")

        data_manager = self._new_data_manager()
        runner = self._new_runner(data_manager=data_manager,selector_bool=True)

        current_iteration = int(self.args.start_it)
        input_file_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_selected_output.parquet"
        )

        print(f"🚀 Splitting selected output file for parallel GT judging...")
        chunk_output_dir = os.path.join(data_manager.experiment_dir, "chunks")
        num_chunks = runner.semaphore_limit
        chunk_paths = split_datafile(
            input_file_path,
            num_chunks,
            chunk_output_dir,
            current_iteration,
            self.args.id,
        )
        print("✅ Splitting complete. Launching parallel GT judging jobs...")

        tasks_to_run = [(current_iteration, path) for path in chunk_paths]

        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            futures = {
                executor.submit(runner.run_gt_judging_selected, it_num=it, input_file=path): path
                for it, path in tasks_to_run
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ A GT judging task for selected answers failed with an error: {e}")
                    raise RuntimeError(f"Task failed: {e}")

        print("\n🎉 All GT judging jobs for selected answers have finished.")

        final_judged_output_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_gt_judged_selected_output.parquet"
        )

        merge_gt_judged_chunks(chunk_output_dir, final_judged_output_path, current_iteration)
        print("🎉 All judged chunks have been merged.")

    def run_selector_no_filter_prep_next(self) -> None:
        print("\n🚀 Starting Prep Stage for SELECTOR (No Filter)...")

        if not all([self.args.id, self.args.model, self.args.subject, self.args.prompt_strategy, self.args.start_it is not None]):
            sys.exit("ERROR: --id, --model, --subject, --prompt_strategy, and --start_it are required for this stage.")

        data_manager = self._new_data_manager()

        current_iteration = int(self.args.start_it)

        input_file_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_gt_judged_selected_output.parquet"
        )
        print("input file path:", input_file_path)

        output_file_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_gt_judged_selected_output.parquet"
        )

        try:
            judged_df = pd.read_parquet(input_file_path)
            print(f"  - Successfully loaded {len(judged_df)} rows from: {input_file_path}")

            correctness_col = f"gt_judged_selected_correct_eval_it_{current_iteration}"
            if correctness_col not in judged_df.columns:
                sys.exit(
                    f"FATAL: The required correctness column '{correctness_col}' was not found. "
                    f"Please run 'gt_judging_selected' first."
                )

            total_rows = len(judged_df)
            correct_count = judged_df[correctness_col].sum()
            incorrect_count = total_rows - correct_count
            accuracy = (correct_count / total_rows) * 100 if total_rows > 0 else 0

            print("\n--- Selector GT No Filter Summary ---")
            print(f"  - Iteration Evaluated: {current_iteration}")
            print(f"  - Total Rows: {total_rows}")
            print(f"  - Correct Rows: {correct_count} ({accuracy:.2f}%)")
            print(f"  - Incorrect Rows: {incorrect_count}")
            print("-------------------------")
            print(f"✅ Prep stage complete. Next iteration will use: {output_file_path}")

        except FileNotFoundError:
            print(
                f"Error: Input file not found at '{input_file_path}'. Please ensure the "
                f"'gt_judging_selected' stage for iteration {current_iteration} has completed."
            )
        except Exception as e:
            print(f"An unexpected error occurred during the filter stage: {e}")

    def run_mad(self) -> None:
        print("\n🚀 Starting MAD stage on GPQA Diamond...")

        if not self.args.input:
            sys.exit("Error: --input must specify the original, unsplit data file when using --run-parallel.")

        if not all([self.args.id, self.args.model, self.args.subject, self.args.prompt_strategy, self.args.start_it is not None]):
            sys.exit("The --id, --model, --subject, --prompt_strategy, and --start_it arguments are required for the MAD stage.")

        current_iteration = int(self.args.start_it)
        if current_iteration != 0:
            sys.exit("MAD baseline branch is currently set up for iteration 0 only.")

        data_manager = self._new_data_manager()
        runner = self._new_runner(data_manager=data_manager)

        vllm_engine, vllm_sampling_params = self._build_vllm_for_generation()
        if vllm_engine is not None:
            runner.vllm_engine = vllm_engine
            runner.vllm_sampling_params = vllm_sampling_params
            runner.use_vllm = True

        if runner.use_vllm and (runner.vllm_engine is None or runner.vllm_sampling_params is None):
            sys.exit(
                "MAD requested a local vLLM generation model, but no vLLM engine was initialized. "
                f"Model was '{self.args.model}'. For Gemma12B instruct, use --model gemma12b "
                "or --model google/gemma-3-12b-it."
            )

        mad_agents = getattr(self.args, "mad_agents", 5)
        mad_rounds = getattr(self.args, "mad_rounds", 8)

        print(f"Using MAD budget = {mad_agents} agents x {mad_rounds} rounds = {mad_agents * mad_rounds} calls/question.")

        print("🚀 Splitting GPQA input into chunks for MAD...")
        chunk_output_dir = os.path.join(data_manager.experiment_dir, "chunks")
        os.makedirs(chunk_output_dir, exist_ok=True)

        chunk_paths = split_datafile(
            self.args.input,
            self.args.num_chunks,
            chunk_output_dir,
            self.args.start_it,
            self.args.id,
        )

        print("chunk_paths:", chunk_paths)
        print("✅ Splitting complete. Launching MAD chunk jobs...")

        tasks_to_run = [(runner, current_iteration, path, mad_agents, mad_rounds) for path in chunk_paths]

        try:
            if runner.use_vllm:
                for _, it, path, n_agents, n_rounds in tasks_to_run:
                    runner.run_mad_stage(
                        it_num=it,
                        input_file=path,
                        num_agents=n_agents,
                        num_rounds=n_rounds,
                    )
            else:
                with ThreadPoolExecutor(max_workers=self.args.num_chunks) as executor:
                    futures = {
                        executor.submit(
                            runner.run_mad_stage,
                            it_num=it,
                            input_file=path,
                            num_agents=n_agents,
                            num_rounds=n_rounds,
                        ): path
                        for runner, it, path, n_agents, n_rounds in tasks_to_run
                    }

                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"❌ A MAD chunk failed with an error: {e}")
                            raise
        finally:
            if runner.use_vllm:
                runner.cleanup_vllm_resources()

        print("🎉 All MAD chunk jobs have finished.")

        raw_output_chunk_dir = data_manager.get_raw_output_path(current_iteration)
        final_output_path = (
            f"{data_manager.experiment_dir}{self.args.subject.capitalize()}_{self.args.model_safe}_it{current_iteration}_mad_output.parquet"
        )

        merge_raw_output_chunks(raw_output_chunk_dir, final_output_path, current_iteration)
        print(f"✅ MAD merged output saved to: {final_output_path}")

        merged_df = pd.read_parquet(final_output_path)
        runner.report_mad_results(merged_df, current_iteration)

    def run_gt_mad(self) -> None:
        print("\n🚀 Starting GT judging for MAD output...")

        if not all([self.args.id, self.args.model, self.args.subject, self.args.prompt_strategy, self.args.start_it is not None]):
            sys.exit("The --id, --model, --subject, --prompt_strategy, and --start_it arguments are required for MAD GT judging.")

        data_manager = self._new_data_manager()
        runner = self._new_runner(data_manager=data_manager)

        mad_output_file = (
            f"{data_manager.experiment_dir}"
            f"{self.args.subject.capitalize()}_{self.args.model_safe}_it{self.args.start_it}_mad_output.parquet"
        )

        if not os.path.exists(mad_output_file):
            sys.exit(f"MAD output file not found: {mad_output_file}")

        judged_df = runner.run_mad_ground_truth_judging_stage(
            it_num=int(self.args.start_it),
            input_file=mad_output_file,
            response_col=f"mad_final_answer_it_{int(self.args.start_it)}",
        )

        if judged_df is not None:
            out_file = (
                f"{data_manager.experiment_dir}"
                f"{self.args.subject.capitalize()}_{self.args.model_safe}_it{self.args.start_it}_mad_gt_judged_output.parquet"
            )
            judged_df.to_parquet(out_file, index=False)
            print(f"✅ Saved MAD GT-judged output to: {out_file}")

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def run(self) -> None:
        setup_logging(self.args.id)
        self._validate_base_args()

        if not self.args.run_parallel:
            sys.exit("This pipeline class currently covers the --run-parallel workflow.")

        stage_map = {
            "generate": self.run_generate,
            "gt_judging": self.run_gt_judging,
            "filter": self.run_filter,
            "select": self.run_select,
            "gt_judging_selected": self.run_gt_judging_selected,
            "selector_no_filter_prep_next": self.run_selector_no_filter_prep_next,
            "mad": self.run_mad,
            "gt_mad": self.run_gt_mad,
        }

        if self.args.stage not in stage_map:
            sys.exit(f"Unsupported stage: {self.args.stage}")

        stage_map[self.args.stage]()
