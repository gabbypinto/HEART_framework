import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a modularized multi-iteration LLM experiment."
    )

    # Experiment identity
    parser.add_argument("--id", required=True, help="A unique instance ID for this run.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=[
            "OlympiadBench",
            "Livecodebench",
            "HLE",
            "SimpleQA",
            "SimpleQA_verified",
            "GPQA",
            "AIME2025",
            "AIME2024",
        ],
        help="The name of the benchmark to run.",
    )
    parser.add_argument("--subject", help="Subject of the experiment.")
    parser.add_argument("--model", required=True, help="Name of the model to use.")
    parser.add_argument("--prompt_strategy", required=True, help="The name of the prompt strategy to use.")
    parser.add_argument("--dataset", required=True, help="Dataset split, e.g. validation or test.")

    # Workflow control
    parser.add_argument("--stage", required=True, help="The specific pipeline stage to run.")
    parser.add_argument("--start_it", type=int, default=None, help="Start iteration number.")
    parser.add_argument("--run-parallel", action="store_true", help="Run the experiment in parallel using split chunks.")
    parser.add_argument("--workflow_type", type=str, default=None, help="Type of workflow pipeline")
    parser.add_argument("--num-chunks", type=int, default=4, help="Number of chunks to split the file into.")
    parser.add_argument("--input", type=str, help="Direct input file for a single run (used by parallel processes).")

    # Generation
    parser.add_argument("--parallel_k", type=int, default=1, help="Number of samples for parallel generation (k)")
    parser.add_argument("--temperature", type=float, default=0.4, help="The temperature setting for the model.")
    parser.add_argument(
        "--get-logprobs",
        action="store_true",
        help="Request and save token log probabilities for this run (Gemini only).",
    )
    parser.add_argument("--prepend_string", type=str, default="", help="String that you want to prepend.")
    parser.add_argument(
        "--base_completion_prefix",
        type=str,
        default="",
        help="Optional continuation prefix appended only for base/PT vLLM models.",
    )

    # MAD
    parser.add_argument("--mad_agents", type=int, default=5, help="Number of MAD agents.")
    parser.add_argument("--mad_rounds", type=int, default=8, help="Number of MAD rounds.")

    parser.add_argument(
        "--rule_based_judging",
        action="store_true",
        help="Use rule-based (no LLM) judging for gt_judging stage instead of Gemini API calls.",
    )
    parser.add_argument(
        "--vllm_llm_as_judge",
        action="store_true",
        help="Use Qwen/Qwen2.5-72B-Instruct via local vLLM as the judge instead of the Gemini API. "
             "Ignored when --rule_based_judging is also set.",
    )

    # Legacy / optional
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="For a given stage, rerun only the rows that failed in the previous attempt.",
    )

    return parser


def run_main() -> None:
    args = build_parser().parse_args()
    # Keep args.model intact for API routing.
    # Add args.model_safe for file-path construction — '/' in OpenRouter / HuggingFace
    # IDs (e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") would be misinterpreted
    # as a directory separator by the OS.
    args.model_safe = args.model.replace("/", "_")

    from heart.experiment_pipeline import ExperimentPipeline

    ExperimentPipeline(args).run()


if __name__ == "__main__":
    run_main()
