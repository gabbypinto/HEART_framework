# livecode/vllm_backend.py
#
# VLLMBackend — wraps a locally-loaded vLLM model so it can be used as a
# drop-in alternative to HEART's ApiClient in LCBExperimentRunner.
#
# Usage
# -----
#   backend = VLLMBackend(
#       model_path="google/gemma-3-12b-it",
#       tensor_parallel_size=8,
#       temperature=0.5,
#       max_tokens=2048,
#   )
#   text = backend.generate(system_instruction, user_prompt)
#   texts = backend.generate_batch(prompt_pairs)   # list of (sys, user) tuples
#
# Note: vLLM is imported lazily so importing this module in a non-vLLM
# environment does not fail; the error only surfaces when VLLMBackend() is
# constructed.

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "no metrics" — keeps the interface compatible with ApiClient
# callers that do `text, metrics = backend.generate_response_with_metrics(...)`
_NULL_METRICS = {}


class VLLMBackend:
    """
    Local-model backend for LCBExperimentRunner.

    Parameters
    ----------
    model_path : str
        HuggingFace model id or absolute local path, e.g. "google/gemma-3-12b-it".
    tensor_parallel_size : int
        Number of GPUs to shard the model across (default 1).
    temperature : float
        Sampling temperature (default 0.7).
    top_p : float
        Nucleus sampling probability (default 0.95).
    max_tokens : int
        Maximum tokens to generate per call (default 2048).
    dtype : str
        Model weight dtype passed to vLLM, e.g. "bfloat16" (default "auto").
    trust_remote_code : bool
        Passed to vLLM / HF tokenizer (default True).
    enable_prefix_caching : bool
        Enable prefix KV-cache in vLLM for prompt reuse (default True).
    stop : list[str] | None
        Stop strings for sampling.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        enable_prefix_caching: bool = True,
        stop: list[str] | None = None,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "vllm is not installed.  Install it with:  pip install vllm"
            ) from exc

        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        except Exception as exc:
            logger.warning("Could not load tokenizer for %s: %s", model_path, exc)
            self._tokenizer = None

        logger.info(
            "Initialising vLLM engine: model=%s  tensor_parallel_size=%d",
            model_path,
            tensor_parallel_size,
        )
        self._llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )

        self._SamplingParams = SamplingParams
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop or []
        self.model_name = model_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_chat(self, system_instruction: str, user_prompt: str) -> str:
        """Apply the model's chat template to produce a single string prompt."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_prompt})

        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: simple concatenation
        parts = []
        if system_instruction:
            parts.append(f"<|system|>\n{system_instruction}\n")
        parts.append(f"<|user|>\n{user_prompt}\n<|assistant|>\n")
        return "".join(parts)

    def _make_sampling_params(self, n: int = 1, temperature: float | None = None) -> Any:
        return self._SamplingParams(
            n=n,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
        )

    # ------------------------------------------------------------------
    # Public API (mirrors ApiClient interface used in experiment_runner)
    # ------------------------------------------------------------------

    def generate_response_with_metrics(
        self,
        model_id: str,
        system_instruction: str,
        prompt: str,
        temperature: float | None = None,
        n: int = 1,
    ) -> tuple[str, dict]:
        """
        Generate a single response.  Returns (text, metrics_dict).
        metrics_dict is empty for local models (no token cost tracking).
        """
        formatted = self._format_chat(system_instruction, prompt)
        sampling = self._make_sampling_params(n=1, temperature=temperature)
        outputs = self._llm.generate([formatted], sampling)
        text = outputs[0].outputs[0].text if outputs else ""
        return text, _NULL_METRICS

    def generate_batch(
        self,
        prompt_pairs: list[tuple[str, str]],
        n: int = 1,
        temperature: float | None = None,
    ) -> list[list[str]]:
        """
        Generate responses for a batch of (system_instruction, user_prompt) pairs.

        Returns list[list[str]] — outer index = prompt, inner = n samples.
        Using vLLM's native batching for throughput.
        """
        formatted = [self._format_chat(sys_, usr_) for sys_, usr_ in prompt_pairs]
        sampling = self._make_sampling_params(n=n, temperature=temperature)
        outputs = self._llm.generate(formatted, sampling)
        return [[o.text for o in out.outputs] for out in outputs]

    def synthesize_candidates(
        self,
        candidate_codes_by_qid: dict[str, list[str]],
    ) -> dict[str, tuple[str, str]]:
        """
        Given {question_id: [code1, code2, ...]}, synthesise a single best
        solution per problem using the LLM.

        Returns {question_id: (raw_response, extracted_code)}.
        """
        from heart.livecode.judge import extract_code as _extract

        unique_qids = list(candidate_codes_by_qid.keys())
        prompts = []
        for qid in unique_qids:
            codes = candidate_codes_by_qid[qid]
            body = (
                "You are an expert programming assistant. You will be given several "
                "Python solutions for the same problem. Analyse all of them, identify "
                "the best parts of each, and synthesise them into a single, new, "
                "robust, and correct solution.\n\n"
            )
            for i, c in enumerate(codes):
                body += f"--- Candidate {i+1} ---\n```python\n{c}\n```\n\n"
            body += (
                "--- Task ---\nProvide a single, complete Python code block enclosed "
                "in ```python ... ``` markers.\n\nFinal Synthesised Code:\n"
            )
            prompts.append(body)

        synthesis_params = self._SamplingParams(
            n=1, temperature=0.1, top_p=self.top_p,
            max_tokens=self.max_tokens, stop=self.stop,
        )
        outputs = self._llm.generate(prompts, synthesis_params)
        results: dict[str, tuple[str, str]] = {}
        for qid, out in zip(unique_qids, outputs):
            raw = out.outputs[0].text if out.outputs else ""
            code = _extract(raw)
            results[qid] = (raw, code)
        return results
