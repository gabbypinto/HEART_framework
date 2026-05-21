"""Resource tracking helpers for HEART model calls.

The metrics here intentionally track resources only: tokens, latency,
throughput, and GPU time. Dollar estimates are kept out of this module.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CallMetrics:
    """Resource metrics for one generation or judging call."""

    prompt_tokens: int = 0
    reasoning_tokens: int = 0
    output_tokens: int = 0

    total_tokens: int = 0
    latency_sec: float = 0.0
    throughput_tok_per_sec: float = 0.0

    peak_vram_gb: float = 0.0
    num_gpus: int = 0
    gpu_seconds: float = 0.0
    gpu_hours: float = 0.0

    def compute_derived(self) -> None:
        self.total_tokens = self.prompt_tokens + self.reasoning_tokens + self.output_tokens
        if self.latency_sec > 0:
            self.throughput_tok_per_sec = self.total_tokens / self.latency_sec
        if self.num_gpus > 0:
            self.gpu_seconds = self.latency_sec * self.num_gpus
            self.gpu_hours = self.gpu_seconds / 3600

    def to_row_dict(self, suffix: str = "") -> Dict[str, Any]:
        return {
            f"prompt_tokens{suffix}": self.prompt_tokens,
            f"reasoning_tokens{suffix}": self.reasoning_tokens,
            f"output_tokens{suffix}": self.output_tokens,
            f"total_tokens{suffix}": self.total_tokens,
            f"latency_sec{suffix}": round(self.latency_sec, 3),
            f"throughput_tok_per_sec{suffix}": round(self.throughput_tok_per_sec, 2),
            f"peak_vram_gb{suffix}": round(self.peak_vram_gb, 3),
            f"num_gpus{suffix}": self.num_gpus,
            f"gpu_seconds{suffix}": round(self.gpu_seconds, 3),
            f"gpu_hours{suffix}": round(self.gpu_hours, 6),
        }


class TimedCall:
    """Context manager for wall-clock latency and optional CUDA peak memory."""

    def __init__(self, track_vram: bool = False):
        self.track_vram = track_vram
        self.elapsed_sec: float = 0.0
        self.peak_vram_gb: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimedCall":
        if self.track_vram:
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_sec = time.perf_counter() - self._start
        if self.track_vram:
            try:
                import torch

                self.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
            except Exception:
                self.peak_vram_gb = 0.0


def vram_stats_gb() -> Dict[str, float]:
    """Return aggregate current and peak CUDA memory in GB, or zeros."""
    try:
        import torch

        current = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
        peak = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        return {
            "current_vram_gb": current / (1024**3),
            "peak_vram_gb": peak / (1024**3),
        }
    except Exception:
        return {"current_vram_gb": 0.0, "peak_vram_gb": 0.0}


def metrics_from_vllm_output(output, elapsed_sec: float, num_gpus: int, peak_vram_gb: float = 0.0) -> CallMetrics:
    """Build metrics from a single vLLM RequestOutput."""
    prompt_tokens = len(output.prompt_token_ids) if getattr(output, "prompt_token_ids", None) else 0
    output_tokens = 0
    if getattr(output, "outputs", None):
        output_tokens = sum(len(o.token_ids) for o in output.outputs if getattr(o, "token_ids", None))

    metrics = CallMetrics(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        latency_sec=elapsed_sec,
        peak_vram_gb=peak_vram_gb,
        num_gpus=num_gpus,
    )
    metrics.compute_derived()
    return metrics


def split_reasoning_from_text(generated_text: str) -> tuple[int, int]:
    """Return approximate reasoning and answer character counts from <think> tags."""
    thinking, answer = extract_reasoning_and_answer(generated_text)
    return len(thinking), len(answer)


def extract_reasoning_and_answer(generated_text: str) -> tuple[str, str]:
    """Split <think>...</think> reasoning from the visible final answer."""
    text = "" if generated_text is None else str(generated_text)
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()
