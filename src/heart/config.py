"""Runtime configuration for HEART.

All paths and credentials are resolved from environment variables.  The
defaults are project-relative so the package can be cloned and run locally.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_file(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
_load_env_file(PROJECT_ROOT / ".env")


def _env_path(name: str, default: Path) -> str:
    return str(Path(os.environ.get(name, str(default))).expanduser().resolve())


BASE_DIR = _env_path("HEART_HOME", PROJECT_ROOT)
DATA_DIR = _env_path("HEART_DATA_DIR", Path(BASE_DIR) / "examples" / "data")
LOG_DIR = _env_path("HEART_LOG_DIR", Path(BASE_DIR) / "logs")
BASE_RESULTS_DIR = _env_path("HEART_RESULTS_DIR", Path(BASE_DIR) / "results")

TEST_DATA_EXPERIMENTS_RESULTS_DIR = str(Path(BASE_RESULTS_DIR) / "test_split_experiment_results")
VALIDATION_DATA_EXPERIMENTS_RESULTS_DIR = str(Path(BASE_RESULTS_DIR) / "validation_split_experiment_results")

for directory in (
    LOG_DIR,
    BASE_RESULTS_DIR,
    TEST_DATA_EXPERIMENTS_RESULTS_DIR,
    VALIDATION_DATA_EXPERIMENTS_RESULTS_DIR,
):
    Path(directory).mkdir(parents=True, exist_ok=True)


# Concurrency and retry settings.
MAX_WORKERS = int(os.environ.get("HEART_MAX_WORKERS", "30"))
SEMAPHORE_LIMIT = int(os.environ.get("HEART_SEMAPHORE_LIMIT", "4"))
API_RETRY_ATTEMPTS = int(os.environ.get("HEART_API_RETRY_ATTEMPTS", "5"))
INDIVIDUAL_TASK_TIMEOUT_SECONDS = int(os.environ.get("HEART_TASK_TIMEOUT_SECONDS", "600"))


# API and model routing.  Values are non-secret IDs only; credentials must be
# supplied by environment variables or provider-native auth.
GEMINI_PROJECT_ID = os.environ.get("VERTEX_GEMINI_PROJECT", "")
GEMINI_LOCATION = os.environ.get("VERTEX_GEMINI_LOCATION", "us-central1")
GEMINI_THINK_BOOL = os.environ.get("HEART_GEMINI_THINK", "true").lower() == "true"
GEMINI_FLASH_MODEL_NAME = os.environ.get("HEART_GEMINI_FLASH_MODEL", "gemini-2.5-flash")
GEMINI_PRO_MODEL_NAME = os.environ.get("HEART_GEMINI_PRO_MODEL", "gemini-2.5-pro")
GEMINI3_FLASH_MODEL_NAME = os.environ.get("HEART_GEMINI3_FLASH_MODEL", "gemini-3-flash-preview")
GEMINI3_PRO_MODEL_NAME = os.environ.get("HEART_GEMINI3_PRO_MODEL", "gemini-3-pro-preview")

USE_OPENROUTER = os.environ.get("HEART_USE_OPENROUTER", "true").lower() == "true"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_GEMINI_FLASH_MODEL = os.environ.get("OPENROUTER_GEMINI_FLASH_MODEL", "google/gemini-2.5-flash")
OPENROUTER_GEMINI_PRO_MODEL = os.environ.get("OPENROUTER_GEMINI_PRO_MODEL", "google/gemini-2.5-pro")
OPENROUTER_GEMINI3_FLASH_MODEL = os.environ.get("OPENROUTER_GEMINI3_FLASH_MODEL", "google/gemini-3-flash-preview")
OPENROUTER_GEMINI3_PRO_MODEL = os.environ.get("OPENROUTER_GEMINI3_PRO_MODEL", "google/gemini-3-pro-preview")

CLAUDE_PROJECT_ID = os.environ.get("VERTEX_CLAUDE_PROJECT", "")
CLAUDE_LOCATION = os.environ.get("VERTEX_CLAUDE_LOCATION", "us-east5")
CLAUDE_THINK_BOOL = os.environ.get("HEART_CLAUDE_THINK", "true").lower() == "true"
CLAUDE_VERTEX_MODEL_NAME = os.environ.get("HEART_CLAUDE_VERTEX_MODEL", "claude-sonnet-4-5@20250929")
CLAUDE_MODEL_NAME = os.environ.get("HEART_CLAUDE_MODEL", CLAUDE_VERTEX_MODEL_NAME)

DEEPSEEK_PROJECT_ID = os.environ.get("VERTEX_DEEPSEEK_PROJECT", "")
DEEPSEEK_LOCATION = os.environ.get("VERTEX_DEEPSEEK_LOCATION", "us-central1")
DEEPSEEK_MODEL_NAME = os.environ.get("HEART_DEEPSEEK_MODEL", "deepseek-ai/deepseek-r1-0528-maas")

OPEN_AI_PROJECT_ID = os.environ.get("VERTEX_OPENAI_PROJECT", "")
OPEN_AI_LOCATION = os.environ.get("VERTEX_OPENAI_LOCATION", "us-central1")
OPEN_AI_MODEL_NAME = os.environ.get("HEART_OPENAI_VERTEX_MODEL", "openai/gpt-oss-120b-maas")

GPT_MODEL_NAME = os.environ.get("HEART_GPT_MODEL", "gpt-5-nano-2025-08-07")
GPT_TEMP = float(os.environ.get("HEART_GPT_TEMP", "0.4"))
GPT_TOP_P = float(os.environ.get("HEART_GPT_TOP_P", "0.2"))
GPT_CONTEXT_LIMIT = int(os.environ.get("HEART_GPT_CONTEXT_LIMIT", "400000"))
GPT_DEFAULT_MAX_TOKENS = int(os.environ.get("HEART_GPT_DEFAULT_MAX_TOKENS", "110000"))
GPT_SAFETY_MARGIN = int(os.environ.get("HEART_GPT_SAFETY_MARGIN", "200"))

QWEN_PROJECT_ID = os.environ.get("VERTEX_QWEN_PROJECT", "")
QWEN_LOCATION = os.environ.get("VERTEX_QWEN_LOCATION", "us-south1")
QWEN_MODEL_NAME = os.environ.get("HEART_QWEN_MODEL", "publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas")

MISTRAL_PROJECT_ID = os.environ.get("VERTEX_MISTRAL_PROJECT", "")
MISTRAL_LOCATION = os.environ.get("VERTEX_MISTRAL_LOCATION", "us-central1")
MISTARL_LARGE_MODEL_NAME = os.environ.get("HEART_MISTRAL_LARGE_MODEL", "mistral-large-2411")
MISTARL_SMALL_MODEL_NAME = os.environ.get("HEART_MISTRAL_SMALL_MODEL", "mistral-small-2503")


TOTAL_ITERATIONS = int(os.environ.get("HEART_TOTAL_ITERATIONS", "5"))
TEMP = float(os.environ.get("HEART_TEMP", "0.4"))
DECODING_BOOL_TEMP_0 = os.environ.get("HEART_DECODING_TEMP_0", "false").lower() == "true"
DECODING_BOOL_TEMP_1 = os.environ.get("HEART_DECODING_TEMP_1", "false").lower() == "true"
