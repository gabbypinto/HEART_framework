# api_client.py

import logging
import hashlib
import json
import os
import threading
from tenacity import (
    retry, wait_exponential, wait_random, stop_after_attempt, 
    retry_if_exception_type, before_sleep_log
)
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log,wait_random_exponential
from google.api_core import exceptions as google_api_exceptions
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable # Example exceptions

try:
    import tiktoken
except ImportError:
    tiktoken = None

from google import genai
from google.genai import types

try:
    from google import genai
    from google.genai.types import GenerateContentConfigDict, ThinkingConfig
except ImportError as e:
    pass

from google.genai import errors as google_genai_errors # <--- ADD THIS
from anthropic import AnthropicVertex
import httpx
from typing import Literal
from pydantic import BaseModel
from heart import config
from mistralai_gcp import MistralGoogleCloud
import re
import openai
from openai import OpenAI
import time 
logger = logging.getLogger(__name__)
import sys 
import anthropic
import os 
from transformers import pipeline
import torch
from heart.cost_tracker import CallMetrics, TimedCall, extract_reasoning_and_answer

def _required_env(name: str, service: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{service} requires {name} to be set.")
    return value

class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability
   

class GeminiExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int

class EmptyResponseError(Exception):
    """Raised when the Gemini API returns a response with no text."""
    pass


   

class ApiClient:
    """A client to interact with LLM APIs, handling retries and model-specific calls."""

    # Class-level 403 tracking — shared across all instances and all threads.
    # Using os._exit(1) so the kill works even when called from a worker thread
    # (sys.exit only exits the calling thread; os._exit exits the whole process).
    _403_lock: threading.Lock = threading.Lock()
    _403_count: int = 0
    _403_threshold: int = 3

    @classmethod
    def _handle_403(cls, error: Exception) -> None:
        with cls._403_lock:
            cls._403_count += 1
            count = cls._403_count
        logger.error("API 403 Forbidden — total count %d: %s", count, error)
        print(f"ERROR: API returned 403 (total {count}/{cls._403_threshold}): {error}")
        if count >= cls._403_threshold:
            msg = (
                f"FATAL: API returned 403 {count} times. "
                "Credits are likely exhausted. Stopping now — re-run to resume."
            )
            print(msg)
            logger.error(msg)
            os._exit(1)

    def __init__(self, model_name: str):
        self.model_name = model_name
        vertex_retry_options = types.HttpRetryOptions(
            attempts=15,
            initial_delay=1.0,
            max_delay=100.0,
            exp_base=2,
            jitter=15.0,
            http_status_codes=[429, 500, 503],  # 429 (ResourceExhausted), 500/503 (Server Errors)
        )
        vertex_http_options = types.HttpOptions(
            timeout=300,  # 5-minute timeout for hangs
            api_version='v1',
            retry_options=vertex_retry_options  # <-- Pass retry options here
        )
        if "claude" in self.model_name.lower(): #claude-opus-4-1@20250805
            # timeout = httpx.Timeout(config.INDIVIDUAL_TASK_TIMEOUT_SECONDS, connect=60.0)
            
            # self.client = anthropic.Anthropic()
            self.client = AnthropicVertex(
                region=config.CLAUDE_LOCATION,
                project_id=config.CLAUDE_PROJECT_ID,
                # timeout=timeout
            )
        elif "gemini-3-flash" in self.model_name.lower():
            if config.USE_OPENROUTER:
                self.client = OpenAI(api_key=_required_env("OPENROUTER_API_KEY", "OpenRouter"),
                                     base_url=config.OPENROUTER_BASE_URL)
            else:
                self.client = genai.Client(
                    vertexai=True,
                    project=config.GEMINI_PROJECT_ID,
                    location="global",
                    http_options={'api_version':'v1'}
                )
        elif "gemini-3-pro" in self.model_name.lower():
            if config.USE_OPENROUTER:
                self.client = OpenAI(api_key=_required_env("OPENROUTER_API_KEY", "OpenRouter"),
                                     base_url=config.OPENROUTER_BASE_URL)
            else:
                self.client = genai.Client(
                    vertexai=True,
                    project=config.GEMINI_PROJECT_ID,
                    location="global",
                    http_options={'api_version':'v1'}
                )
        elif "gemini" in self.model_name.lower():
            if config.USE_OPENROUTER:
                self.client = OpenAI(api_key=_required_env("OPENROUTER_API_KEY", "OpenRouter"),
                                     base_url=config.OPENROUTER_BASE_URL)
            else:
                self.client = genai.Client(
                    vertexai=True,
                    project=config.GEMINI_PROJECT_ID,
                    location=config.GEMINI_LOCATION,
                    http_options={'api_version':'v1'}
                )
        elif "r1-distill" in self.model_name.lower():
            # DeepSeek R1 Distill family (e.g. deepseek/deepseek-r1-distill-qwen-32b) via OpenRouter
            self.client = OpenAI(
                api_key=_required_env("OPENROUTER_API_KEY", "OpenRouter"),
                base_url=config.OPENROUTER_BASE_URL
            )
        elif "deepseek" in self.model_name.lower():
            self.client=genai.Client(
                vertexai=True,
                project=config.DEEPSEEK_PROJECT_ID,
                location=config.DEEPSEEK_LOCATION,
                http_options={'api_version':'v1'}
            ) 
        elif "gpt_5_nano" in self.model_name.lower():
            self.client=OpenAI(api_key=_required_env("OPENAI_API_KEY", "OpenAI"))
            
        elif "openai" in self.model_name.lower():
            self.client=genai.Client(
                vertexai=True,
                project=config.OPEN_AI_PROJECT_ID,
                location=config.OPEN_AI_LOCATION,
                http_options={'api_version':'v1'}
            ) 
        elif "qwen" in self.model_name.lower():
            self.client=genai.Client(
                vertexai=True,
                project=config.QWEN_PROJECT_ID,
                location=config.QWEN_LOCATION,
                http_options={'api_version':'v1'}
            ) 
        elif "mistral-large" in self.model_name.lower():
            self.client = MistralGoogleCloud(
                region=config.MISTRAL_LOCATION, 
                project_id=config.MISTRAL_PROJECT_ID
            )
        elif "mistral-small" in self.model_name.lower():
            self.client=MistralGoogleCloud(
                region=config.MISTRAL_LOCATION,
                project_id=config.MISTRAL_PROJECT_ID
            )
        else:
            print(f"MODEL NOT FOUND: {self.model_name.lower()}")
            sys.exit(1)

        # --- Caching setup ---
        # self.cache_dir = "api_cache"
        # os.makedirs(self.cache_dir, exist_ok=True)
        # self.cache_file = os.path.join(self.cache_dir, f"{self.model_name}_cache.json")
        # self.cache = self._load_cache()
        # self.cache_lock = threading.Lock()

    def _load_cache(self):
        """Loads the API cache from a file."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError):
            print("Could not load cache file. Starting with an empty cache.")
            logger.warning("Could not load cache file. Starting with an empty cache.")
        return {}

    def _save_cache(self):
        """Saves the current API cache to a file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except IOError:
            print("COULD NOT SAVE CACHE FILE")
            logger.error("Could not save cache file.")

    @retry(
        # wait=wait_exponential(multiplier=1, min=4, max=400) + wait_random(min=0, max=14),
        wait=wait_random_exponential(multiplier=2, min=4, max=400),
        stop=stop_after_attempt(10),
        reraise=True,
        retry=retry_if_exception_type((
            google_api_exceptions.ResourceExhausted,
            google_api_exceptions.InternalServerError,
            google_api_exceptions.ServiceUnavailable,
            google_api_exceptions.Aborted,
            
            google_genai_errors.ClientError,  # <--- ADD THIS LINE

            # google_genai_errors.ResourceExhausted,  # <--- THIS IS THE FIX (uncommented + typo fixed)
            # google_genai_errors.ResourceExcouldhausted,  # <--- ADD THIS LINE
            # Anthropic
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            # OpenAI / DeepSeek
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            EmptyResponseError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            ResourceExhausted, 
            ServiceUnavailable
        )),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def generate_response(self, model_id:str, system_instruction: str, prompt: str, force_json: bool = False, force_json_schema:bool = False) -> str:
        """
        Generates a response from the configured LLM, with built-in retries.
        """
        # --- Caching Logic ---
        # prompt_hash = hashlib.md5((system_instruction + prompt).encode()).hexdigest()
        # cache_key = f"{self.model_name}_{'json' if force_json else 'text'}_{prompt_hash}"
        
        # --- MODIFICATION 3: Use the lock to protect all cache operations ---
        # with self.cache_lock:
        #     if cache_key in self.cache:
        #         logger.info(f"Cache HIT for key: {cache_key}")
        #         return self.cache[cache_key]
        
        # logger.info(f"Cache MISS for key: {cache_key}. Calling API.")

        # --- API Call Logic ---
        if "gemini-3-flash-preview" in self.model_name.lower():
            if config.USE_OPENROUTER:
                response_text = self._call_openrouter_api(
                    config.OPENROUTER_GEMINI3_FLASH_MODEL, system_instruction, prompt, force_json)
            else:
                response_text = self._call_gemini3_api(
                    config.GEMINI3_FLASH_MODEL_NAME, system_instruction, prompt, force_json)
        if "gemini-3-pro-preview" in self.model_name.lower():
            if config.USE_OPENROUTER:
                response_text = self._call_openrouter_api(
                    config.OPENROUTER_GEMINI3_PRO_MODEL, system_instruction, prompt, force_json)
            else:
                response_text = self._call_gemini3_api(
                    config.GEMINI3_PRO_MODEL_NAME, system_instruction, prompt, force_json)

        elif "gemini" in self.model_name.lower():
            if config.USE_OPENROUTER:
                model_id = (config.OPENROUTER_GEMINI_PRO_MODEL
                            if "pro" in self.model_name.lower()
                            else config.OPENROUTER_GEMINI_FLASH_MODEL)
                response_text = self._call_openrouter_api(
                    model_id, system_instruction, prompt, force_json)
            else:
                response_text = self._call_gemini_api(
                    config.GEMINI_FLASH_MODEL_NAME, system_instruction, prompt, force_json)
        elif "claude" in self.model_name.lower():
            response_text = self._call_claude_api(config.CLAUDE_MODEL_NAME,system_instruction, prompt,force_json)
        elif "r1-distill" in self.model_name.lower():
            # Route DeepSeek R1 Distill models through OpenRouter using the model name as-is
            response_text = self._call_openrouter_api(self.model_name, system_instruction, prompt, force_json)
        elif "deepseek-chat" in self.model_name.lower():
            response_text = self._call_deepseek_chat_api(config.DEEPSEEK_MODEL_NAME,system_instruction,prompt,force_json)
        elif "deepseek" in self.model_name.lower():
            response_text = self._call_deepseek_api(config.DEEPSEEK_MODEL_NAME,system_instruction,prompt,force_json)
        elif "openai" in self.model_name.lower():
            response_text = self._call_openai_api(config.OPEN_AI_MODEL_NAME,system_instruction,prompt,force_json)
        elif "gpt_5_nano" in self.model_name.lower():
            response_text = self._call_gpt_api(config.GPT_MODEL_NAME, system_instruction, prompt, force_json)
        elif "qwen" in self.model_name.lower():
            response_text = self._call_qwen_api(config.QWEN_MODEL_NAME,system_instruction,prompt,force_json)
        elif "mistral-large" in self.model_name.lower():
            response_text = self._call_mistral_api(config.MISTARL_LARGE_MODEL_NAME,system_instruction,prompt,force_json)
        elif "mistral-small" in self.model_name.lower():
            response_text = self._call_mistral_api(config.MISTARL_SMALL_MODEL_NAME,system_instruction,prompt,force_json)
        # elif "gemma" in self.model_name.lower():
        #     response_text = self._call_gemma_api("google/gemma-3-1b-it",system_instruction,prompt,force_json)
        #     torch.cuda.empty_cache() 

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return response_text

    def _estimate_tokens(self, text: str) -> int:
        """Best-effort token count for resource tracking without changing API calls."""
        if text is None:
            return 0
        text = str(text)
        if not text:
            return 0
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text) // 4)

    def generate_response_with_metrics(
        self,
        model_id: str,
        system_instruction: str,
        prompt: str,
        force_json: bool = False,
        force_json_schema: bool = False,
    ) -> tuple[str, CallMetrics]:
        """Generate a response and return resource metrics for post-analysis.

        This wraps the existing generate_response path to avoid changing model
        routing behavior. Token counts are estimated when provider usage fields
        are not exposed by the existing call helpers.
        """
        with TimedCall() as timer:
            response_text = self.generate_response(
                model_id=model_id,
                system_instruction=system_instruction,
                prompt=prompt,
                force_json=force_json,
                force_json_schema=force_json_schema,
            )

        thinking_text, answer_text = extract_reasoning_and_answer(response_text)
        metrics = CallMetrics(
            prompt_tokens=self._estimate_tokens(f"{system_instruction or ''}\n{prompt or ''}"),
            reasoning_tokens=self._estimate_tokens(thinking_text),
            output_tokens=self._estimate_tokens(answer_text),
            latency_sec=timer.elapsed_sec,
        )
        metrics.compute_derived()
        return response_text, metrics

    # def _call_gemma_api(self,model_id:str,system_instruction:str,prompt:str,force_json:bool=True):
    #     # "content": "Xenia and Sergey play the following game. Xenia thinks of a positive integer $N$ not exceeding 5000. Then she fixes 20 distinct positive integers $a_{1}, a_{2}, \ldots, a_{20}$ such that, for each $k=1,2, \ldots, 20$, the numbers $N$ and $a_{k}$ are congruent modulo $k$. By a move, Sergey tells Xenia a set $S$ of positive integers not exceeding 20 , and she tells him back the set $\left\{a_{k}: k \in S\right\}$ without spelling out which number corresponds to which index. How many moves does Sergey need to determine for sure the number Xenia thought of?"},

    #     messages = [
    #         {
    #             "role": "user", 
    #             "content": prompt,
    #         },
    #         {
    #             "role":"system",
    #             "content": system_instruction
    #         }
    #     ]

    #     # Add generation parameters directly to the call
    #     output = self.pipe(
    #         messages,
    #         max_new_tokens=40000, # Sets the max new tokens to generate
    #         do_sample=True,      # Enables sampling, required for temperature
    #         temperature=0.6,     # Adjusts randomness (lower = more deterministic)
    #         top_p=0.95           # Nucleus sampling
    #     )

    #     response = output[0]['generated_text'][-1]['content']
    #     return response

    def _call_mistral_api(self,model_id:str, system_instruction:str, prompt:str,force_json:bool=True,force_json_schema:bool=True):
        if force_json:
            resp = self.client.chat.complete(
            model=f"{model_id}",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role":"system",
                    "content": system_instruction
                }
            ],
            response_format={"type":"json_object"}
            )
        else:
            resp = self.client.chat.complete(
            model=f"{model_id}",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role":"system",
                    "content": system_instruction
                }
            ],)
        
        # print(resp)
        # print(resp.choices[0].message.content)
        res = resp.choices[0].message.content
        try:
            return res
        except Exception as e:
            print(f"An error occurred: {e}")
            return ""

    def _call_openai_api(self,model_id:str, system_instruction:str, prompt:str, force_json:bool=True):
        config_args = {
            "temperature": 0.7,
            "max_output_tokens": 64000,
            "thinking_config": types.ThinkingConfig(thinking_budget=1024),
            "top_p": 0.2
            # top_p = 0.2 # default top_p is 0.95
            # "responseLogprobs":True
            # "response_logprobs":True
        }

        if force_json:
            config_args["responseSchema"] = GeminiExtractedAnswer
            config_args["response_mime_type"] = "application/json"
            
        if force_json_schema == False:
            config_args["responseSchema"] = GeminiExtractedAnswer



        response = self.client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                **config_args
            )
        )

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
            raise RuntimeError(f"Content blocked by API. Reason: {block_reason}")
        else:
            # raise EmptyResponseError("Gemini API returned an empty response.")
            return "" # Return empty string for empty responses

    def _call_gpt_api(self,model_id:str, system_instruction:str, prompt:str, force_json:bool=True):

        """
        Makes a single API call to an OpenAI GPT model, with robust error handling.
        """
        # --- Edge Case 1: Handle empty or whitespace-only prompts ---
        clean_prompt = prompt.strip() if prompt else ""
        if not clean_prompt:
            logger.warning("OpenAI API call was attempted with an empty prompt. Skipping.")
            return ""

        # --- Construct the messages payload for the API ---
        if force_json:
            system_instruction+= "Return your responses in JSON format."
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": clean_prompt}
        ]

        # --- START: Dynamic Token Calculation ---
        try:
            # "cl100k_base" is the correct encoding for modern OpenAI models (GPT-4, etc.)
            encoding = tiktoken.get_encoding("cl100k_base")
            input_token_count = sum(len(encoding.encode(message["content"])) for message in messages)

            # Calculate the available space left for the model's response
            available_for_completion = config.GPT_CONTEXT_LIMIT - input_token_count - config.GPT_SAFETY_MARGIN

            # Edge Case: If the prompt is too long and leaves no room for a response
            if available_for_completion <= 0:
                logger.error(f"Input prompt ({input_token_count} tokens) is too long and exceeds the model's context limit.")
                return "ERROR: Input prompt is too long for the model's context window."

            # The max tokens for the API call is the smaller of our desired default and the actual available space
            max_tokens_for_call = min(config.GPT_DEFAULT_MAX_TOKENS, available_for_completion)

        except Exception as e:
            # Fallback in case tiktoken library has an issue
            logger.warning(f"Could not calculate token count, falling back to default max_tokens. Error: {e}")
            max_tokens_for_call = config.GPT_DEFAULT_MAX_TOKENS
        # --- END: Dynamic Token Calculation ---


        # --- Set up API call parameters, including JSON mode ---
        api_params = {
            "model": model_id,
            "messages": messages,
            # "temperature": config.GPT_TEMP,
            "max_completion_tokens": max_tokens_for_call,
            # "top_p": config.GPT_TOP_P
        }
        if force_json:
            api_params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**api_params)
            
            # --- Edge Case 2: Check for a valid response and content ---
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                
                # --- Edge Case 3: Check if the response was stopped by a content filter ---
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "content_filter":
                    logger.warning("OpenAI API response was stopped due to content filtering.")
                    return "" # Return empty string if content is flagged

                return response.choices[0].message.content.strip()
            else:
                logger.warning("OpenAI API returned a response with no content.")
                return ""

        # --- Edge Case 4: Handle specific OpenAI API errors ---
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API request failed to connect: {e}")
            raise # Reraise for tenacity to handle retries
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API request hit a rate limit: {e}")
            raise # Reraise for tenacity to handle retries
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI API authentication failed. Check your API key. Error: {e}")
            return "" # Authentication errors are not retryable
        except openai.BadRequestError as e:
            logger.error(f"OpenAI API request was invalid (e.g., malformed JSON request): {e}")
            return "" # Bad requests are not retryable
        except Exception as e:
            logger.error(f"An unexpected error occurred during the OpenAI API call: {e}")
            return ""


    def _call_deepseek_chat_api(self, model_id: str, system_instruction: str, prompt: str, force_json: bool = True) -> str:
        # 1. Initialize Client
        print("DEEPSEEK CHAT API IN USE")
        client = OpenAI(
            api_key=_required_env("DEEPSEEK_API_KEY", "DeepSeek"),
            base_url="https://api.deepseek.com"
        )
        
        MODEL_CONTEXT_LIMIT = 128000 
        DEFAULT_MAX_TOKENS = 4096   
        SAFETY_MARGIN = 200   
        
        prompt_as_string = str(prompt if prompt is not None else "")
        clean_prompt = prompt_as_string.strip()

        if not clean_prompt:
            logger.warning("DeepSeek API call was attempted with an empty prompt. Skipping.")
            return "" 

        # 2. Fix Message Order (System should usually be first)
        # Note: Do NOT append <think> manually here.
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": clean_prompt} 
        ]
        
        # --- Dynamic Token Calculation (Approximation) ---
        try:
            # Note: cl100k_base is OpenAI's encoding. DeepSeek is compatible but not identical.
            # This serves as a decent approximation for the input buffer.
            encoding = tiktoken.get_encoding("cl100k_base")
            input_token_count = sum(len(encoding.encode(m["content"])) for m in messages)
            available_for_completion = MODEL_CONTEXT_LIMIT - input_token_count - SAFETY_MARGIN

            if available_for_completion <= 0:
                logger.error(f"Input prompt ({input_token_count} tokens) exceeds the model's context limit.")
                return "ERROR: Input is too long."

            max_tokens_for_call = min(DEFAULT_MAX_TOKENS, available_for_completion)

        except Exception as e:
            logger.warning(f"Could not calculate token count, falling back to default. Error: {e}")
            max_tokens_for_call = DEFAULT_MAX_TOKENS

        # --- 3. Make the API Call ---
        try:
            response = client.chat.completions.create(
                model="deepseek-chat", # MUST use 'deepseek-reasoner' for think tokens
                messages=messages,
                max_tokens=max_tokens_for_call,
                stream=False
            )

            # --- 4. Extract Content AND Reasoning ---
            message_obj = response.choices[0].message
            
            # The final answer
            content = message_obj.content 
            
            # The thinking process (This field is specific to DeepSeek R1)
            # We use .get() or getattr because standard 'deepseek-chat' responses 
            # won't have this field.
            reasoning = getattr(message_obj, 'reasoning_content', "")

            # --- 5. Format Return ---
            # Since your function returns a single string, we can format it 
            # to visually separate the thought from the answer.
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                return content

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"ERROR: API call failed - {e}"
        






    def _call_deepseek_api(self, model_id: str, system_instruction: str, prompt: str, force_json: bool = True) -> str:
        # 1. Initialize Client
        client = OpenAI(
            api_key=_required_env("DEEPSEEK_API_KEY", "DeepSeek"),
            base_url="https://api.deepseek.com"
        )
        
        MODEL_CONTEXT_LIMIT = 128000 
        DEFAULT_MAX_TOKENS = 4096   
        SAFETY_MARGIN = 200   
        
        prompt_as_string = str(prompt if prompt is not None else "")
        clean_prompt = prompt_as_string.strip()

        if not clean_prompt:
            logger.warning("DeepSeek API call was attempted with an empty prompt. Skipping.")
            return "" 

        # 2. Fix Message Order (System should usually be first)
        # Note: Do NOT append <think> manually here.
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": clean_prompt} 
        ]
        
        # --- Dynamic Token Calculation (Approximation) ---
        try:
            # Note: cl100k_base is OpenAI's encoding. DeepSeek is compatible but not identical.
            # This serves as a decent approximation for the input buffer.
            encoding = tiktoken.get_encoding("cl100k_base")
            input_token_count = sum(len(encoding.encode(m["content"])) for m in messages)
            available_for_completion = MODEL_CONTEXT_LIMIT - input_token_count - SAFETY_MARGIN

            if available_for_completion <= 0:
                logger.error(f"Input prompt ({input_token_count} tokens) exceeds the model's context limit.")
                return "ERROR: Input is too long."

            max_tokens_for_call = min(DEFAULT_MAX_TOKENS, available_for_completion)

        except Exception as e:
            logger.warning(f"Could not calculate token count, falling back to default. Error: {e}")
            max_tokens_for_call = DEFAULT_MAX_TOKENS

        # --- 3. Make the API Call ---
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner", # MUST use 'deepseek-reasoner' for think tokens
                messages=messages,
                max_tokens=max_tokens_for_call,
                stream=False
            )

            # --- 4. Extract Content AND Reasoning ---
            message_obj = response.choices[0].message
            
            # The final answer
            content = message_obj.content 
            
            # The thinking process (This field is specific to DeepSeek R1)
            # We use .get() or getattr because standard 'deepseek-chat' responses 
            # won't have this field.
            reasoning = getattr(message_obj, 'reasoning_content', "")

            # --- 5. Format Return ---
            # Since your function returns a single string, we can format it 
            # to visually separate the thought from the answer.
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                return content

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"ERROR: API call failed - {e}"
        

    # def _call_deepseek_api(self,model_id:str, system_instruction:str, prompt:str, force_json:bool = True) -> str:
    #     MODEL_CONTEXT_LIMIT = 128000 
    #     DEFAULT_MAX_TOKENS = 4096   
    #     SAFETY_MARGIN = 200   
    #     prompt_as_string = str(prompt if prompt is not None else "")
    #     clean_prompt = prompt_as_string.strip()

    #     if not clean_prompt:
    #         logger.warning("DeepSeek API call was attempted with an empty prompt. Skipping.")
    #         return "" # Exit the function before making the bad API call.

    #     messages = [{"role": "user", "content": clean_prompt+"<think>\n"},
    #                 {"role":"system","content": system_instruction}
    #                 ]
        
    #     # --- Dynamic Token Calculation ---
    #     try:
    #         encoding = tiktoken.get_encoding("cl100k_base")
    #         input_token_count = sum(len(encoding.encode(message["content"])) for message in messages)
    #         available_for_completion = MODEL_CONTEXT_LIMIT - input_token_count - SAFETY_MARGIN

    #         if available_for_completion <= 0:
    #             logger.error(f"Input prompt ({input_token_count} tokens) exceeds the model's context limit.")
    #             return "ERROR: Input is too long."

    #         max_tokens_for_call = min(DEFAULT_MAX_TOKENS, available_for_completion)

    #     except Exception as e:
    #         logger.warning(f"Could not calculate token count, falling back to default. Error: {e}")
    #         max_tokens_for_call = DEFAULT_MAX_TOKENS



    #     response = self.client.chat.completions.create(
    #         model="deepseek-reasoner",
    #         messages=messages,
    #         max_tokens=max_tokens_for_call,
    #         temperature=0.2,
    #         top_p=0.3,
    #         stream=False
    #     )
    #     time.sleep(1)
    #     content = response.choices[0].message.content if response.choices else ""
    #     return content

    #     # content = ""
    #     # reasoning_content = ""

    #     # if hasattr(response,'choices') and response.choices[0]:
    #     #     if hasattr(response.choices[0],'message'):
    #     #         if hasattr(response.choices[0].message,'reasoning_content'):
    #     #             reasoning_content = response.choices[0].message.reasoning_content
    #     #             reasoning_content = f"<think>{reasoning_content}</think>"
    #     #         if hasattr(response.choices[0].message,'content'):
    #     #             content = response.choices[0].message.content
        
    #     # full_response = f"{reasoning_content} {content}"
    #     # return full_response
    

    # def _call_deepseek_api(self,model_id:str, system_instruction:str, prompt:str, force_json:bool = True) -> str:
    #     """Makes a single API to a Deepseek model"""
    #     config_args = {
    #         "temperature": config.TEMP,
    #         "max_output_tokens": 25000,
    #     }

    #     if force_json:
    #         print("force json true")
    #         config_args["responseSchema"] = GeminiExtractedAnswer
    #         config_args["response_mime_type"] = "application/json"
            
    #     # print("pre response")
    #     # print("prompt",prompt)
    #     # print()
    #     # print("-")
    #     # print("sys:",system_instruction)
    #     # 1. Ensure prompt is a string and handle if it's None.
    #     #    str(None) becomes "None", but str("") remains "".
    #     prompt_as_string = str(prompt if prompt is not None else "")

    #     # 2. Use .strip() to remove any leading/trailing whitespace.
    #     #    This turns a prompt like "   " into an empty string "".
    #     clean_prompt = prompt_as_string.strip()

    #     # 3. CRITICAL: Check if the resulting string is empty.
    #     #    This is the main safeguard that prevents sending an invalid request.
    #     if not clean_prompt:
    #         logger.warning("DeepSeek API call was attempted with an empty prompt. Skipping.")
    #         return "" # Exit the function before making the bad API call.

        
    #     response = self.client.models.generate_content(
    #         model="deepseek-ai/deepseek-r1-0528-maas",
    #         contents=clean_prompt,
    #         config=types.GenerateContentConfig(
    #             system_instruction=system_instruction,
    #             **config_args
    #         ),
    #     )

    #     print("post response")

    #     if hasattr(response, 'text') and response.text:
    #         return response.text.strip()
    #     elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
    #         block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
    #         raise RuntimeError(f"Content blocked by API. Reason: {block_reason}")
    #     else:
    #         return "" # Return empty string for empty responses

    def _call_qwen_api(self, model_id:str, system_instruction: str, prompt: str, force_json: bool = True) -> str:
        """Makes a single API call to a Qwen model."""
        
        config_args = {
            "temperature": config.TEMP,
            "max_output_tokens": 10000
            # "thinking_config": types.ThinkingConfig(thinking_budget=1024)
        }

        if force_json:
            print("force json")
            config_args["responseSchema"] = GeminiExtractedAnswer
            config_args["response_mime_type"] = "application/json"
            
        response = self.client.models.generate_content(
            model=config.QWEN_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                **config_args
            )
        )

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
            raise RuntimeError(f"Content blocked by API. Reason: {block_reason}")
        else:
            return "" # Return empty string for empty responses

    def generate_response_with_logprobs(self, model_id: str, system_instruction: str, prompt: str, force_json: bool = True) -> tuple[str, list | None]:
            """
            Makes a single API call to a Gemini model, requesting and returning token log probabilities.
            This version is more robust against varied API response structures.
            """
            # --- Configuration Setup (same as your reference function) ---
            config_args = {
                "temperature": config.TEMP,
                "max_output_tokens": 64000,
                "thinking_config": types.ThinkingConfig(thinking_budget=1024),
                "response_logprobs": True
            }
            if "flash" in model_id or "pro" in model_id:
                config_args.update({"temperature": 0.7, "top_p": 0.2})
            if force_json:
                config_args["responseSchema"] = GeminiExtractedAnswer
                config_args["response_mime_type"] = "application/json"
            if config.GEMINI_THINK_BOOL == False:
                print("***** Thinking Budget set to 0 *****")
                config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

            # --- API Call (same as before) ---
            response = self.client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    **config_args
                )
            )

            # --- NEW: More Robust Response Handling ---
            text_response = ""
            logprobs_data = None

            # # print(response)
            # print(type(response))
            # print(len(response.candidates))
            # print(type(response.candidates))
            # print(type(response.candidates[0]))
            # print(dir(response.candidates[0]))
            # print(response.candidates[0].avg_logprobs)
            # print(type(response.candidates[0].logprobs_result))
            # # print(len(response.candidates[0].logprobs_result))
            # print(dir(response.candidates[0].logprobs_result))
            # print(len(response.candidates[0].logprobs_result.chosen_candidates))

            # print(response.candidates[0].logprobs_result.chosen_candidates)
            # print("------")
            # print(len(response.candidates[0]))
            #.logprobs_results[0])
            #print(response.candidates[0].logprobs_results[0].log_probability)

            # import sys
            # sys.exit(1)

            # if hasattr(response, 'candidates') and response.candidates:
            #     candidate = response.candidates[0]

            if hasattr(response, 'text') and response.text:
                text_response= response.text.strip()

            #     if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
            #         logprobs_data = []
            #         try:
            #             # **THE FIX IS HERE**: This logic specifically handles the nested structure
            #             # seen in your logs. It finds the list of LogprobsResultCandidate objects.
                        
            #             # The actual list of probabilities is nested deep inside the object
            #             list_of_probs = candidate.logprobs_result.chosen_candidates[0].log_probability

            #             for item in list_of_probs:
            #                 # Now, we correctly access the attributes of each item
            #                 logprobs_data.append({
            #                     "token": item.token,
            #                     "log_prob": item.log_probability
            #                 })

            #         except (AttributeError, IndexError, TypeError) as e:
            #             # This will catch any future unexpected structure changes from the API
            #             print(f"Warning: Could not process logprobs due to an unexpected structure. Error: {e}")
            #             logprobs_data = None
            
            # elif hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            #     block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
            #     raise RuntimeError(f"Content blocked by API. Reason: {block_reason}")
            print(text_response)
            return text_response, response.candidates[0].avg_logprobs

    def _call_gemini3_api(self, model_id:str, system_instruction: str, prompt: str, force_json: bool = True) -> str:    
        # Gemini 3 Flash Thinking requires Temp 1.0
        config_args = {
            "max_output_tokens": 64000,
            "temperature": 1.0, 
            "top_p": 0.95, # Increased slightly for better reasoning fluidity
            "top_k": 40,
            "thinking_config": types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH
            ),
        }

        if force_json:
            config_args["response_mime_type"] = "application/json"
            # Highly Recommended: Pass your Pydantic model here if using the latest SDK
            config_args["responseSchema"] = GeminiExtractedAnswer
        
        # """Makes a single API call to a Gemini model."""

        # print("GEMINI3 MODEL:",model_id)
        
        # config_args = {
        #     "max_output_tokens": 64000,
        #     "top_p": 0.2,
        #     "temperature":1.0,
        #     "thinking_config":ThinkingConfig(include_thoughts=True,thinking_level=types.ThinkingLevel.HIGH),
        #     # "responseLogprobs":True
        #     # "response_logprobs":True
        # }

        # config=types.GenerateContentConfig(
            #     system_instruction=system_instruction,
            #     **config_args
            # ),
            

        # if force_json:
        #     # config_args["responseSchema"] = GeminiExtractedAnswer
        #     config_args["response_mime_type"] = "application/json"
            
        # #  thinking_config=ThinkingConfig(include_thoughts=False,thinking_level=types.ThinkingLevel.HIGH),

        # print("using model id: ",model_id)
        response = self.client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
                thinking_config=types.ThinkingConfig(
                    thinking_level="High",
                    include_thoughts=True
                )
            )

        )
        # print("RESPONSE (GEM3):",response)

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
            raise RuntimeError(f"Content blocked by API. Reason: {block_reason}")
        else:
            # raise EmptyResponseError("Gemini API returned an empty response.")
            return "" # Return empty string for empty responses


    def _call_openrouter_api(self, model_id: str, system_instruction: str,
                              prompt: str, force_json: bool = False) -> str:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": prompt},
        ]
        api_params = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 64000,
            "temperature": config.TEMP,
        }
        if force_json:
            api_params["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**api_params)
        except openai.PermissionDeniedError as e:
            ApiClient._handle_403(e)
            raise

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    def _call_gemini_api(self, model_id:str, system_instruction: str, prompt: str, force_json: bool = True) -> str:
        """Makes a single API call to a Gemini model."""
        
        config_args = {
            "temperature": config.TEMP,
            "max_output_tokens": 64000,
            "thinking_config": types.ThinkingConfig(thinking_budget=1024)
            # top_p = 0.2 # default top_p is 0.95
            # "responseLogprobs":True
            # "response_logprobs":True
        }

        if "flash" in model_id:
            config_args = {
            "temperature": 0.4,
            "max_output_tokens": 64000,
            "thinking_config": types.ThinkingConfig(thinking_budget=1024),
            "top_p": 0.2
            # top_p = 0.2 # default top_p is 0.95
            # "responseLogprobs":True
            # "response_logprobs":True
        }
        elif "pro" in model_id:
            config_args = {
            "temperature": 0.7,
            "max_output_tokens": 64000,
            "thinking_config": types.ThinkingConfig(thinking_budget=1024),
            "top_p": 0.2
            # top_p = 0.2 # default top_p is 0.95
            # "responseLogprobs":True
            # "response_logprobs":True
        }


        if config.DECODING_BOOL_TEMP_0 == True:
            print("TEMPERATURE SET TO 0")
            config_args["temperature"] = 0.0

        if config.DECODING_BOOL_TEMP_1 == True:
            print("TEMPERATURE SET TO 1")
            config_args["temperature"] = 1.0
        

        if force_json:
            # config_args["responseSchema"] = GeminiExtractedAnswer
            config_args["response_mime_type"] = "application/json"
            

        if config.GEMINI_THINK_BOOL == False:
            print("***** Thinking Budget set to 0 *****")
            config_args["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        
        response = self.client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                **config_args
            )
        )

        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            block_reason = getattr(response.prompt_feedback, 'block_reason', 'Unknown')
            raise RuntimeError(f"Content blocked by API. Reason: {block_reason}")
        else:
            # raise EmptyResponseError("Gemini API returned an empty response.")
            return "" # Return empty string for empty responses

    
    def _call_claude_api(self, model_id:str, system_instruction: str, prompt: str,force_json:bool = True) -> str:
        """Makes a single API call to the Claude model."""
        
        if config.CLAUDE_THINK_BOOL == True or config.CLAUDE_THINK_BOOL == False:


            MODEL_CONTEXT_LIMIT = 200000
            DEFAULT_MAX_TOKENS = 5000
            SAFETY_MARGIN = 300

            final_system_instruction = system_instruction

            # --- Dynamic Token Calculation (Approximation) ---
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = len(encoding.encode(prompt))
                system_tokens = len(encoding.encode(system_instruction))
                input_token_count = prompt_tokens + system_tokens

                available_for_completion = MODEL_CONTEXT_LIMIT - input_token_count - SAFETY_MARGIN

                if available_for_completion <= 0:
                    logger.error(f"Input ({input_token_count} tokens) exceeds model's context limit.")
                    return "ERROR: Input is too long."

                max_tokens_for_call = min(DEFAULT_MAX_TOKENS, available_for_completion)

            except Exception as e:
                logger.warning(f"Could not calculate token count, falling back to default. Error: {e}")
                max_tokens_for_call = DEFAULT_MAX_TOKENS
            
            thinking_config_dict = {
                        "type": "enabled",
                        "budget_tokens": 1024
            }
            if config.CLAUDE_THINK_BOOL == False:      
                print("THINK IS OFF")  
                thinking_config_type = "disabled"
                thinking_config_dict = {
                    "type": thinking_config_type
                }
                
                #{
                 #           "type": "disabled",
                  #      }
            else:
                thinking_config_type = "enabled"
                thinking_tokens = 1024
                thinking_config_dict = {
                        "type": thinking_config_type,
                        "budget_tokens": thinking_tokens
                },
                # print("THINK IS ON")
                # thinking_config = {
                #         "type": "enabled",
                #         "budget_tokens": 1024
                #     },
            
            try:

                message=""
                if config.CLAUDE_THINK_BOOL == True:


                    message = self.client.messages.create(
                        model=model_id,
                        system=final_system_instruction, # Use the potentially modified system prompt
                        messages=[
                            {"role": "user", "content": prompt},

                        ],
                        thinking={
                        "type": "enabled",
                        "budget_tokens": 1024
                        },
                        max_tokens=max_tokens_for_call, # Use your calculated max_tokens_for_call
                        temperature=1.0,  #temperature setting 
                    )
                else:
                    message = self.client.messages.create(
                        model=model_id,
                        system=final_system_instruction, # Use the potentially modified system prompt
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        thinking=thinking_config_dict,
                        max_tokens=max_tokens_for_call, # Use your calculated max_tokens_for_call
                        temperature=1, 
                    )
                # time.sleep(2)
                ## , 'message': '`temperature` may only be set to 1 when thinking is enabled. Please consult our documentation at https://docs.claude.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking'}, 'request_id': 'req_vrtx_011CVbKei2VTUxC4oRG2vuDF'}
# model_response ERROR: API Error - Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': '`temperature` may only be set to 1 when thinking is enabled. Please consult our documentation at https://docs.claude.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking'}, 'request_id': 'req_vrtx_011CVbKei2VTUxC4oRG2vuDF'}

                response_text = ""
                for content_block in message.content:
                    if content_block.type == "text":
                        response_text += content_block.text

                print("RESPONSE TEXT:",response_text)
                
                return response_text

            except anthropic.APIError as e:
                logger.error(f"An Anthropic API error occurred: {e}")
                return f"ERROR: API Error - {e}"
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                return f"ERROR: Unexpected - {e}"
