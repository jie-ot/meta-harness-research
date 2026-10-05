"""LLM abstraction for memory systems.

Thin wrapper around litellm.completion with GPT-OSS harmony parsing.
"""

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from .pilot_observation import record_call, submit_with_context

from litellm import completion as litellm_completion
from litellm import completion_cost, token_counter
from openai_harmony import HarmonyEncodingName, Role, load_harmony_encoding
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "TEXT_CLASSIFICATION_LLM_CACHE_DIR",
            "~/.cache/text-classification/litellm",
        )
    )
)
CACHE_VERSION = 1
KNOWN_PROVIDER_PREFIXES = (
    "anthropic/",
    "azure/",
    "bedrock/",
    "cohere/",
    "gemini/",
    "groq/",
    "ollama/",
    "openai/",
    "openrouter/",
    "together_ai/",
    "togethercomputer/",
    "vertex_ai/",
    "xai/",
)

MAX_PROMPT_CHARS = 224_000

_HARMONY_ENC = None


def _get_harmony_enc():
    """Lazy-load harmony encoder to avoid import overhead when not using GPT-OSS."""
    global _HARMONY_ENC
    if _HARMONY_ENC is None:
        _HARMONY_ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _HARMONY_ENC


def parse_harmony_response(raw_content: str) -> str:
    """Parse GPT-OSS harmony format, extracting final channel content.

    Returns the 'final' channel content, or raw content if parsing fails.
    """
    enc = _get_harmony_enc()
    try:
        tokens = enc.encode(raw_content, allowed_special="all")
        parsed = enc.parse_messages_from_completion_tokens(
            tokens, role=Role.ASSISTANT, strict=False
        )

        for msg in parsed:
            if msg.channel == "final":
                return "".join(c.text for c in msg.content if hasattr(c, "text"))

        if parsed:
            return "".join(c.text for c in parsed[-1].content if hasattr(c, "text"))
    except Exception:
        pass

    return raw_content


def _is_retryable(exc: Exception) -> bool:
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "code", None)
        or getattr(exc, "status", None)
    )
    if status == 429 or (isinstance(status, int) and 500 <= status < 600):
        return True
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "429",
            "rate limit",
            "too many requests",
            "retry in",
            "timed out",
            "timeout",
            "connection error",
        )
    )


def _completion_cost_usd(response: Any, usage: Any, allow_price_table: bool) -> float:
    """Prefer the provider's own cost, then LiteLLM's price table."""
    reported = []
    if usage is not None:
        reported.append(getattr(usage, "cost", None))
        extra = getattr(usage, "model_extra", None) or {}
        if isinstance(extra, dict):
            reported.append(extra.get("cost"))
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        reported.append(hidden.get("response_cost"))
    for value in reported:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    if not allow_price_table:
        return 0.0
    try:
        return float(completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


def _extract_content(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "".join(parts)
    return ""


class LLMCallable(Protocol):
    """Protocol for LLM call functions."""

    def __call__(self, prompt: str) -> str: ...


class ProviderLLM:
    """Thin shim that provides the old provider interface on top of litellm."""

    def __init__(
        self,
        model: str,
        max_concurrent: int = 4,
        api_key: str | None = None,
        api_base: str | None = None,
        cache_mode: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.cache_mode = cache_mode or os.environ.get("TEXT_CLASSIFICATION_LLM_CACHE_MODE", "run")
        supplied_cache = cache_dir or os.environ.get("TEXT_CLASSIFICATION_LLM_CACHE_DIR")
        if self.cache_mode not in {"off", "run"}:
            raise ValueError("Pilot cache mode must be off or run")
        if self.cache_mode == "run" and not supplied_cache:
            raise ValueError("run cache requires an explicit isolated cache directory")
        self.cache_dir = Path(supplied_cache) if supplied_cache else None
        self.logical_calls = 0
        self.api_requests = 0
        self.cache_hits = 0
        self.failed_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self._usage_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        if self.cache_mode == "run":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __exit__(self, *exc):
        self._executor.shutdown(wait=True)
        return False

    def _normalized_model(self) -> str:
        if self.api_base and not self.model.startswith(KNOWN_PROVIDER_PREFIXES):
            return f"openai/{self.model}"
        return self.model

    def _cache_path(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> Path:
        payload = {
            "version": CACHE_VERSION,
            "model": self._normalized_model(),
            "api_base": self.api_base,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "kwargs": kwargs,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return (self.cache_dir or Path(".")) / f"{digest}.json"

    def _load_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            with self._cache_lock:
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                tmp.replace(path)
        except OSError:
            pass

    def _call_completion(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        call_kwargs = dict(kwargs)
        call_kwargs["timeout"] = 600.0
        if self.api_base:
            call_kwargs["base_url"] = self.api_base
            call_kwargs.setdefault("api_key", self.api_key or "local")
        elif self.api_key is not None:
            call_kwargs["api_key"] = self.api_key

        model = self._normalized_model()
        # Retries are counted here by _generate_one; disable hidden SDK retries.
        response = litellm_completion(model=model, messages=messages, num_retries=0, **call_kwargs)
        content = _extract_content(response)

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None:
            try:
                input_tokens = token_counter(model=model, messages=messages)
            except Exception:
                input_tokens = 0
        if output_tokens is None:
            try:
                output_tokens = token_counter(model=model, text=content)
            except Exception:
                output_tokens = 0

        cost = _completion_cost_usd(response, usage, allow_price_table=not self.api_base)
        raw_usage = usage.model_dump() if hasattr(usage, "model_dump") else {}
        provider_cost = raw_usage.get("cost")
        raw_response = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "choices": [c.model_dump() if hasattr(c, "model_dump") else str(c) for c in response.choices],
            "usage": raw_usage,
        }

        return {
            "content": content,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cost": cost,
            "reported_usd": float(provider_cost) if provider_cost is not None else None,
            "estimated_usd": cost if provider_cost is None else None,
            "cost_source": "provider_usage" if provider_cost is not None else "litellm_estimate",
            "raw_response": raw_response,
            "provider_usage": raw_usage,
        }

    def _generate_one(
        self, prompt: str, system_prompt: str | None, kwargs: dict[str, Any]
    ) -> str:
        started = time.monotonic()
        call_id = uuid.uuid4().hex
        call = {"call_id": call_id, "model": self._normalized_model(), "prompt": prompt,
                "system_prompt": system_prompt, "parameters": kwargs, "cache_mode": self.cache_mode}
        with self._usage_lock:
            self.logical_calls += 1
        cache_path = self._cache_path(prompt, system_prompt, kwargs)
        cached = self._load_cache(cache_path) if self.cache_mode == "run" else None
        if cached is not None:
            with self._usage_lock:
                self.cache_hits += 1
            record_call({**call, "event": "cache_hit", "content": cached["content"],
                         "raw_response": cached.get("raw_response"), "cost": 0.0,
                         "wall_seconds": time.monotonic() - started})
            return cached["content"]

        result = None
        for attempt in range(4):  # One initial request and at most three retries.
            attempt_id = f"{call_id}:{attempt}"
            attempt_start = time.monotonic()
            record_call({**call, "event": "api_started", "attempt_id": attempt_id, "attempt": attempt})
            with self._usage_lock:
                self.api_requests += 1
            try:
                result = self._call_completion(prompt, system_prompt, kwargs)
            except Exception as exc:
                with self._usage_lock:
                    self.failed_requests += 1
                status = getattr(exc, "status_code", None)
                record_call({**call, "event": "api_failed", "attempt_id": attempt_id,
                             "error_type": type(exc).__name__, "http_status": status,
                             "cost": None, "cost_source": "unreported_failure",
                             "wall_seconds": time.monotonic() - attempt_start})
                if status in (401, 403) or attempt == 3 or not _is_retryable(exc):
                    raise
                time.sleep(min(32, 2 ** (attempt + 1)))
                continue
            record_call({**call, **result, "event": "api_result", "attempt_id": attempt_id,
                         "wall_seconds": time.monotonic() - attempt_start})
            break
        if self.cache_mode == "run":
            self._save_cache(cache_path, result)

        with self._usage_lock:
            self.total_input_tokens += result["input_tokens"]
            self.total_output_tokens += result["output_tokens"]
            self.total_cost += result["cost"]

        return result["content"]

    def generate(self, prompts, system_prompt: str | None = None, **kwargs):
        if isinstance(prompts, str):
            return [[self._generate_one(prompts, system_prompt, kwargs)]]

        prompts = list(prompts)
        if not prompts:
            return []

        results = [None] * len(prompts)
        futures = {
            submit_with_context(self._executor,
                self._generate_one, prompt, system_prompt, kwargs
            ): idx
            for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
        return [[content] for content in results]


class LLM:
    """LLM caller backed by litellm. Handles caching, batching, retries."""

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-120b",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 16384,
        max_workers: int = 32,
        cache_mode: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.model = model
        self.is_gpt_oss = "gpt-oss" in model.lower()

        if temperature is not None:
            self.temperature = temperature
        elif "gpt-5" in model or self.is_gpt_oss:
            self.temperature = 1.0
        else:
            self.temperature = 0.0

        if api_base is not None and not self.is_gpt_oss and max_tokens > 4096:
            max_tokens = 4096
        self.max_tokens = max_tokens

        self._model_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        self._system_prompt = "Reasoning: medium" if self.is_gpt_oss else None

        if api_base is None and max_workers > 4:
            max_workers = 4
        self._provider = ProviderLLM(
            model=model,
            max_concurrent=max_workers,
            api_key=api_key,
            api_base=api_base,
            cache_mode=cache_mode,
            cache_dir=cache_dir,
        )

        self._usage_lock = threading.Lock()
        self.total_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if hasattr(self._provider, "__exit__"):
            self._provider.__exit__(*exc)
        return False

    @property
    def total_input_tokens(self):
        return self._provider.total_input_tokens

    @property
    def total_output_tokens(self):
        return self._provider.total_output_tokens

    @property
    def total_cost(self):
        return self._provider.total_cost

    def _truncate(self, prompt: str) -> str:
        if len(prompt) > MAX_PROMPT_CHARS:
            original_len = len(prompt)
            half = MAX_PROMPT_CHARS // 2
            prompt = prompt[:half] + "\n\n... [TRUNCATED] ...\n\n" + prompt[-half:]
            logger.warning(
                "Prompt truncated: %d -> %d chars (limit %d)",
                original_len,
                len(prompt),
                MAX_PROMPT_CHARS,
            )
        return prompt

    def __call__(self, prompt: str) -> str:
        """Single prompt call."""
        prompt = self._truncate(prompt)

        results = self._provider.generate(
            prompt, system_prompt=self._system_prompt, **self._model_kwargs
        )
        content = results[0][0]

        with self._usage_lock:
            self.total_calls += 1

        if self.is_gpt_oss:
            content = parse_harmony_response(content)
        return content

    def get_usage(self) -> dict[str, Any]:
        """Return token usage stats and estimated cost."""
        with self._usage_lock:
            calls = self.total_calls
        return {
            "model": self.model,
            "calls": calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "cost_usd_unrounded": self.total_cost,
            "logical_calls": self._provider.logical_calls,
            "api_requests": self._provider.api_requests,
            "cache_hits": self._provider.cache_hits,
            "failed_requests": self._provider.failed_requests,
        }

    def reset_usage(self):
        """Reset token counters."""
        with self._usage_lock:
            self.total_calls = 0
        self._provider.total_input_tokens = 0
        self._provider.total_output_tokens = 0
        self._provider.total_cost = 0.0
        self._provider.logical_calls = 0
        self._provider.api_requests = 0
        self._provider.cache_hits = 0
        self._provider.failed_requests = 0

    def batch(self, prompts: list[str]) -> list[str]:
        """Parallel batch call. Returns results in same order as prompts."""
        if not prompts:
            return []

        prompts = [self._truncate(p) for p in prompts]
        results = self._provider.generate(
            prompts, system_prompt=self._system_prompt, **self._model_kwargs
        )
        contents = [r[0] for r in results]

        with self._usage_lock:
            self.total_calls += len(prompts)

        if self.is_gpt_oss:
            contents = [parse_harmony_response(c) for c in contents]
        return contents


def make_local_llm(
    model: str = "gpt-oss-120b",
    host: str = os.environ.get("LOCAL_LLM_HOST", "localhost"),
    port: int = int(os.environ.get("LOCAL_LLM_PORT", "30000")),
    max_tokens: int = 4096,
    max_workers: int = 16,
) -> LLM:
    """Create a local LLM caller against an OpenAI-compatible server.

    Usage:
        llm = make_local_llm()
        response = llm("Hello!")
    """
    return LLM(
        model=model,
        api_base=f"http://{host}:{port}/v1",
        max_tokens=max_tokens,
        max_workers=max_workers,
    )


def make_stub_llm(response: str = '{"reasoning": "stub", "final_answer": "stub"}'):
    """Create a stub LLM caller for testing."""
    return lambda prompt: response
