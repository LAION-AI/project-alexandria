"""Inference adapters for OpenAI-compatible servers and offline vLLM."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Protocol, Sequence


class GenerationBackend(Protocol):
    model_name: str

    def generate(self, system: str, prompt: str, max_tokens: Optional[int] = None) -> str:
        ...

    def generate_batch(
        self, system: str, prompts: Sequence[str], max_tokens: Optional[int] = None
    ) -> List[str]:
        ...


class OpenAICompatibleBackend:
    """Minimal client for vLLM, llama.cpp, Ollama proxies, and hosted APIs."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        timeout: int = 300,
        concurrency: int = 8,
        retries: int = 3,
        thinking: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> None:
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("ALEXANDRIA_API_KEY", "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.concurrency = concurrency
        self.retries = retries
        self.thinking = thinking
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def generate(self, system: str, prompt: str, max_tokens: Optional[int] = None) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.95,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "chat_template_kwargs": {"enable_thinking": self.thinking},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload, headers=headers, method="POST"
        )
        last_error = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                content = choice["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        "server returned no final message content (finish_reason={!r}); "
                        "disable thinking or increase max_tokens".format(choice.get("finish_reason"))
                    )
                return content
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError("generation failed after {} attempts: {}".format(self.retries, last_error))

    def generate_batch(
        self, system: str, prompts: Sequence[str], max_tokens: Optional[int] = None
    ) -> List[str]:
        # Concurrent requests are continuously batched by a vLLM server.
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(self.generate, system, prompt, max_tokens) for prompt in prompts
            ]
            return [future.result() for future in futures]


class VLLMBackend:
    """Offline vLLM backend; imports vLLM only when selected."""

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        thinking: bool = False,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError("install the 'vllm' optional dependency") from error
        self.model_name = model
        self._sampling_params = SamplingParams
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._chat_template_kwargs = {"enable_thinking": thinking}
        kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "enable_prefix_caching": True,
        }
        if tokenizer:
            kwargs["tokenizer"] = tokenizer
        self._llm = LLM(**kwargs)

    def generate(self, system: str, prompt: str, max_tokens: Optional[int] = None) -> str:
        return self.generate_batch(system, [prompt], max_tokens=max_tokens)[0]

    def generate_batch(
        self, system: str, prompts: Sequence[str], max_tokens: Optional[int] = None
    ) -> List[str]:
        conversations = [
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        sampling = self._sampling_params(
            temperature=self._temperature,
            top_p=0.95,
            max_tokens=max_tokens or self._max_tokens,
        )
        outputs = self._llm.chat(
            conversations,
            sampling,
            use_tqdm=True,
            chat_template_kwargs=self._chat_template_kwargs,
        )
        return [output.outputs[0].text for output in outputs]
