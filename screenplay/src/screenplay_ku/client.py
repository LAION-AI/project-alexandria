"""Client for a pool of OpenAI-compatible vLLM endpoints.

Parallelism comes from spreading work across independent endpoints, not from piling
concurrency onto one. On a sparse MoE, aggregate throughput measured flat from 1 to 8
concurrent requests against a single server, so per-endpoint concurrency stays low and the
eight GPUs each run their own model copy.

Prefix caching is per-endpoint. A run's shared prefix therefore prefills once per endpoint
(15.8 s measured on a 39.6k-token screenplay) and costs ~0.8 s thereafter.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


DEFAULT_PORTS = tuple(range(8100, 8108))


@dataclass
class CallResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    port: int
    attempts: int


@dataclass
class Endpoint:
    port: int
    host: str = "127.0.0.1"
    lock: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(2))

    @property
    def url(self) -> str:
        return "http://{}:{}/v1/chat/completions".format(self.host, self.port)


class EndpointPool:
    def __init__(
        self,
        ports: Sequence[int] = DEFAULT_PORTS,
        model: str = "qwen3.8-27b",
        *,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        timeout: int = 1800,
        retries: int = 3,
        thinking: bool = False,
    ) -> None:
        self.endpoints = [Endpoint(port) for port in ports]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.thinking = thinking
        self._cursor = 0
        self._cursor_lock = threading.Lock()

    def _next_endpoint(self) -> Endpoint:
        with self._cursor_lock:
            endpoint = self.endpoints[self._cursor % len(self.endpoints)]
            self._cursor += 1
            return endpoint

    def call(
        self,
        system: str,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        pin_port: Optional[int] = None,
    ) -> CallResult:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": 0.95,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ku", "schema": schema, "strict": True},
            }
        payload = json.dumps(body).encode("utf-8")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            endpoint = (
                next(e for e in self.endpoints if e.port == pin_port)
                if pin_port is not None
                else self._next_endpoint()
            )
            request = urllib.request.Request(
                endpoint.url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            started = time.time()
            endpoint.lock.acquire()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.load(response)
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as error:
                last_error = error
                time.sleep(min(2 ** attempt, 20))
                continue
            finally:
                endpoint.lock.release()

            choice = parsed["choices"][0]
            content = choice["message"].get("content")
            finish = choice.get("finish_reason")
            if finish == "length":
                # Truncation would drop trailing units silently. Fail loudly instead.
                last_error = RuntimeError("response hit max_tokens; output truncated")
                continue
            if not isinstance(content, str) or not content.strip():
                last_error = RuntimeError("empty content (finish_reason={!r})".format(finish))
                continue
            usage = parsed.get("usage") or {}
            return CallResult(
                text=content,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                seconds=time.time() - started,
                port=endpoint.port,
                attempts=attempt,
            )
        raise RuntimeError("all {} attempts failed: {}".format(self.retries, last_error))

    def warm(self, system: str, prefix: str) -> List[float]:
        """Prefill the shared prefix once on every endpoint, in parallel.

        Without this the first eight real windows each pay the cold prefill and the timing
        protocol is dominated by an artifact of scheduling.
        """
        results: List[float] = [0.0] * len(self.endpoints)

        def _warm(slot: int, endpoint: Endpoint) -> None:
            started = time.time()
            try:
                self.call(system, prefix + "\n\nReply with the single word: ready.",
                          max_tokens=8, pin_port=endpoint.port)
            except Exception:
                pass
            results[slot] = time.time() - started

        threads = [
            threading.Thread(target=_warm, args=(slot, endpoint))
            for slot, endpoint in enumerate(self.endpoints)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return results

    def health(self) -> List[Tuple[int, bool]]:
        status = []
        for endpoint in self.endpoints:
            try:
                with urllib.request.urlopen(
                    "http://{}:{}/v1/models".format(endpoint.host, endpoint.port), timeout=5
                ):
                    status.append((endpoint.port, True))
            except Exception:
                status.append((endpoint.port, False))
        return status


def run_parallel(
    items: Sequence[Any],
    worker: Callable[[Any], Any],
    *,
    max_workers: int = 16,
    on_done: Optional[Callable[[int, int, Any], None]] = None,
) -> List[Any]:
    """Map ``worker`` over ``items``, preserving order. Failures return the exception."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): index for index, item in enumerate(items)}
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as error:  # recorded, never swallowed
                results[index] = error
            completed += 1
            if on_done:
                on_done(completed, len(items), results[index])
    return results
