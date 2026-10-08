"""Pinned inputs and atomic checkpoint helpers for the 97-paper KU experiment."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(os.environ.get("ALEXANDRIA_KU_RUN", "/e/fscratch/reformo/schuhmann1/scientific-qwen27-ku1000-97-20261008"))
sys.path.insert(0, str(ROOT / "vendor/src"))
sys.path.insert(0, str(ROOT / "vendor/audit"))


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


class TracedBackend:
    """Complete HTTP traces and context checks, with the repository backend interface."""
    def __init__(self, endpoint, alias, phase, *, max_tokens, temperature,
                 concurrency=4, penalties=0.0, context_limit=32768):
        self.endpoint, self.model_name, self.phase = endpoint, alias, phase
        self.max_tokens, self.temperature = max_tokens, temperature
        self.concurrency, self.context_limit = concurrency, context_limit
        self.penalties = penalties
        self.local = threading.local()

    def post(self, route, payload):
        import urllib.request
        request = urllib.request.Request(self.endpoint + route,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)

    def count(self, system, prompt):
        return self.post("/tokenize", dict(model=self.model_name, messages=[
            dict(role="system", content=system), dict(role="user", content=prompt)],
            add_generation_prompt=True, chat_template_kwargs={"enable_thinking": False}))["count"]

    def generate(self, system, prompt, max_tokens=None):
        maximum = max_tokens or self.max_tokens
        count = self.count(system, prompt)
        if count + maximum > self.context_limit:
            raise ValueError(f"Context overflow: {count}+{maximum}>{self.context_limit}; no truncation")
        payload = dict(model=self.model_name, messages=[dict(role="system", content=system),
            dict(role="user", content=prompt)], max_tokens=maximum,
            temperature=self.temperature, top_p=.95,
            frequency_penalty=self.penalties, presence_penalty=self.penalties,
            chat_template_kwargs={"enable_thinking": False})
        if self.phase == "extraction":
            payload.update(response_format={"type": "json_object"},
                           seed=getattr(self.local, "seed", 250219413))
        started = time.monotonic()
        last_error = None
        for attempt in range(3):
            try:
                response = self.post("/v1/chat/completions", payload)
                choice = response["choices"][0]
                text = choice["message"].get("content")
                trace = dict(phase=self.phase, request=payload, response=response,
                    prompt_tokens_preflight=count, elapsed_seconds=time.monotonic()-started,
                    job_id=os.environ.get("SLURM_JOB_ID"),
                    document_id=getattr(self.local, "document_id", None),
                    chunk_index=getattr(self.local, "chunk_index", None))
                name = f"{time.time_ns()}-{threading.get_ident()}.json"
                path = ROOT / "traces" / self.phase / name
                write(path, trace)
                self.local.last = dict(trace_file=str(path.relative_to(ROOT)),
                    finish_reason=choice.get("finish_reason"), usage=response.get("usage", {}),
                    elapsed_seconds=trace["elapsed_seconds"], prompt_sha256=digest(prompt),
                    prompt_tokens=count, response=text)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Empty final content; raw response retained")
                return text
            except Exception as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"HTTP generation failed after three attempts: {last_error}")


def sources():
    papers = load(ROOT / "inputs/sources_only.json")
    if len(papers) != 97:
        raise ValueError("Expected all 97 frozen sources")
    for paper in papers:
        if digest(paper["fulltext"]) != paper["fulltext_sha256"]:
            raise ValueError("Source checksum changed")
        if set(paper) != {"document_id", "fulltext", "fulltext_sha256", "source"}:
            raise ValueError("Extraction inputs must contain source text and metadata only")
    return papers
