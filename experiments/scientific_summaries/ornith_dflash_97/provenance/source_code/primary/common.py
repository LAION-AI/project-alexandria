import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a'
VENDOR = REPO / 'experiments/scientific_summaries'
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(REPO / 'src'))
from summarize import KEYS, system_prompt, narrative_context, validate_summary
from summary_runtime import parse_json, invalid_fields, digest
from summary_repair_v3 import normalize_shapes, structural_fields, repair_document

PROMPT = system_prompt(VENDOR / 'summary-systemprompt+.txt')
GENERATION_PREFIX = (
    'Apply the system prompt to the supplied dataset paper text only. The dataset may '
    'omit the bibliography; if it does, top_influential_citations must be "". '
    'For unreported data/code or ethics, use "" for the entire field. '
    'Do not reproduce template placeholders. Copy proof quotes literally, including symbols.\n\n'
    'BEGIN_PAPER\n{source}\nEND_PAPER'
)
REVIEW_SYSTEM = '''Audit the supplied scientific summary against ONLY the supplied paper.
Do not use external knowledge, questions, answer keys or imagined facts. Check numeric values,
attribution, causality, contradictions, omissions of central findings, unsupported speculation,
methods and limitations. An exact short quote by itself does not prove a long narrative claim.
Return exactly one JSON object with keys scores, issues, missing_central_facts, verdict.
scores has factual_accuracy, coverage, clarity, faithfulness, scientific_precision, each an integer
from 1 (poor) to 5 (excellent). issues is a list of objects with field, statement, problem,
source_quote, proposed_correction; source_quote must be copied literally from the paper or be
the empty string when the issue is absence of reported evidence. missing_central_facts is a
list of objects with fact and source_quote. verdict is pass or needs_correction. A pass requires
no issues, no missing central facts and all five scores >=4. Do not add filler to meet length
targets, and do not treat unreported fields as errors. Return concise, actionable findings.'''

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def load(path):
    return json.loads(Path(path).read_text())

def source_user(paper):
    if digest(paper['fulltext']) != paper['fulltext_sha256']:
        raise ValueError('Frozen source checksum changed')
    return GENERATION_PREFIX.format(source=paper['fulltext'])

def review_user(paper, summary):
    return 'BEGIN_PAPER\n' + paper['fulltext'] + '\nEND_PAPER\n\nSUMMARY_TO_AUDIT\n' + json.dumps(summary, ensure_ascii=False)

def correction_user(paper, summary, review):
    return source_user(paper) + '\n\nCORRECTION_TASK\n' + (
        'Revise the draft below using the source and the audit. Treat the audit as fallible: '
        'verify every proposed change against the paper. Correct unsupported, contradictory, '
        'misattributed or numerically wrong statements and include omitted central results '
        'only if reported. Preserve correct detail. Do not add filler or speculate. '
        'Return the complete corrected 19-field JSON, not a patch. All proof quotes must be '
        'literal source fragments of at most five words. If the draft is already correct, '
        'preserve it.\nDRAFT\n') + json.dumps(summary, ensure_ascii=False) + '\nAUDIT\n' + json.dumps(review, ensure_ascii=False)

class Client:
    def __init__(self, endpoint, alias, trace_dir=None, context_limit=65536):
        self.endpoint, self.alias = endpoint.rstrip('/'), alias
        self.runtime, self.context_limit = 'vllm', context_limit
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.lock, self.sequence = threading.Lock(), 0

    def post(self, route, payload, timeout=1800):
        request = urllib.request.Request(self.endpoint + route, json.dumps(payload, ensure_ascii=False).encode(), {'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace')[:4000]
            raise RuntimeError('Server HTTP ' + str(error.code) + ': ' + detail) from error

    def token_count(self, messages):
        return self.post('/tokenize', dict(model=self.alias, messages=messages, add_generation_prompt=True,
            chat_template_kwargs={'enable_thinking': False}))['count']

    def metrics(self):
        with urllib.request.urlopen(self.endpoint + '/metrics', timeout=30) as response:
            return response.read().decode()

    def generate(self, system, user, max_tokens, seed, phase='generation', bounded_probe=False):
        messages = [dict(role='system', content=system), dict(role='user', content=user)]
        count = self.token_count(messages)
        if count + max_tokens > self.context_limit:
            raise ValueError('Input plus output budget exceeds context; no truncation permitted: ' + str(count))
        payload = dict(model=self.alias, messages=messages, max_tokens=max_tokens, temperature=0.2,
            top_p=0.95, top_k=-1, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
            seed=seed, chat_template_kwargs={'enable_thinking': False}, response_format={'type': 'json_object'})
        started = time.monotonic()
        body = self.post('/v1/chat/completions', payload)
        elapsed = time.monotonic() - started
        choice = body['choices'][0]
        message = choice['message']
        raw = message.get('content')
        record = dict(response=raw, reasoning_content=message.get('reasoning_content', message.get('reasoning')),
            finish_reason=choice.get('finish_reason'), usage=body.get('usage', {}), input_tokens_preflight=count,
            elapsed_seconds=elapsed, max_tokens=max_tokens, seed=seed, phase=phase,
            response_format=payload['response_format'], system_prompt_sha256=digest(system), user_prompt_sha256=digest(user))
        if self.trace_dir:
            with self.lock:
                self.sequence += 1
                name = str(time.time_ns()) + '-' + str(self.sequence) + '.json'
            path = self.trace_dir / name
            write(path, dict(request=payload, raw_response=body, record=record,
                bounded_throughput_probe=bounded_probe, job_id=os.getenv('SLURM_JOB_ID')))
            record['trace_file'] = str(path.relative_to(ROOT))
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('No final content; raw response preserved')
        return record

def parse_summary(record):
    if record['finish_reason'] == 'length':
        raise ValueError('Incomplete output, token budget exhausted')
    obj, changes = parse_json(record['response'])
    if not isinstance(obj, dict) or set(obj) != set(KEYS):
        raise ValueError('Generation lacks all 19 schema fields')
    obj = {key: obj[key] for key in KEYS}
    obj, shape_changes = normalize_shapes(obj)
    if structural_fields(obj):
        raise ValueError('Ambiguous or malformed schema shape: ' + str(structural_fields(obj)))
    record['normalizations'] = changes + shape_changes
    return obj

def validate_review(record, paper):
    from evaluate import evidence_span
    if record['finish_reason'] == 'length':
        raise ValueError('Review output incomplete')
    obj, _ = parse_json(record['response'])
    expected = {'factual_accuracy', 'coverage', 'clarity', 'faithfulness', 'scientific_precision'}
    if set(obj) != {'scores', 'issues', 'missing_central_facts', 'verdict'} or set(obj['scores']) != expected:
        raise ValueError('Review schema mismatch')
    if any(type(v) is not int or not 1 <= v <= 5 for v in obj['scores'].values()):
        raise ValueError('Invalid review score')
    if not isinstance(obj['issues'], list) or not isinstance(obj['missing_central_facts'], list):
        raise ValueError('Review findings must be lists')
    for issue in obj['issues']:
        if not isinstance(issue, dict) or set(issue) != {'field','statement','problem','source_quote','proposed_correction'}:
            raise ValueError('Issue schema mismatch')
        if issue['field'] not in KEYS or any(not isinstance(v, str) for v in issue.values()):
            raise ValueError('Invalid issue field')
        if issue['source_quote']:
            evidence_span(paper['fulltext'], issue['source_quote'])
    for fact in obj['missing_central_facts']:
        if set(fact) != {'fact','source_quote'} or not all(isinstance(v, str) and v for v in fact.values()):
            raise ValueError('Missing-fact schema mismatch')
        evidence_span(paper['fulltext'], fact['source_quote'])
    obj['model_emitted_verdict'] = obj['verdict']
    obj['verdict'] = 'pass' if not obj['issues'] and not obj['missing_central_facts'] and min(obj['scores'].values()) >= 4 else 'needs_correction'
    obj['assessment_kind'] = 'same_model_source_only_audit_not_independent_ground_truth'
    return obj
