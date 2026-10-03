"""Audit Ornith token journals and extrapolate explicitly hypothetical GH200 rates."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluate import ROOT, load
from project_alexandria.io import write_json_atomic


def token_workload(directory):
    cache = load(directory / 'summaries.json')
    first, calls, seen = {}, [], set()
    for record in cache.get('failures', []) + cache['documents']:
        if record.get('raw_journal_file'):
            file = directory / record['raw_journal_file']
            file.resolve().relative_to(directory.resolve())
            if hashlib.sha256(file.read_bytes()).hexdigest() != record['raw_journal_sha256']:
                raise ValueError('Raw journal checksum mismatch')
            record = load(file)
        identifier = record['document_id']
        for call in record.get('attempts', []):
            raw = call['response']  # Reject a lean journal which was not hydrated.
            key = (identifier, call.get('phase'), call.get('seed'),
                call.get('user_prompt_sha256'), call.get('system_prompt_sha256'),
                hashlib.sha256(raw.encode()).hexdigest(), call.get('elapsed_seconds'))
            if key in seen:
                continue
            seen.add(key)
            calls.append(call)
            if call['phase'] == 'generation' and identifier not in first:
                first[identifier] = call
    documents = len(cache['documents'])
    if len(first) != documents:
        raise ValueError('Initial generations do not cover completed cohort')

    def total(items):
        return dict(calls=len(items), prompt_tokens=sum(c['usage']['prompt_tokens'] for c in items),
            output_tokens=sum(c['usage']['completion_tokens'] for c in items),
            cached_input_tokens=sum(c['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0)
                                    for c in items))
    initial = total(list(first.values()))
    all_calls = total(calls)
    correction = {k: all_calls[k] - initial[k] for k in all_calls}
    return dict(papers=documents, phase_calls=dict(Counter(c['phase'] for c in calls)),
        initial_generation=initial, correction_including_regeneration=correction, total=all_calls)


def gpu_hours(work, corpus, papers, prefill, decode, utilization, use_observed_cache):
    inputs = work['prompt_tokens'] - (work['cached_input_tokens'] if use_observed_cache else 0)
    return corpus / papers / 3600 / utilization * (inputs / prefill + work['output_tokens'] / decode)


def estimate(directory, corpus):
    work = token_workload(directory)
    scenarios = {}
    for name, prefill, decode, utilization in (
            ('optimistic', 40000, 4000, 0.90),
            ('central', 25000, 2500, 0.85),
            ('cautious', 15000, 1500, 0.80)):
        scenarios[name] = dict(assumed_prefill_tokens_per_second_per_gpu=prefill,
            assumed_output_tokens_per_second_per_gpu=decode, useful_capacity_fraction=utilization)
        for use_cache, label in ((True, 'observed_cache_proxy'), (False, 'no_prefix_cache')):
            scenarios[name][label] = {phase: gpu_hours(work[phase], corpus, work['papers'],
                prefill, decode, utilization, use_cache) for phase in (
                    'initial_generation', 'correction_including_regeneration', 'total')}
    return dict(corpus_papers=corpus, measured_token_workload=work, scenarios=scenarios,
        measurement='97-paper RTX3090 llama.cpp Q8 run, including failures and successful retries; '
                    'raw calls deduplicated across copied journals',
        assumptions=['Rates are hypothetical aggregate per-GPU serving capacities, not Ornith GH200 measurements.',
            'FP8 target, TP=1 per GH200, well-filled continuous batches, optimized runtime assumed.',
            'Additive prefill/decode equivalent-work model; useful-capacity factor reserves overhead.',
            'Observed llama.cpp cached tokens are only a cache proxy, not verified vLLM hybrid-state reuse.',
            '60M documents assumed to have the same length, output and repair distribution as this small cohort.',
            'QA student, PDF extraction, CPU validation, storage and queue waiting excluded.',
            'Historical pathological retries retained; improving repair policy requires a fresh quality benchmark.',
            'No DFlash speedup assumed; pilot and GH200 high-batch performance must be measured separately.'],
        references=dict(jupiter='https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html',
            related_architecture_h100='https://docs.gpustack.ai/2.1/performance-lab/qwen3.5-9b/h100/',
            model='https://huggingface.co/ornith-ai/Ornith-1.5-9B'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--papers', type=int, default=60000000)
    parser.add_argument('--directory', type=Path, default=ROOT / 'summary_runs/ornith15_9b')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.papers <= 0:
        parser.error('Paper count must be positive')
    result = estimate(args.directory, args.papers)
    if args.output:
        write_json_atomic(str(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
