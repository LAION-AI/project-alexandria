import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from common import ROOT, Client, PROMPT, load, write, source_user, correction_user, digest

def memory(gpu):
    return subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip()

def run_benchmark(endpoint, mode, gpu):
    samples = load(ROOT / 'inputs/throughput_sources_only.json')
    client = Client(endpoint, 'summary-model', ROOT / 'outputs/throughput' / mode / 'calls')
    output = ROOT / 'outputs/throughput' / (mode + '.json')
    results = dict(mode=mode, gpu=gpu, output_budget=512, probe_only=True,
        note='512-token bounded output probes; complete-paper timings reported separately', batches=[])
    # Warm all kernels, JSON grammar and tokenizer before measuring steady state.
    client.generate(PROMPT, source_user(samples[0]), 64, 1701, bounded_probe=True)
    for phase in ('generation', 'correction'):
        for concurrency in (1, 4, 8, 16, 32, 64):
            client.post('/reset_prefix_cache', {})
            prompts = []
            for sample in samples[:concurrency]:
                if phase == 'generation':
                    prompt = source_user(sample)
                else:
                    prompt = correction_user(sample, sample['summary'], dict(
                        instruction='Check every narrative statement against the source and preserve supported detail.'))
                prompts.append(prompt)
            before = client.metrics()
            # Two measured rounds expose cold-source prefill and cached-source decode.
            for cache_state in ('cold', 'warm'):
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    records = list(executor.map(lambda item: client.generate(PROMPT, item[1],512,
                        42000+item[0], phase=phase, bounded_probe=True), enumerate(prompts)))
                seconds = time.monotonic()-started
                tokens = sum(x['usage'].get('completion_tokens',0) for x in records)
                entry = dict(phase=phase, concurrency=concurrency, cache_state=cache_state,
                    seconds=seconds, completion_tokens=tokens, output_tokens_per_second=tokens/seconds,
                    request_latency_mean_seconds=sum(x['elapsed_seconds'] for x in records)/len(records),
                    input_tokens=sum(x['input_tokens_preflight'] for x in records), gpu_memory=memory(gpu),
                    finish_reasons=[x['finish_reason'] for x in records])
                results['batches'].append(entry)
                write(output, results)
                print('BENCHMARK',mode,phase,concurrency,cache_state,round(tokens/seconds,2),flush=True)
            after = client.metrics()
            metric_dir = ROOT/'outputs/throughput'/mode
            metric_dir.mkdir(parents=True,exist_ok=True)
            (metric_dir/(phase+'-'+str(concurrency)+'-before.prom')).write_text(before)
            (metric_dir/(phase+'-'+str(concurrency)+'-after.prom')).write_text(after)
    results['complete'] = True
    write(output, results)
    return results

def select(results):
    candidates = []
    for result in results:
        if not result.get('complete'):
            continue
        best = {}
        for phase in ('generation','correction'):
            # Use cold inputs for generation and warm source prefixes for correction.
            state = 'cold' if phase == 'generation' else 'warm'
            rows = [r for r in result['batches'] if r['phase']==phase and r['cache_state']==state]
            best[phase] = max(rows,key=lambda r:r['output_tokens_per_second'])
        combined = 2/(1/best['generation']['output_tokens_per_second']+1/best['correction']['output_tokens_per_second'])
        candidates.append(dict(mode=result['mode'], effective_combined_tokens_per_second=combined, phases=best))
    dflash = [r for r in candidates if r['mode'].startswith('dflash')]
    if not dflash:
        raise RuntimeError('No DFlash configuration passed the runtime benchmark')
    winner = max(dflash,key=lambda r:r['effective_combined_tokens_per_second'])
    plan = dict(winner=winner, candidates=candidates, selection='fastest measured DFlash configuration; AR separately reported',
        generation_concurrency=winner['phases']['generation']['concurrency'],
        correction_concurrency=winner['phases']['correction']['concurrency'],
        production_concurrency=min(32,max(winner['phases'][p]['concurrency'] for p in ('generation','correction'))),
        production_note='Up to 97 unique papers across three replicas; bounded probes may overestimate long-output throughput')
    write(ROOT/'outputs/performance_recipe.json',plan)
    return plan
