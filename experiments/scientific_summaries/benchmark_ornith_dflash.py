"""Paired, resumable 20-paper BF16 Ornith/DFlash generation and fixed-student QA.

Never overwrites the published 97-paper experiment. Only owned server process
groups are stopped. No credentials are accepted on the command line.
"""
import argparse
import copy
import hashlib
import html
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluate import ROOT, load
from finish_ornith9b import output_schema
from project_alexandria.io import write_json_atomic
from publish_summary_results import SECRET, git
from queue_control import exclusive_lock
from run_ornith9b import run_paper
from run_summary_comparison import stop
from summarize import system_prompt, validate_summary
from summary_comparison_report import paired
from summary_runtime import SummaryClient, digest

TARGET = 'ornith-ai/Ornith-1.5-9B'
TARGET_REV = '489cb97981b8654bcfcf30ce1f94ed1b62e07b53'
DRAFT = 'ornith-ai/Ornith-1.5-9B-DFlash'
DRAFT_REV = '21be3446a606afa67e20c33e68ca37396499248f'
POLICY = 'paired_bf16_exact_schema_v3_grounding_dflash_20_v1'
PORT = 8020
DIRECTORY = ROOT / 'summary_runs/ornith15_9b_dflash20'


def select_papers(papers, eligible):
    """10 per dataset source, fixed before inference; never select by QA outcome."""
    selected = []
    for source in ('arxiv', 'bethgelab'):
        candidates = [p for p in papers if p['document_id'] in eligible
                      and p['viewer_config'] == source]
        candidates.sort(key=lambda p: digest('dflash20-v1:' + p['document_id']))
        if len(candidates) < 10:
            raise ValueError('Insufficient frozen papers for ' + source)
        selected.extend(candidates[:10])
    # Interleave the two sources; identical request submission order in both arms.
    return [selected[i + offset] for i in range(10) for offset in (0, 10)]


def counters(raw):
    result = {}
    for line in raw.splitlines():
        if not line.startswith('vllm:spec_decode_'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if not name.endswith('_total'):
            continue
        value = float(line.rsplit(' ', 1)[1])
        result[name] = result.get(name, 0.0) + value
    return result


def request_metrics():
    with urllib.request.urlopen('http://127.0.0.1:%s/metrics' % PORT, timeout=20) as response:
        return response.read().decode()


class SchemaClient(SummaryClient):
    """Same bounded decoder schema in both arms, not a DFlash-only intervention."""
    def post(self, endpoint, payload, timeout=1800):
        if endpoint == '/v1/chat/completions':
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'scientific_summary', 'strict': True,
                'schema': output_schema(payload['messages'])}}
            payload.update(top_k=-1, min_p=0.0, repetition_penalty=1.0)
        return super().post(endpoint, payload, timeout)


def server_command(args, arm):
    command = [args.server_python, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', TARGET, '--revision', TARGET_REV, '--download-dir', args.cache_dir,
        '--served-model-name', 'ornith-paired', '--host', '127.0.0.1', '--port', str(PORT),
        '--tensor-parallel-size', '2', '--dtype', 'bfloat16', '--max-model-len', '65536',
        '--max-num-seqs', str(args.concurrency), '--max-num-batched-tokens', '4096',
        '--gpu-memory-utilization', '0.90', '--attention-backend', 'FLASH_ATTN',
        '--language-model-only', '--enable-prefix-caching', '--enforce-eager',
        '--no-enable-log-requests', '--generation-config', 'vllm']
    if arm == 'dflash':
        command.extend(['--speculative-config', json.dumps(dict(method='dflash', model=DRAFT,
            revision=DRAFT_REV, num_speculative_tokens=8, draft_tensor_parallel_size=2,
            attention_backend='FLASH_ATTN'))])
    return command


def wait_ready(process):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Owned server exited; inspect server log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:%s/v1/models' % PORT, timeout=2) as response:
                if any(m['id'] == 'ornith-paired' for m in json.load(response)['data']):
                    return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(3)
    raise RuntimeError('Server startup timed out')


def save_record(record, directory, failure_index=None):
    folder = directory / ('documents' if failure_index is None else 'failure_journals')
    folder.mkdir(exist_ok=True)
    name = record['document_id'] + '.json' if failure_index is None else '%04d_%s.json' % (
        failure_index, record['document_id'])
    path = folder / name
    write_json_atomic(str(path), record)
    lean = copy.deepcopy(record)
    lean['attempts'] = [{k: v for k, v in call.items() if k not in ('response', 'tasks')}
                        for call in record.get('attempts', [])]
    lean['raw_journal_file'] = str(path.relative_to(directory))
    lean['raw_journal_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return lean


def hydrate(record, directory):
    if not record.get('raw_journal_file'):
        return record
    path = directory / record['raw_journal_file']
    path.resolve().relative_to(directory.resolve())
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['raw_journal_sha256']:
        raise ValueError('Raw journal checksum mismatch')
    return load(path)


def generation(args, arm, papers, manifest):
    directory = DIRECTORY / arm
    directory.mkdir(exist_ok=True)
    path = directory / 'summaries.json'
    prompt = system_prompt(ROOT / 'summary-systemprompt+.txt')
    config = dict(model=TARGET, revision=TARGET_REV, draft_model=DRAFT if arm == 'dflash' else None,
        draft_revision=DRAFT_REV if arm == 'dflash' else None, policy=POLICY,
        manifest_sha256=digest(json.dumps(manifest, sort_keys=True)), runtime='vLLM 0.27.1',
        precision='BF16', tensor_parallel_size=2, allocated_gpus=2,
        concurrency=args.concurrency, context_limit=65536, max_tokens=16000,
        temperature=0.2, top_p=0.95, thinking=False,
        system_prompt_sha256=digest(prompt), num_speculative_tokens=8 if arm == 'dflash' else 0,
        structured_output='same exact 19-field schema in both arms; unchanged final validator',
        server_command=server_command(args, arm),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cache = load(path) if path.exists() else dict(config=config, documents=[], failures=[],
                                                 elapsed_seconds=0.0, metrics_sessions=[])
    if cache['config'] != config:
        raise ValueError('Run identity changed; use a new output directory, not an overwrite')
    by_id = {p['document_id']: p for p in papers}
    for d in cache['documents']:
        if d['fulltext_sha256'] != by_id[d['document_id']]['fulltext_sha256']:
            raise ValueError('Source text changed')
        _, narrative, _ = validate_summary(json.dumps(d['summary'], ensure_ascii=False),
                                           by_id[d['document_id']]['fulltext'])
        if narrative != d['judge_context']:
            raise ValueError('Student context changed')
        hydrate(d, directory)
    completed = {d['document_id'] for d in cache['documents']}
    pending = [p for p in papers if p['document_id'] not in completed]
    if not pending:
        return
    write_json_atomic(str(path), cache)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1',
        VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_CACHE_ROOT='/mnt/nvme/alexandria-vllm-cache',
        TORCHINDUCTOR_CACHE_DIR='/mnt/nvme/alexandria-inductor-cache')
    with (directory / 'server.log').open('a') as log:
        process = subprocess.Popen(config['server_command'], env=environment,
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            wait_ready(process)
            client = SchemaClient('http://127.0.0.1:%s/v1' % PORT, 'ornith-paired', 'vllm', 65536)
            # Small untimed warmup, separate from the measured paper run and journal.
            warmup = SummaryClient(client.base_url + '/v1', client.alias, 'vllm', 65536)
            warmup.generate('Return JSON only.', 'Return {"ready": true}.', 64, 250219413)
            before = request_metrics()
            (directory / 'metrics_before.txt').write_text(before)
            started = time.monotonic()
            last = {d['document_id']: hydrate(d, directory) for d in cache['failures']}
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = {executor.submit(run_paper, p, client, prompt,
                    last.get(p['document_id'])): p for p in pending}
                checkpoint = started
                for future in as_completed(futures):
                    paper = futures[future]
                    try:
                        document = future.result()
                    except Exception as error:
                        document = dict(document_id=paper['document_id'], failed=True,
                            fulltext_sha256=paper['fulltext_sha256'], error=str(error), attempts=[])
                    if document.get('failed'):
                        cache['failures'].append(save_record(document, directory, len(cache['failures'])))
                        print('FAILED', arm, paper['document_id'], document.get('error',
                              document.get('validation_errors')), flush=True)
                    else:
                        cache['documents'].append(save_record(document, directory))
                        print('COMPLETED', arm, len(cache['documents']), '/20', paper['document_id'], flush=True)
                    now = time.monotonic()
                    cache['elapsed_seconds'] += now - checkpoint
                    checkpoint = now
                    write_json_atomic(str(path), cache)
            after = request_metrics()
            (directory / 'metrics_after.txt').write_text(after)
            prior = counters(before)
            delta = {k: v - prior.get(k, 0) for k, v in counters(after).items()}
            cache['metrics_sessions'].append(dict(wall_seconds=time.monotonic() - started,
                                                   speculative_counters=delta))
            write_json_atomic(str(path), cache)
            if arm == 'dflash' and not delta.get('vllm:spec_decode_num_draft_tokens_total', 0):
                raise RuntimeError('No draft tokens measured: DFlash not demonstrably active')
            if {d['document_id'] for d in cache['documents']} != set(by_id):
                raise RuntimeError('Incomplete arm: retained failures; rerun to resume, never reduce denominator')
        finally:
            stop(process)


def evaluate_arms(args, papers):
    baseline = load(ROOT / 'pre_qwen_summary_results.json')
    identifiers = {p['document_id'] for p in papers}
    baseline['documents'] = [d for d in baseline['documents'] if d['document_id'] in identifiers]
    if len(baseline['documents']) != 20:
        raise ValueError('Missing frozen controls')
    write_json_atomic(str(DIRECTORY / 'fixed_controls.json'), baseline)
    skip = [x for p in load(ROOT / 'data/papers.json') if p['document_id'] not in identifiers
            for x in ('--skip-id', p['document_id'])]
    environment = dict(os.environ, ALEXANDRIA_JUDGE_GPU='0', ALEXANDRIA_JUDGE_PORT='8021',
        ALEXANDRIA_JUDGE_MAX_LEN='32768', ALEXANDRIA_JUDGE_GPU_UTIL='0.90',
        ALEXANDRIA_JUDGE_PYTHON=args.server_python)
    with (DIRECTORY / 'judge.log').open('a') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'serve_judge.sh')], env=environment,
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Fixed student exited; inspect judge.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8021/v1/models', timeout=2) as r:
                        if any(m['id'] == 'qwen25' for m in json.load(r)['data']):
                            break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(3)
            else:
                raise RuntimeError('Fixed student startup timed out')
            for arm in ('dflash', 'baseline'):
                directory = DIRECTORY / arm
                subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--with-kus',
                    '--with-qwen-summaries', '--base-url', 'http://127.0.0.1:8021/v1',
                    '--context-limit', '32768', '--summary-cache', str(directory / 'summaries.json'),
                    '--output', str(directory / 'results.json'), '--baseline-checkpoint',
                    str(DIRECTORY / 'fixed_controls.json')] + skip, check=True)
        finally:
            stop(process)


def report(manifest):
    existing = load(ROOT / 'summary_runs/ornith15_9b/results.json')
    identifiers = {d['document_id'] for d in manifest['documents']}
    old = {d['document_id']: sum(r['predictions']['qwen_summary'] == r['gold'] for r in d['rows'])
           for d in existing['documents'] if d['document_id'] in identifiers}
    counts, stages = {}, {}
    for arm in ('baseline', 'dflash'):
        directory = DIRECTORY / arm
        cache, results = load(directory / 'summaries.json'), load(directory / 'results.json')
        if {d['document_id'] for d in results['documents']} != identifiers:
            raise ValueError('QA denominator mismatch')
        counts[arm] = {d['document_id']: sum(r['predictions']['qwen_summary'] == r['gold']
            for r in d['rows']) for d in results['documents']}
        # Verify no unrelated fixed-control answers changed in the evaluation.
        frozen = {d['document_id']: d for d in load(DIRECTORY / 'fixed_controls.json')['documents']}
        for d in results['documents']:
            for row, original in zip(d['rows'], frozen[d['document_id']]['rows']):
                for condition in ('no_context', 'original', 'summary', 'knowledge_units'):
                    if row['responses'][condition] != original['responses'][condition]:
                        raise ValueError('Frozen control regenerated')
        calls = [a for d in cache['documents'] for a in hydrate(d, directory)['attempts']]
        stages[arm] = dict(qa=results['summary']['qwen_summary'],
            wall_seconds=cache['elapsed_seconds'], gpu_hours=cache['elapsed_seconds'] * 2 / 3600,
            calls=len(calls), prompt_tokens=sum(a['usage']['prompt_tokens'] for a in calls),
            output_tokens=sum(a['usage']['completion_tokens'] for a in calls),
            metrics_sessions=cache['metrics_sessions'], config=cache['config'])
    comparison = dict(manifest=manifest, arms=stages,
        end_to_end_speedup=stages['baseline']['wall_seconds'] / stages['dflash']['wall_seconds'],
        dflash_minus_same_precision_baseline=paired(counts['dflash'], counts['baseline']),
        dflash_minus_historical_q8=paired(counts['dflash'], old),
        historical_q8=dict(correct=sum(old.values()), total=200, accuracy=sum(old.values()) / 200),
        caveats=['20 documents / 200 MCQs is a pilot, not a new 97-paper result.',
                 'New arms use the same BF16 target, TP=2, exact schema and eager serving.',
                 'Historical reference used Q8 GGUF, llama.cpp and mixed repair decoder policies.',
                 'Speedup is warmed generation + repair wall time, excludes loading and QA.',
                 'Stochastic decoding is not expected to be byte-identical; QA CI is paper-bootstrap.',
                 'Local RTX3090 speedup is not a GH200 optimal-batching benchmark.'])
    write_json_atomic(str(DIRECTORY / 'comparison.json'), comparison)
    rows = ''.join('<tr><td>%s</td><td>%.2f%%</td><td>%.1f min</td><td>%.3f</td></tr>' % (
        html.escape(arm), 100 * stages[arm]['qa']['accuracy'], stages[arm]['wall_seconds'] / 60,
        stages[arm]['gpu_hours']) for arm in ('baseline', 'dflash'))
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Ornith DFlash paired pilot</title>'
        '<style>body{font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px}'
        'td,th{padding:12px;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style>'
        '<h1>Ornith DFlash: paired 20-paper pilot</h1><p>Identical frozen papers, 200 MCQs, '
        'Qwen2.5-7B BF16 fixed student; 10 arxiv and 10 bethgelab papers.</p>'
        '<table><tr><th>Arm</th><th>QA accuracy</th><th>Generation + correction</th><th>GPU hours</th></tr>'
        + rows + '</table><p>Measured end-to-end speedup: %.3f×.</p>' % comparison['end_to_end_speedup']
        + '<h2>Limitations</h2><ul>' + ''.join('<li>' + html.escape(x) + '</li>'
            for x in comparison['caveats']) + '</ul><h2>Complete metrics and provenance</h2><pre>'
        + html.escape(json.dumps(comparison, indent=2, ensure_ascii=False)) + '</pre></html>')
    (DIRECTORY / 'comparison.html').write_text(page, encoding='utf-8')
    print('COMPARISON', json.dumps({k: v for k, v in comparison.items()
                                  if k not in ('arms', 'manifest')}), flush=True)


def publish():
    repo = ROOT.parents[1]
    paths = [p for p in DIRECTORY.rglob('*') if p.is_file() and p.suffix != '.log']
    for path in paths:
        if SECRET.search(path.read_bytes()):
            raise ValueError('Potential secret detected; publication stopped')
        if path.stat().st_size > 95 * 1024 * 1024:
            raise ValueError('Artifact too large; publication stopped')
    targets = [str(p.relative_to(repo)) for p in paths]
    git(['add', '--'] + targets)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--'] + targets, cwd=str(repo))
    if changed.returncode == 1:
        git(['commit', '--only', '-m', 'Publish paired 20-paper Ornith DFlash pilot', '--'] + targets)
    elif changed.returncode:
        raise RuntimeError('Could not inspect publication changes')
    git(['push', 'origin', 'main'])
    print('PUBLISHED_DFLASH', git(['rev-parse', 'HEAD']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-python', default='/home/rvv/llm-inference/.venv27/bin/python')
    parser.add_argument('--cache-dir', default='/mnt/nvme/alexandria-hf')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('Concurrency must be positive')
    DIRECTORY.mkdir(exist_ok=True)
    selected = select_papers(load(ROOT / 'data/papers.json'),
                             set(load(ROOT / 'summary_comparison_manifest.json')['documents']))
    manifest = dict(policy=POLICY, documents=[dict(document_id=p['document_id'],
        source=p['viewer_config'], fulltext_sha256=p['fulltext_sha256']) for p in selected],
        selection='10 per dataset source, ascending SHA256(dflash20-v1:document_id), interleaved',
        target=TARGET, target_revision=TARGET_REV, draft=DRAFT, draft_revision=DRAFT_REV)
    path = DIRECTORY / 'manifest.json'
    if path.exists() and load(path) != manifest:
        raise ValueError('Frozen pilot selection changed')
    write_json_atomic(str(path), manifest)
    write_json_atomic(str(DIRECTORY / 'prompt_snapshot.json'), dict(
        effective_system_prompt=system_prompt(ROOT / 'summary-systemprompt+.txt'),
        schema_policy='finish_ornith9b.output_schema; identical both arms',
        repair_prompt_snapshot=load(ROOT / 'summary_runs/ornith15_9b/summary_prompt_snapshot.json')))
    for arm in ('dflash', 'baseline'):
        generation(args, arm, selected, manifest)
    evaluate_arms(args, selected)
    report(manifest)
    if args.publish:
        publish()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with exclusive_lock(ROOT / '.summary_queue.lock'):
        main()
