"""Missing-only Ornith recovery with constrained JSON and unchanged fixed student."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluate import ROOT, load
from finish_qwen9b import validate_existing
from project_alexandria.io import write_json_atomic
from queue_control import AlreadyRunning, exclusive_lock
from run_summary_comparison import MODELS, free_gpus, ready, status, stop, weights
from run_ornith9b import prepare_draft
from summarize import KEYS, GROUNDED, system_prompt
from summary_runtime import SummaryClient, ANCHOR_SYSTEM, REPAIR_SYSTEM, generate_document
from summary_repair_v3 import repair_document

POLICY = 'ornith_missing_only_exact_schema_anchor_ids_v1'


def field_schema(field):
    text = {'type': 'string'}
    absent = {'type': 'string', 'enum': ['']}
    entry = dict(type='object', minProperties=1, maxProperties=1, additionalProperties=text)
    sequence = {'anyOf': [absent, dict(type='array', minItems=1, maxItems=12, items=entry)]}
    if field in GROUNDED:
        return sequence
    if field == 'claims':
        properties = dict(description=text, supporting_evidence=sequence,
                          contradicting_evidence=sequence, implications=sequence)
        claim = dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
        return {'anyOf': [absent, dict(type='array', minItems=1, maxItems=6, items=claim)]}
    if field == 'top_influential_citations':
        citation = dict(type='object', minProperties=2, maxProperties=2,
            properties={'quotes': dict(type='array', minItems=1, maxItems=5, items=text)},
            required=['quotes'], additionalProperties=text)
        return {'anyOf': [absent, dict(type='array', minItems=1, maxItems=5, items=citation)]}
    return text


def output_schema(messages):
    system, user = messages[0]['content'], messages[1]['content']
    if system == ANCHOR_SYSTEM:
        tasks = json.loads(user.split('\n\nTASKS\n', 1)[1])
        properties = {t['id']: {'enum': list(range(len(t['candidates']))) + ['drop']} for t in tasks}
    elif system == REPAIR_SYSTEM:
        field = user.split('\nFIELD\n', 1)[1].split('\n', 1)[0]
        properties = {field: field_schema(field)}
    else:
        properties = {k: field_schema(k) for k in KEYS}
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


class ConstrainedClient(SummaryClient):
    def post(self, endpoint, payload, timeout=1800):
        if endpoint == '/v1/chat/completions':
            payload['response_format'] = {'type': 'json_object', 'schema': output_schema(payload['messages'])}
        return super().post(endpoint, payload, timeout)


def compact_journals(cache, directory):
    """Keep complete raw calls in per-record files, not duplicated in a >100MiB cache."""
    compact = copy.deepcopy(cache)
    for group in ('documents', 'failures'):
        folder = directory / ('documents' if group == 'documents' else 'failure_journals')
        folder.mkdir(exist_ok=True)
        for index, record in enumerate(cache[group]):
            target = folder / (record['document_id'] + '.json' if group == 'documents'
                               else '%04d_%s.json' % (index, record['document_id']))
            write_json_atomic(str(target), record)
            lean = compact[group][index]
            lean['attempts'] = [{k: v for k, v in call.items() if k not in ('response', 'tasks')}
                                for call in record.get('attempts', [])]
            lean['raw_journal_file'] = str(target.relative_to(directory))
            lean['raw_journal_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    compact['journal_storage'] = 'full per-document/per-failure JSON; summary cache retains attempt metadata'
    return compact


def main():
    directory = ROOT / 'summary_runs/ornith15_9b'
    path = directory / 'summaries.json'
    cache = load(path)
    papers = load(ROOT / 'data/papers.json')
    selected = set(load(ROOT / 'summary_comparison_manifest.json')['documents'])
    model = next(m for m in MODELS if m['name'] == 'ornith15_9b')
    for key in ('model', 'revision', 'weights_sha256'):
        if cache['config'][key] != model[key]:
            raise ValueError('Pinned Ornith identity changed')
    prompt = system_prompt(ROOT / 'summary-systemprompt+.txt')
    if hashlib.sha256(prompt.encode()).hexdigest() != cache['config']['system_prompt_sha256']:
        raise ValueError('Initial system prompt changed')
    done = validate_existing(cache, papers, selected)
    pending = [p for p in papers if p['document_id'] in selected - done]
    backup = directory / 'pre_exact_schema_recovery'
    backup.mkdir(exist_ok=True)
    if not (backup / 'summaries.json').exists():
        shutil.copyfile(path, backup / 'summaries.json')
        shutil.copyfile(directory / 'summary_prompt_snapshot.json', backup / 'summary_prompt_snapshot.json')
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    recovery = cache['config'].setdefault('completion_recovery', dict(policy=POLICY,
        implementation_sha256=implementation, affected_document_ids=[p['document_id'] for p in pending],
        original_documents_preserved=len(done), allocated_gpus=1, elapsed_seconds=0.0,
        source_checkpoint_sha256=hashlib.sha256((backup / 'summaries.json').read_bytes()).hexdigest(),
        context_limit=65536, note='Exact 19-field decoding; at most 12 entries per grounded field '
        'and six claims for the three recovery papers; exact allowed anchor IDs/indices; unchanged final validator.'))
    if recovery['implementation_sha256'] != implementation:
        raise ValueError('Recovery implementation changed; reconcile explicitly')
    write_json_atomic(str(directory / 'completion_schema_snapshot.json'), dict(
        policy=POLICY, implementation_sha256=implementation,
        generation_schema=output_schema([{'content': prompt}, {'content': ''}]),
        anchor_schema_policy='Exactly task IDs; only provided candidate indices or drop.',
        original_system_prompt_sha256=cache['config']['system_prompt_sha256']))
    last = {d['document_id']: d for d in cache['failures']}
    for identifier, record in list(last.items()):
        if record.get('raw_journal_file'):
            file = directory / record['raw_journal_file']
            if hashlib.sha256(file.read_bytes()).hexdigest() != record['raw_journal_sha256']:
                raise ValueError('Raw failure journal changed')
            last[identifier] = load(file)
    servers = []
    free_gpus()
    status('loading_ornith_recovery', remaining=len(pending))
    try:
        with (directory / 'completion_runtime.log').open('a') as log:
            environment = dict(os.environ, ALEXANDRIA_JUDGE_GPU='1', ALEXANDRIA_JUDGE_MAX_LEN='32768',
                               ALEXANDRIA_JUDGE_GPU_UTIL='0.90', ALEXANDRIA_JUDGE_PORT='8011')
            judge = subprocess.Popen(['bash', str(ROOT / 'serve_judge.sh')], env=environment,
                                     stdout=log, stderr=log, start_new_session=True)
            servers.append(judge)
            if pending:
                environment['CUDA_VISIBLE_DEVICES'] = '0'
                server = subprocess.Popen([
                    '/home/c4r33u19/moss15v2/llama.cpp/build/bin/llama-server',
                    '--model', str(weights(model)), '--alias', model['alias'],
                    '--host', '127.0.0.1', '--port', '8010', '--gpu-layers', '999',
                    '--ctx-size', '262144', '--parallel', '4', '--flash-attn', 'on',
                    '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--jinja',
                    '--reasoning', 'off', '--chat-template-kwargs', '{"enable_thinking":false}',
                    '--threads', '8', '--threads-batch', '16'], env=environment,
                    stdout=log, stderr=log, start_new_session=True)
                servers.append(server)
                ready(server, model['alias'])
                client = ConstrainedClient('http://127.0.0.1:8010/v1', model['alias'], 'llama.cpp', 65536)
                status('finishing_missing_summaries', model=model['name'], remaining=len(pending))

                def process(paper):
                    previous = last.get(paper['document_id'])
                    for index in range(3):
                        if previous and previous.get('draft_summary'):
                            previous = repair_document(paper, client, prepare_draft(previous))
                        else:
                            prior = previous
                            previous = generate_document(paper, client, prompt, 16000, 1, seed_offset=1000 + index)
                            if prior:
                                previous['attempts'] = prior.get('attempts', []) + previous['attempts']
                                previous['elapsed_seconds'] += prior.get('elapsed_seconds', 0)
                            if previous.get('failed') and previous.get('draft_summary'):
                                previous = repair_document(paper, client, prepare_draft(previous))
                        if not previous.get('failed'):
                            return dict(previous, completion_recovery=dict(policy=POLICY,
                                implementation_sha256=implementation, context_limit=65536))
                    return previous

                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(process, p): p for p in pending}
                    for future in as_completed(futures):
                        document = future.result()
                        if document.get('failed'):
                            cache['failures'].append(document)
                        else:
                            cache['documents'].append(document)
                            write_json_atomic(str(directory / 'documents' / (document['document_id'] + '.json')), document)
                            print('COMPLETED', len(cache['documents']), '/97', document['document_id'], flush=True)
                        delta = time.monotonic() - started
                        cache['elapsed_seconds'] += delta
                        cache['current_run_seconds'] += delta
                        recovery['elapsed_seconds'] += delta
                        started = time.monotonic()
                        write_json_atomic(str(path), cache)
                stop(server)
            if validate_existing(cache, papers, selected) != selected:
                raise ValueError('Incomplete Ornith cohort; failed drafts preserved')
            cache = compact_journals(cache, directory)
            write_json_atomic(str(path), cache)
            import urllib.request
            status('loading_fixed_student', model=model['name'], papers=97)
            for _ in range(180):
                if judge.poll() is not None:
                    raise RuntimeError('Owned fixed student exited')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8011/v1/models', timeout=2) as response:
                        if any(m['id'] == 'qwen25' for m in json.load(response)['data']):
                            break
                except (OSError, ValueError, KeyError):
                    pass
                time.sleep(5)
            else:
                raise RuntimeError('Fixed student startup timed out')
            status('evaluating_summaries', model=model['name'], papers=97)
            skip = [x for identifier in sorted({p['document_id'] for p in papers} - selected)
                    for x in ('--skip-id', identifier)]
            subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--with-kus',
                '--with-qwen-summaries', '--base-url', 'http://127.0.0.1:8011/v1', '--context-limit', '32768',
                '--summary-cache', str(path), '--output', str(directory / 'results.json'),
                '--baseline-checkpoint', str(ROOT / 'pre_qwen_summary_results.json')] + skip,
                stdout=log, stderr=log, check=True)
            subprocess.run([sys.executable, str(ROOT / 'summary_comparison_report.py')], check=True)
        status('complete', papers=97, models=[model['name']], scope='selected_models')
    finally:
        for process in reversed(servers):
            stop(process)


if __name__ == '__main__':
    try:
        with exclusive_lock(ROOT / '.summary_queue.lock'):
            main()
    except AlreadyRunning as error:
        print(str(error), flush=True)
        raise SystemExit(2)
    except Exception as error:
        status('failed', error=str(error))
        raise
