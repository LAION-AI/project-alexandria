"""Complete only missing Qwen9B papers, preserving the original 94 outputs byte-for-byte.

Explicit completion-policy provenance, strict final validation, no source truncation.
The fixed student can warm up on the second GPU while extraction finishes.
"""
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
from project_alexandria.io import write_json_atomic
from queue_control import exclusive_lock
from run_summary_comparison import MODELS, free_gpus, ready, status, stop, weights
from summarize import system_prompt, validate_summary
from summary_runtime import SummaryClient
from summary_repair_v3 import PROTOCOL, generate, repair_document

POLICY = 'missing_only_v3_claim_shapes_context64k_v1'


def normalize_claim_shapes(failure):
    """Preserve bare narrative strings; evidence remains invalid until model-grounded."""
    result = copy.deepcopy(failure)
    state = result.get('draft_summary')
    if not isinstance(state, dict) or not isinstance(state.get('claims'), list):
        return result
    changes = result.setdefault('shape_normalizations', [])
    for index, claim in enumerate(state['claims']):
        if not isinstance(claim, dict):
            continue
        for field in ('supporting_evidence', 'contradicting_evidence', 'implications'):
            value = claim.get(field)
            path = ['claims', index, field]
            if isinstance(value, str) and value.strip():
                claim[field] = [{value: ''}]
                changes.append(dict(path=path, operation='bare_narrative_to_unverified_entry'))
            elif isinstance(value, list):
                for j, entry in enumerate(value):
                    if isinstance(entry, str) and entry.strip():
                        value[j] = {entry: ''}
                        changes.append(dict(path=path + [j], operation='bare_narrative_to_unverified_entry'))
    return result


def validate_existing(cache, papers, selected):
    by_id = {p['document_id']: p for p in papers}
    done = set()
    for document in cache['documents']:
        identifier = document['document_id']
        if identifier in done or identifier not in selected:
            raise ValueError('Duplicate or out-of-cohort completed summary')
        paper = by_id[identifier]
        if document['fulltext_sha256'] != paper['fulltext_sha256']:
            raise ValueError('Completed source changed')
        _, narrative, spans = validate_summary(json.dumps(document['summary']), paper['fulltext'])
        # Revalidation starts from saved literal spans, so the old alignment flag and
        # pre-alignment quote need not equal newly recomputed alignment metadata.
        essential = lambda rows: [(s['path'], s['quote'], s['start'], s['end']) for s in rows]
        if narrative != document['judge_context'] or essential(spans) != essential(document['evidence_spans']):
            raise ValueError('Completed summary narrative/evidence ledger changed')
        done.add(identifier)
    return done


def main():
    directory = ROOT / 'summary_runs/qwen35_9b'
    path = directory / 'summaries.json'
    cache = load(path)
    papers = load(ROOT / 'data/papers.json')
    selected = set(load(ROOT / 'summary_comparison_manifest.json')['documents'])
    model = next(m for m in MODELS if m['name'] == 'qwen35_9b')
    for field in ('model', 'revision', 'weights_sha256'):
        if cache['config'][field] != model[field]:
            raise ValueError('Pinned model identity changed')
    prompt = system_prompt(ROOT / 'summary-systemprompt+.txt')
    if hashlib.sha256(prompt.encode()).hexdigest() != cache['config']['system_prompt_sha256']:
        raise ValueError('Initial system prompt changed')
    done = validate_existing(cache, papers, selected)
    pending = [p for p in papers if p['document_id'] in selected - done]
    backup = directory / 'pre_completion_recovery'
    backup.mkdir(exist_ok=True)
    if not (backup / 'summaries.json').exists():
        shutil.copyfile(path, backup / 'summaries.json')
        shutil.copyfile(directory / 'summary_prompt_snapshot.json', backup / 'summary_prompt_snapshot.json')
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache['config']['completion_recovery'] = dict(policy=POLICY, implementation_sha256=code_sha,
        original_checkpoint_sha256=hashlib.sha256((backup / 'summaries.json').read_bytes()).hexdigest(),
        original_documents_preserved=len(load(backup / 'summaries.json')['documents']),
        context_limit=65536, affected_document_ids=sorted(selected - {
            d['document_id'] for d in load(backup / 'summaries.json')['documents']}),
        explanation='Only missing papers: rewrap bare claim narratives with unverified empty quotes; '
                    'same-model anchors or audited drops; 64k context without source truncation.')
    write_json_atomic(str(path), cache)
    last = {d['document_id']: d for d in cache['failures']}
    environment = dict(os.environ)
    servers = []
    free_gpus()
    try:
        with (directory / 'completion_runtime.log').open('a') as log:
            # Unchanged BF16 student weights/decoding, separate localhost port and GPU.
            environment.update(ALEXANDRIA_JUDGE_GPU='1', ALEXANDRIA_JUDGE_MAX_LEN='32768',
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
                status('finishing_missing_summaries', model=model['name'], remaining=len(pending))
                backend = SummaryClient('http://127.0.0.1:8010/v1', model['alias'], 'llama.cpp', 65536)

                def process(paper):
                    previous = last.get(paper['document_id'])
                    for _ in range(3):
                        if previous and previous.get('attempts') and previous.get('draft_summary'):
                            previous = repair_document(paper, backend, normalize_claim_shapes(previous))
                        else:
                            previous = generate(paper, backend, prompt, 16000)
                        if not previous.get('failed'):
                            previous['completion_recovery'] = dict(policy=POLICY, implementation_sha256=code_sha,
                                                                   context_limit=65536)
                            return previous
                    return previous

                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(process, p): p for p in pending}
                    for future in as_completed(futures):
                        document = future.result()
                        if document.get('failed'):
                            cache['failures'].append(document)
                        else:
                            cache['documents'].append(document)
                            write_json_atomic(str(directory / 'documents' / (document['document_id'] + '.json')), document)
                            print('COMPLETED', len(cache['documents']), '/97', document['document_id'], flush=True)
                        cache['elapsed_seconds'] += time.monotonic() - started
                        started = time.monotonic()
                        write_json_atomic(str(path), cache)
                stop(server)
            if validate_existing(cache, papers, selected) != selected:
                raise ValueError('Unresolved summaries remain; no papers silently dropped')
            status('loading_fixed_student', model=model['name'], papers=97)
            # ready() checks port8010; here the judge is deliberately on8011.
            import urllib.request
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
                '--with-qwen-summaries', '--context-limit', '32768', '--base-url', 'http://127.0.0.1:8011/v1',
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
    except Exception as error:
        status('failed', error=str(error))
        raise
