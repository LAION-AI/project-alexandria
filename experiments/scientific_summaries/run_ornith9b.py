"""Pinned Ornith cohort: two owned GPU replicas, strict repairs, unchanged fixed student."""
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
from finish_qwen9b import normalize_claim_shapes, validate_existing
from project_alexandria.io import write_json_atomic
from queue_control import AlreadyRunning, exclusive_lock
from run_summary_comparison import MODELS, free_gpus, ready, status, stop, weights
from summarize import KEYS, system_prompt
from summary_runtime import SummaryClient, generate_document, parse_json
from summary_repair_v3 import (PROTOCOL, normalize_shapes, repair_document,
                               FIELD_USER_TEMPLATE, ANCHOR_USER_TEMPLATE, field_example)

POLICY = 'ornith_v3_claim_shapes_64k_two_replicas_v1'


def prepare_draft(failure):
    """Recover original V2 detail, then mark only shape-aligned drafts for V3 repair."""
    result = copy.deepcopy(failure)
    if result.get('repair_protocol') != PROTOCOL:
        for record in result.get('attempts', []):
            if record.get('phase') != 'generation' or record.get('finish_reason') == 'length':
                continue
            try:
                value, _ = parse_json(record['response'])
                if isinstance(value, dict) and set(value) == set(KEYS):
                    result['draft_summary'] = {k: value[k] for k in KEYS}
                    break
            except (ValueError, KeyError, TypeError):
                pass
    value = result.get('draft_summary')
    if isinstance(value, dict) and set(value) == set(KEYS):
        result['draft_summary'], changes = normalize_shapes(value)
        result.setdefault('shape_normalizations', []).extend(changes)
        result = normalize_claim_shapes(result)
        result['repair_protocol'] = PROTOCOL
    return result


def run_paper(paper, client, prompt, previous=None):
    history, elapsed = [], 0.0
    for index in range(3):
        if previous and previous.get('draft_summary'):
            previous = repair_document(paper, client, prepare_draft(previous))
        else:
            previous = generate_document(paper, client, prompt, 16000, 1, seed_offset=index)
            history.extend(previous['attempts'])
            elapsed += previous.get('elapsed_seconds', 0)
            previous['attempts'], previous['elapsed_seconds'] = copy.deepcopy(history), elapsed
            if previous.get('failed') and previous.get('draft_summary'):
                previous = repair_document(paper, client, prepare_draft(previous))
        if not previous.get('failed'):
            return dict(previous, repair_protocol=PROTOCOL, cohort_policy=POLICY,
                        context_limit=65536)
    return previous


def main():
    directory = ROOT / 'summary_runs/ornith15_9b'
    directory.mkdir(exist_ok=True)
    path = directory / 'summaries.json'
    archive = directory / 'v2_failed_archive'
    model = next(m for m in MODELS if m['name'] == 'ornith15_9b')
    papers = load(ROOT / 'data/papers.json')
    selected = set(load(ROOT / 'summary_comparison_manifest.json')['documents'])
    if len(selected) != 97:
        raise ValueError('Expected frozen 97-paper cohort')
    prompt = system_prompt(ROOT / 'summary-systemprompt+.txt')
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = dict(model=model['model'], revision=model['revision'], weights_sha256=model['weights_sha256'],
        system_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        prompt_file_sha256=hashlib.sha256((ROOT / 'summary-systemprompt+.txt').read_bytes()).hexdigest(),
        runtime='llama.cpp', temperature=0.2, top_p=0.95, thinking=False, max_tokens=16000,
        context_limit=65536, concurrency=8, allocated_gpus=2, repair_protocol=PROTOCOL,
        cohort_policy=POLICY, cohort_implementation_sha256=code_sha,
        repair_implementation_sha256=hashlib.sha256((ROOT / 'summary_repair_v3.py').read_bytes()).hexdigest(),
        claim_shape_implementation_sha256=hashlib.sha256((ROOT / 'finish_qwen9b.py').read_bytes()).hexdigest(),
        student_input='all_substantive_narrative_fields_without_grounding_quotes_or_citation_rankings')
    prior = load(path) if path.exists() else None
    if prior and prior['config'].get('cohort_policy') != POLICY:
        if prior['documents'] or (directory / 'results.json').exists():
            raise ValueError('Refusing to replace an existing valid or evaluated run')
        for key in ('model', 'revision', 'weights_sha256', 'system_prompt_sha256', 'prompt_file_sha256'):
            if prior['config'][key] != config[key]:
                raise ValueError('Archived experiment identity changed')
        archive.mkdir(exist_ok=True)
        if (archive / 'summaries.json').exists():
            raise ValueError('Existing archive requires manual reconciliation')
        shutil.copyfile(path, archive / 'summaries.json')
        shutil.copyfile(directory / 'summary_prompt_snapshot.json', archive / 'summary_prompt_snapshot.json')
        cache = dict(config=config, documents=[], failures=prior['failures'],
                     elapsed_seconds=prior.get('elapsed_seconds', 0.0), current_run_seconds=0.0)
    elif prior:
        cache = prior
    else:
        cache = dict(config=config, documents=[], failures=[], elapsed_seconds=0.0, current_run_seconds=0.0)
    if archive.exists():
        config['recovery_checkpoint_sha256'] = hashlib.sha256((archive / 'summaries.json').read_bytes()).hexdigest()
        config['historical_elapsed_seconds_included'] = load(archive / 'summaries.json').get('elapsed_seconds', 0)
    if cache['config'] != config:
        raise ValueError('Run identity/config/implementation changed')
    done = validate_existing(cache, papers, selected)
    by_id = {p['document_id']: p for p in papers}
    for d in cache['failures']:
        if d.get('fulltext_sha256') and d['fulltext_sha256'] != by_id[d['document_id']]['fulltext_sha256']:
            raise ValueError('Failed draft source changed')
    from summary_runtime import REPAIR_SYSTEM, ANCHOR_SYSTEM
    write_json_atomic(str(directory / 'summary_prompt_snapshot.json'), dict(
        effective_system_prompt=prompt, effective_prompt_sha256=config['system_prompt_sha256'],
        file_sha256=config['prompt_file_sha256'], file_name='summary-systemprompt+.txt',
        repair_system_prompt=REPAIR_SYSTEM, anchor_selection_system_prompt=ANCHOR_SYSTEM,
        repair_protocol=PROTOCOL, field_user_template=FIELD_USER_TEMPLATE,
        anchor_user_template=ANCHOR_USER_TEMPLATE, field_shape_examples={k: field_example(k) for k in KEYS},
        cohort_policy=POLICY, cohort_implementation_sha256=code_sha))
    write_json_atomic(str(path), cache)
    (directory / 'documents').mkdir(exist_ok=True)
    pending = [p for p in papers if p['document_id'] in selected - done]
    servers = []
    free_gpus()
    status('loading_summary_model', model=model['name'], remaining=len(pending), replicas=2)
    try:
        with (directory / 'runtime_v3.log').open('a') as log:
            if pending:
                pinned_weights = weights(model)
                clients = []
                for gpu, port in ((0, 8010), (1, 8012)):
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                    process = subprocess.Popen([
                        '/home/c4r33u19/moss15v2/llama.cpp/build/bin/llama-server',
                        '--model', str(pinned_weights), '--alias', model['alias'],
                        '--host', '127.0.0.1', '--port', str(port), '--gpu-layers', '999',
                        '--ctx-size', '262144', '--parallel', '4', '--flash-attn', 'on',
                        '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--jinja',
                        '--reasoning', 'off', '--chat-template-kwargs', '{"enable_thinking":false}',
                        '--threads', '8', '--threads-batch', '16'], env=environment,
                        stdout=log, stderr=log, start_new_session=True)
                    servers.append(process)
                    clients.append(SummaryClient('http://127.0.0.1:%s/v1' % port, model['alias'], 'llama.cpp', 65536))
                # Verify both ports, not merely that the first replica is available.
                import urllib.request
                for process, port in zip(servers, (8010, 8012)):
                    for _ in range(180):
                        if process.poll() is not None:
                            raise RuntimeError('Owned Ornith replica exited during startup')
                        try:
                            with urllib.request.urlopen('http://127.0.0.1:%s/v1/models' % port, timeout=2) as response:
                                if any(m['id'] == model['alias'] for m in json.load(response)['data']):
                                    break
                        except (OSError, ValueError, KeyError):
                            pass
                        time.sleep(5)
                    else:
                        raise RuntimeError('Ornith replica startup timed out')

                def batch(items):
                    last = {d['document_id']: d for d in cache['failures']}
                    started = time.monotonic()
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(run_paper, p, clients[i % 2], prompt,
                            last.get(p['document_id'])): p for i, p in enumerate(items)}
                        for future in as_completed(futures):
                            paper = futures[future]
                            try:
                                document = future.result()
                            except Exception as error:
                                document = dict(document_id=paper['document_id'], failed=True,
                                                fulltext_sha256=paper['fulltext_sha256'], error=str(error))
                            document['cohort_implementation_sha256'] = code_sha
                            if document.get('failed'):
                                cache['failures'].append(document)
                                print('FAILED', paper['document_id'], document.get('validation_errors', document.get('error')), flush=True)
                            else:
                                cache['documents'].append(document)
                                write_json_atomic(str(directory / 'documents' / (document['document_id'] + '.json')), document)
                                print('COMPLETED', len(cache['documents']), '/97', paper['document_id'], flush=True)
                            delta = time.monotonic() - started
                            cache['elapsed_seconds'] += delta
                            cache['current_run_seconds'] += delta
                            started = time.monotonic()
                            write_json_atomic(str(path), cache)

                status('summary_smoke_test', model=model['name'], replicas=2)
                batch(pending[:2])
                done = validate_existing(cache, papers, selected)
                if not {p['document_id'] for p in pending[:2]} <= done:
                    raise ValueError('Ornith smoke test failed; all attempts preserved')
                status('generating_summaries', model=model['name'], papers=97, replicas=2, concurrency=8)
                batch([p for p in pending if p['document_id'] not in done])
                done = validate_existing(cache, papers, selected)
                if done != selected:
                    status('retrying_failed_summaries', model=model['name'], remaining=len(selected - done))
                    batch([p for p in pending if p['document_id'] not in done])
                for process in servers:
                    stop(process)
            if validate_existing(cache, papers, selected) != selected:
                raise ValueError('Incomplete Ornith cohort; no denominator change')
            free_gpus()
            status('loading_fixed_student', model=model['name'], papers=97)
            environment = dict(os.environ, ALEXANDRIA_JUDGE_GPU='0', ALEXANDRIA_JUDGE_MAX_LEN='32768',
                               ALEXANDRIA_JUDGE_GPU_UTIL='0.90', ALEXANDRIA_JUDGE_PORT='8010')
            judge = subprocess.Popen(['bash', str(ROOT / 'serve_judge.sh')], env=environment,
                                     stdout=log, stderr=log, start_new_session=True)
            servers.append(judge)
            ready(judge, 'qwen25')
            status('evaluating_summaries', model=model['name'], papers=97)
            skip = [x for identifier in sorted({p['document_id'] for p in papers} - selected)
                    for x in ('--skip-id', identifier)]
            subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--with-kus',
                '--with-qwen-summaries', '--context-limit', '32768', '--summary-cache', str(path),
                '--output', str(directory / 'results.json'), '--baseline-checkpoint',
                str(ROOT / 'pre_qwen_summary_results.json')] + skip, stdout=log, stderr=log, check=True)
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
