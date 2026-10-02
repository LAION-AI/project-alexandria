"""Owned, resumable GPU queue: 27B, fixed judge, Ornith 9B, judge, Qwen 9B, judge."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from evaluate import ROOT, load, questions_for, question_signature
from project_alexandria.io import write_json_atomic
from queue_control import AdoptedServer, AlreadyRunning, exclusive_lock

MODELS = [
    dict(name='qwen27b', model='Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound',
         revision='4756e3e4871aefd8d7cd5b0f6155ae5490451c1e', runtime='vllm',
         alias='qwen38', allocated_gpus=2, concurrency=8, weights_sha256=''),
    dict(name='ornith15_9b', model='ornith-ai/Ornith-1.5-9B-GGUF',
         revision='abdd624b12ebf020b767fff532ff44fe552b28c3', runtime='llama.cpp',
         alias='summary-model', allocated_gpus=1, concurrency=4,
         filename='Ornith-1.5-9B-Q8_0.gguf',
         weights_sha256='22086870b009dbe9815ee752c48a82de930118a7c5ce5599590892ae03b8b010'),
    dict(name='qwen35_9b', model='unsloth/Qwen3.5-9B-GGUF',
         revision='3885219b6810b007914f3a7950a8d1b469d598a5', runtime='llama.cpp',
         alias='summary-model', allocated_gpus=1, concurrency=4,
         filename='Qwen3.5-9B-Q8_0.gguf',
         weights_sha256='809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4')]


def status(phase, **fields):
    value = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write_json_atomic(str(ROOT / 'summary_comparison_status.json'), value)
    print(phase, fields, flush=True)


def ready(process, alias):
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError('Owned model server exited during startup')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8010/v1/models', timeout=2) as response:
                if any(model['id'] == alias for model in json.load(response)['data']):
                    return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(5)
    raise RuntimeError('Owned model server startup timed out')


def free_gpus():
    for _ in range(120):
        values = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(values) >= 2 and all(int(value.strip()) < 1024 for value in values[:2]):
            return
        time.sleep(5)
    raise RuntimeError('GPUs occupied by another job; no unrelated process was stopped')


def stop(process):
    if process.poll() is None:
        if isinstance(process, AdoptedServer):
            process.assert_identity()
        # Only a new session/process group created here is targeted, never arbitrary GPU PIDs.
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError('Owned server process-group identity changed')
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            if isinstance(process, AdoptedServer):
                process.assert_identity()
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def weights(model):
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=model['model'], filename=model['filename'],
        revision=model['revision'], local_dir='/home/c4r33u19/models/scientific-summary-gguf', token=False))
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    if sha.hexdigest() != model['weights_sha256']:
        raise ValueError('Downloaded GGUF does not match the pinned weight checksum')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=Path, default=ROOT / 'summary-systemprompt+.txt')
    parser.add_argument('--llama-server', default='/home/c4r33u19/moss15v2/llama.cpp/build/bin/llama-server')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--reuse-qwen-server-pid', type=int,
                        help='Adopt only this identity-checked orphaned pinned 27B server')
    args = parser.parse_args()
    papers = load(ROOT / 'data/papers.json')
    baseline_path = ROOT / 'pre_qwen_summary_results.json'
    baseline = load(baseline_path)
    judged = {d['document_id']: d for d in baseline['documents']}
    selected, excluded = [], []
    for index, paper in enumerate(papers):
        questions = questions_for(paper, index)
        previous = judged.get(paper['document_id'])
        if not questions or not previous:
            excluded.append(paper['document_id'])
            continue
        if question_signature(questions) != question_signature(previous['questions']):
            raise ValueError('QA changed since the fixed preceding comparison')
        if not all({'no_context', 'original', 'summary', 'knowledge_units'} <= set(row['predictions'])
                   for row in previous['rows']):
            raise ValueError('Preceding comparison is incomplete')
        selected.append(paper['document_id'])
    if len(selected) != 97:
        raise ValueError('User approved exactly the existing 97-paper cohort')
    write_json_atomic(str(ROOT / 'summary_comparison_manifest.json'), dict(
        documents=selected, excluded_ids=excluded, models=MODELS,
        prompt_file_sha256=hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        protocol='field_local_strict_grounding_v2', attempts=6,
        comparison_note='Same source texts and MCQs; runtime, quantization and batch size differ.'))
    skip = [item for identifier in excluded for item in ('--skip-id', identifier)]
    adopted = AdoptedServer(args.reuse_qwen_server_pid, MODELS[0]) if args.reuse_qwen_server_pid else None
    if adopted:
        ready(adopted, MODELS[0]['alias'])
        status('adopting_existing_summary_server', model=MODELS[0]['name'], server_pid=adopted.pid)
    for model in MODELS:
        directory = ROOT / 'summary_runs' / model['name']
        directory.mkdir(parents=True, exist_ok=True)
        cache, result = directory / 'summaries.json', directory / 'results.json'
        if cache.exists() and result.exists():
            saved_result = load(result)
            if 'summary' in saved_result and len(saved_result['documents']) == len(selected):
                from summary_comparison_report import audit_run
                audit_run(model, saved_result, load(cache), papers, baseline)
                if model['name'] == MODELS[0]['name'] and adopted:
                    stop(adopted)
                    adopted = None
                status('reusing_audited_completed_evaluation', model=model['name'])
                continue
        generation = [sys.executable, str(ROOT / 'summarize.py'), '--prompt', str(args.prompt),
            '--output', str(cache), '--runtime', model['runtime'], '--alias', model['alias'],
            '--model', model['model'], '--revision', model['revision'], '--weights-sha256',
            model['weights_sha256'], '--allocated-gpus', str(model['allocated_gpus']),
            '--concurrency', str(model['concurrency']), '--attempts', '6'] + skip
        environment = dict(os.environ)
        if model['runtime'] == 'vllm':
            server_command = ['bash', str(ROOT / 'serve_extractor.sh')]
        else:
            status('preparing_gguf', model=model['name'])
            model_path = weights(model)
            environment['CUDA_VISIBLE_DEVICES'] = '0'
            server_command = [args.llama_server, '--model', str(model_path), '--alias', model['alias'],
                '--host', '127.0.0.1', '--port', '8010', '--gpu-layers', '999',
                '--ctx-size', '131072', '--parallel', '4', '--flash-attn', 'on',
                '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--jinja',
                '--reasoning', 'off', '--chat-template-kwargs', '{"enable_thinking":false}',
                '--threads', '8', '--threads-batch', '16']
        complete_ids = {d['document_id'] for d in load(cache)['documents']} if cache.exists() else set()
        with (directory / 'runtime.log').open('a', encoding='utf-8') as log:
            if not set(selected) <= complete_ids:
                if model['name'] == MODELS[0]['name'] and adopted:
                    server = adopted
                else:
                    free_gpus()
                    status('loading_summary_model', model=model['name'], papers=len(selected))
                    server = subprocess.Popen(server_command, env=environment, stdout=log, stderr=log,
                                              start_new_session=True)
                try:
                    ready(server, model['alias'])
                    status('summary_smoke_test', model=model['name'], papers=len(selected))
                    subprocess.run(generation + ['--max-new-documents', '2'], stdout=log, stderr=log, check=True)
                    if args.smoke_only:
                        status('smoke_test_passed', model=model['name'])
                        return
                    status('generating_summaries', model=model['name'], papers=len(selected))
                    # Failed documents are retained and retried once, never silently omitted.
                    completed = subprocess.run(generation, stdout=log, stderr=log)
                    if completed.returncode:
                        status('retrying_failed_summaries', model=model['name'])
                        subprocess.run(generation, stdout=log, stderr=log, check=True)
                finally:
                    stop(server)
                    if server is adopted:
                        adopted = None
            elif model['name'] == MODELS[0]['name'] and adopted:
                stop(adopted)
                adopted = None
            free_gpus()
            status('loading_fixed_student', model=model['name'])
            environment.update(ALEXANDRIA_JUDGE_GPU='0', ALEXANDRIA_JUDGE_MAX_LEN='32768',
                               ALEXANDRIA_JUDGE_GPU_UTIL='0.90')
            # Reset CUDA visibility for the fixed judge shell's explicit GPU selection.
            judge = subprocess.Popen(['bash', str(ROOT / 'serve_judge.sh')], env=environment,
                stdout=log, stderr=log, start_new_session=True)
            try:
                ready(judge, 'qwen25')
                status('evaluating_summaries', model=model['name'], papers=len(selected))
                subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--with-kus',
                    '--with-qwen-summaries', '--context-limit', '32768', '--summary-cache', str(cache),
                    '--output', str(result), '--baseline-checkpoint', str(baseline_path)] + skip,
                    stdout=log, stderr=log, check=True)
                subprocess.run([sys.executable, str(ROOT / 'summary_comparison_report.py')],
                               stdout=log, stderr=log, check=True)
            finally:
                stop(judge)
    status('complete', papers=len(selected), models=[m['name'] for m in MODELS],
           report='summary_comparison.html')


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
