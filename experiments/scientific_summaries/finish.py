"""Wait for extraction, then automatically run the fixed judge and render the report.

Only the explicitly supplied extractor PID is stopped, after checking its command.
Missing questions or KUs remain pending instead of being silently scored or excluded.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from evaluate import ROOT, load, questions_for
from project_alexandria.io import write_json_atomic


def status(phase, **fields):
    write_json_atomic(str(ROOT / 'phase_status.json'), dict(
        phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields))
    print(phase, fields, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor-pid', type=int, required=True)
    parser.add_argument('--skip-id', action='append', default=[])
    args = parser.parse_args()
    while True:
        papers = load(ROOT / 'data/papers.json')
        selected = [(i, p) for i, p in enumerate(papers) if p['document_id'] not in args.skip_id]
        cache = load(ROOT / 'kus.json') if (ROOT / 'kus.json').exists() else {'documents': []}
        by_id = {d['document_id']: d for d in cache['documents']}
        missing = [p['document_id'] for _, p in selected if p['document_id'] not in by_id]
        for _, paper in selected:
            if paper['document_id'] in by_id:
                if by_id[paper['document_id']]['fulltext_sha256'] != paper['fulltext_sha256']:
                    raise ValueError('KU source changed; refusing an inconsistent evaluation.')
        bad_qa = []
        for index, paper in selected:
            try:
                if questions_for(paper, index) is None:
                    bad_qa.append(paper['document_id'])
            except (ValueError, KeyError, json.JSONDecodeError):
                bad_qa.append(paper['document_id'])
        status('awaiting_extraction_and_qa', target_papers=len(selected),
               ku_ready=len(selected) - len(missing), qa_pending=bad_qa, excluded_ids=args.skip_id)
        if not missing and not bad_qa:
            break
        time.sleep(30)
    command = (Path('/proc') / str(args.extractor_pid) / 'cmdline').read_bytes()
    if b'vllm.entrypoints.openai.api_server' not in command or b'qwen38' not in command:
        raise RuntimeError('Extractor PID identity changed; will not terminate another process.')
    status('switching_to_fixed_judge')
    os.kill(args.extractor_pid, signal.SIGTERM)
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen('http://127.0.0.1:8010/v1/models', timeout=2).close()
        except Exception:
            break
        time.sleep(1)
    else:
        raise RuntimeError('Extractor did not release the serving port.')
    for _ in range(60):
        used = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        if int(used) < 1024:
            break
        time.sleep(1)
    else:
        raise RuntimeError('GPU 0 is still occupied; refusing to stop unrecognized new processes.')
    judge_python = os.environ.get('ALEXANDRIA_JUDGE_PYTHON', '/home/rvv/llm-inference/.venv27/bin/python')
    judge_weights = os.environ.get('ALEXANDRIA_JUDGE_WEIGHTS', '/home/c4r33u19/models/Qwen2.5-7B-Instruct')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', VLLM_USE_FLASHINFER_SAMPLER='0')
    # This subprocess owns only the new judge; the unrelated application remains stopped.
    with (ROOT / 'judge_runtime.log').open('a', encoding='utf-8') as log:
        judge = subprocess.Popen([judge_python, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', judge_weights, '--served-model-name', 'qwen25', '--host', '127.0.0.1',
            '--port', '8010', '--tensor-parallel-size', '1', '--attention-backend', 'TRITON_ATTN',
            '--enable-prefix-caching', '--max-model-len', '32768', '--max-num-batched-tokens', '4096',
            '--max-num-seqs', '4', '--gpu-memory-utilization', '0.90', '--enforce-eager',
            '--no-enable-log-requests', '--generation-config', 'vllm'], env=environment, stdout=log, stderr=log)
        try:
            for _ in range(120):
                if judge.poll() is not None:
                    raise RuntimeError('Judge startup failed; inspect judge_runtime.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8010/v1/models', timeout=2) as response:
                        if any(m['id'] == 'qwen25' for m in json.load(response)['data']):
                            break
                except Exception:
                    pass
                time.sleep(5)
            else:
                raise RuntimeError('Judge startup timed out')
            status('judging', target_papers=len(selected))
            command = [sys.executable, str(ROOT / 'evaluate.py'), '--with-kus', '--context-limit', '32768']
            for identifier in args.skip_id:
                command += ['--skip-id', identifier]
            subprocess.run(command, check=True)
            incomplete = len(selected) != len(papers)
            subprocess.run([sys.executable, str(ROOT / 'report.py')] + (['--partial'] if incomplete else []), check=True)
            subprocess.run([sys.executable, str(ROOT / 'validate.py')] + (['--partial'] if incomplete else []), check=True)
            status('partial_report_ready' if incomplete else 'complete', papers=len(selected),
                   report='report.partial.html' if incomplete else 'report.html', excluded_ids=args.skip_id)
        finally:
            judge.terminate()
            try:
                judge.wait(timeout=45)
            except subprocess.TimeoutExpired:
                status('judge_shutdown_pending', judge_pid=judge.pid)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
