"""Queue summaries after the KU comparison, with an explicitly approved partial cohort.

Unresolved QA/cohort decisions wait without reserving GPUs. Only subprocesses created
by this program are terminated; unrelated new GPU processes are never stopped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

from evaluate import ROOT, load, questions_for, question_signature
from project_alexandria.io import write_json_atomic


def status(phase, **fields):
    payload = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write_json_atomic(str(ROOT / 'summary_phase_status.json'), payload)
    print(phase, fields, flush=True)


def prerequisites():
    papers = load(ROOT / 'data/papers.json')
    kus = load(ROOT / 'kus.json') if (ROOT / 'kus.json').exists() else {'documents': []}
    results = load(ROOT / 'results.json') if (ROOT / 'results.json').exists() else {'documents': []}
    ku_ids = {d['document_id']: d for d in kus['documents']}
    judged = {d['document_id']: d for d in results['documents']}
    missing = {'qa': [], 'kus': [], 'previous_ku_evaluation': []}
    for index, paper in enumerate(papers):
        identifier = paper['document_id']
        try:
            questions = questions_for(paper, index)
        except (ValueError, KeyError, json.JSONDecodeError):
            questions = None
        if not questions:
            missing['qa'].append(identifier)
        if (identifier not in ku_ids
                or ku_ids[identifier]['fulltext_sha256'] != paper['fulltext_sha256']):
            missing['kus'].append(identifier)
        previous = judged.get(identifier)
        if (not questions or not previous
                or question_signature(previous['questions']) != question_signature(questions)
                or not all({'no_context', 'original', 'summary', 'knowledge_units'} <= set(r['predictions'])
                           for r in previous['rows'])):
            missing['previous_ku_evaluation'].append(identifier)
    return len(papers), missing


def wait_for_free_gpus():
    for _ in range(120):
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(used) >= 2 and all(int(value.strip()) < 1024 for value in used[:2]):
            return
        time.sleep(5)
    raise RuntimeError('GPUs remain occupied; no unrecognized process was terminated.')


def ready(process, alias):
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(alias + ' startup failed; inspect summary_runtime.log')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8010/v1/models', timeout=2) as response:
                if any(model['id'] == alias for model in json.load(response)['data']):
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(alias + ' startup timed out')


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Owned server did not shut down; no forced GPU kill was attempted.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default=str(ROOT / 'summary-systemprompt+.txt'))
    parser.add_argument('--required-papers', type=int, default=100)
    parser.add_argument('--allow-partial-cohort', action='store_true',
                        help='Explicitly exclude papers missing the preceding QA/KU comparison.')
    args = parser.parse_args()
    previous_state = None
    while True:
        count, missing = prerequisites()
        state = (count, json.dumps(missing, sort_keys=True))
        if state != previous_state:
            status('awaiting_complete_cohort_and_ku_comparison', required_papers=args.required_papers,
                   corpus_papers=count, pending=missing)
            previous_state = state
        excluded = sorted(set(identifier for values in missing.values() for identifier in values))
        if count == args.required_papers and (not excluded or
                (args.allow_partial_cohort and count > len(excluded))):
            break
        time.sleep(30)
    selected_count = count - len(excluded)
    skip_arguments = [argument for identifier in excluded for argument in ('--skip-id', identifier)]
    status('starting_qwen_summary_phase', papers=selected_count, corpus_papers=count,
           excluded_ids=excluded, partial_cohort_authorized=args.allow_partial_cohort)
    # The preceding comparison is complete before any summary generation starts.
    write_json_atomic(str(ROOT / 'pre_qwen_summary_results.json'), load(ROOT / 'results.json'))
    wait_for_free_gpus()
    environment = dict(os.environ)
    with (ROOT / 'summary_runtime.log').open('a', encoding='utf-8') as log:
        extractor = subprocess.Popen(['bash', str(ROOT / 'serve_extractor.sh')],
                                     env=environment, stdout=log, stderr=log)
        try:
            ready(extractor, 'qwen38')
            status('generating_qwen_summaries', papers=selected_count, excluded_ids=excluded,
                   prompt=args.prompt)
            subprocess.run([sys.executable, str(ROOT / 'summarize.py'), '--prompt', args.prompt,
                            '--max-new-documents', '2', '--concurrency', '8'] + skip_arguments, check=True)
            status('summary_smoke_test_passed', papers=2)
            subprocess.run([sys.executable, str(ROOT / 'summarize.py'), '--prompt', args.prompt,
                            '--concurrency', '8'] + skip_arguments, check=True)
        finally:
            stop(extractor)
        wait_for_free_gpus()
        environment.update(ALEXANDRIA_JUDGE_GPU='0', ALEXANDRIA_JUDGE_MAX_LEN='32768',
                           ALEXANDRIA_JUDGE_GPU_UTIL='0.90')
        judge = subprocess.Popen(['bash', str(ROOT / 'serve_judge.sh')],
                                 env=environment, stdout=log, stderr=log)
        try:
            ready(judge, 'qwen25')
            status('student_evaluation_of_qwen_summaries', papers=selected_count, excluded_ids=excluded)
            subprocess.run([sys.executable, str(ROOT / 'evaluate.py'), '--with-kus',
                            '--with-qwen-summaries', '--context-limit', '32768'] + skip_arguments, check=True)
            report_arguments = ['--partial'] if excluded else []
            subprocess.run([sys.executable, str(ROOT / 'report.py')] + report_arguments, check=True)
            subprocess.run([sys.executable, str(ROOT / 'validate.py')] + report_arguments, check=True)
            status('partial_cohort_complete' if excluded else 'complete', papers=selected_count,
                   excluded_ids=excluded, report='report.partial.html' if excluded else 'report.html')
        finally:
            stop(judge)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
