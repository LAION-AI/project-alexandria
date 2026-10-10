# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
"""QA-blind KU back-translation, numeric/formula guard, and fixed QA comparison."""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, Backend, MODELS, digest, jsonl, load, read_jsonl, sources, write
from critical_values import check
from passages import sentence_spans
from ngram_overlap import SourceIndex, audit_summary
from project_alexandria.schema import DocumentResult
from project_alexandria.experiments.reproduce import knowledge_unit_context

PRIOR = ROOT.parent/'scientific-ku-distillation-20261009'
FOLLOW = ROOT.parent/'scientific-gemma-ku500-eval20-20261010'
HOME = ROOT
METRICS = ROOT/'inputs/reference_metrics.json'
KU_CONDITIONS = [f'{model}_ku{words}' for words in [500, 1000] for model in ['qwen', 'gemma4_base', 'gemma4_r128', 'gemma12_base', 'gemma12_r128']]
PROTECTED = re.compile(r'author|title|doi|citation|reference|identifier|equation|formula|expression|symbol|unit|variable', re.I)

def get(value, path):
    for part in path: value = value[part]
    return value

def set_value(value, path, replacement):
    for part in path[:-1]: value = value[part]
    value[path[-1]] = replacement

def descriptive_fields(result):
    def attributes(value, path):
        if isinstance(value, str):
            if len(value.split()) >= 6: yield path, value
        elif isinstance(value, dict):
            for key, child in value.items():
                if not PROTECTED.search(str(key)): yield from attributes(child, path+[key])
        elif isinstance(value, list):
            for i, child in enumerate(value): yield from attributes(child, path+[i])
    for i, unit in enumerate(result['knowledge_units']):
        yield ['knowledge_units', i, 'context_summary'], unit['context_summary']
        for j, entity in enumerate(unit['entities']):
            path = ['knowledge_units', i, 'entities', j]
            yield from attributes(entity['attributes'], path+['attributes'])
            for k, rel in enumerate(entity['relationships']):
                yield from attributes(rel['attributes'], path+['relationships', k, 'attributes'])

def audit_fields(result):
    # Exactly the primary field grouping of the preceding KU benchmark.
    def strings(value):
        if isinstance(value, str): return [value]
        if isinstance(value, dict): return [s for k, v in value.items() for s in [str(k)]+strings(v)]
        if isinstance(value, list): return [s for v in value for s in strings(v)]
        return [] if value is None else [str(value)]
    prose = []
    for u in result['knowledge_units']:
        prose.append(u['context_summary'])
        for e in u['entities']:
            prose += [e['name'], e['entity_type']]+e.get('aliases', [])+strings(e['attributes'])
            for r in e['relationships']: prose += [r['predicate'], r['target']]+strings(r['attributes'])
    return {'narrative': prose, 'title': result['title'], 'author': result.get('author', '')}

def selected_windows(result, source):
    index = SourceIndex(source)
    rows = []
    for path, text in descriptive_fields(result):
        if index.fragment(text, str(path), 'narrative')['longest_contiguous_match_words'] < 6: continue
        spans = sentence_spans(text, max_words=40); flagged = set()
        for i, (a, b) in enumerate(spans):
            own = index.fragment(text[a:b], str(path), 'narrative')['longest_contiguous_match_words']
            if own >= 6: flagged.add(i)
            if i and index.fragment(text[spans[i-1][0]:b], str(path), 'narrative')['longest_contiguous_match_words'] >= 6 and own < 6:
                flagged.update([i-1, i])
        expanded = sorted(flagged | {i-1 for i in flagged if i})
        groups = []
        for i in expanded:
            if groups and len(groups[-1]) < 2 and i == groups[-1][-1]+1: groups[-1].append(i)
            else: groups.append([i])
        for group in groups:
            a, b = spans[group[0]][0], spans[group[-1]][1]
            rows.append(dict(path=path, fragment=text, start=a, end=b, text=text[a:b]))
    return rows

def prepare():
    papers = sources(); prior_scores = load(METRICS)
    assert digest((ROOT/'inputs/new20_questions.json').read_bytes()) == load(ROOT/'inputs/new20_questions_frozen.json')['sha256']
    windows = []
    for c in prior_scores['new20']['conditions']:
        if c in {'no_context', 'original'}: continue
        for p in papers:
            candidate = FOLLOW/'outputs/generation/new20'/c/(p['document_id']+'.json')
            if not candidate.exists(): candidate = PRIOR/'outputs/generation/new20'/c/(p['document_id']+'.json')
            row = load(candidate); assert row['source_sha256'] == p['fulltext_sha256']
            target = ROOT/'outputs/generation'/c/(p['document_id']+'.json')
            if target.exists(): assert load(target) == row
            else: write(target, row)
            if c in KU_CONDITIONS:
                for w in selected_windows(row['result'], p['fulltext']):
                    ident = digest(json.dumps([c, p['document_id'], w['path'], w['start'], w['end']]))[:24]
                    windows.append(dict(w, window_id=ident, condition=c, document_id=p['document_id']))
    jsonl(ROOT/'inputs/windows.jsonl', windows)
    chosen_ids = [papers[i]['document_id'] for i in [0, 4, 9, 12, 15]]
    protocol = dict(papers=20, questions_per_condition=200, primary_conditions=28, primary_qa_slots=5600,
                    prior_raw_conditions=prior_scores['new20']['conditions'], ku_conditions=KU_CONDITIONS,
                    translation_model='google/translategemma-4b-it', translation_revision='10042cb0e6e7fdce748996a71dc3dc432a4e0c89',
                    pivot='de', maximum_roundtrips=2, greedy_temperature=0, translation_batch=512,
                    targeted_fields='context_summary and descriptive attribute string values only',
                    protected_fields='entity names, aliases, IDs, relation predicates/targets, bibliographic and mathematical fields, JSON keys',
                    numeric_formula_guard_version='1.1', bidirectional_nli_minimum_entailment=0.9,
                    source_copy_window_maximum_words=5, selection_minimum_copied_words=6,
                    no_qa_feedback_to_generation_or_repair=True, all_conditions_freshly_judged=True,
                    source_cohort_sha256=digest((ROOT/'inputs/new20_sources.json').read_bytes()),
                    frozen_questions_sha256=digest((ROOT/'inputs/new20_questions.json').read_bytes()),
                    original_metrics_sha256=digest(METRICS.read_bytes()), window_count=len(windows),
                    demo_document_ids=chosen_ids, demo_selection='Fixed cohort indices 0,4,9,12,15 before this repair evaluation; no QA-based selection',
                    context='Previous ten KUs for naming; no next-source lookahead; current context_summary follows existing schema',
                    failures='Preserved; any original failed generation keeps ten wrong slots, including after paraphrasing')
    write(ROOT/'inputs/protocol.json', protocol)
    print('PREPARED', len(windows), 'translation windows;', 28, 'QA conditions', flush=True)

def translate_worker():
    from engines import TranslateGemma
    t = time.monotonic(); engine = TranslateGemma()
    write(ROOT/'outputs/translation_setup.json', dict(seconds=time.monotonic()-t))
    pending = read_jsonl(ROOT/'inputs/windows.jsonl')
    for round_index in [0, 1]:
        if round_index:
            while not (ROOT/'outputs/quality/round0.complete.json').exists(): time.sleep(2)
            status = read_jsonl(ROOT/'outputs/quality/round0.jsonl')
            prior = {r['window_id']: r for r in read_jsonl(ROOT/'outputs/translation/round0.jsonl')}
            pending = [dict(w, translation_input=prior[w['window_id']]['text']) for w, q in zip(pending, status)
                       if not q['accepted'] and prior[w['window_id']]['status'] == 'translated']
        tick = time.monotonic()
        forward, fperf = engine.translate([w.get('translation_input', w['text']) for w in pending], 'en_de', 512)
        indices = [i for i, row in enumerate(forward) if row['status'] == 'translated']
        back, bperf = engine.translate([forward[i]['text'] for i in indices], 'de_en', 512)
        returned = dict(zip(indices, back)); rows = []
        for i, w in enumerate(pending):
            b = returned.get(i, dict(text=w['text'], status='forward_failed', output_tokens=0))
            rows.append(dict(b, window_id=w['window_id'], german=forward[i]['text'],
                             forward_status=forward[i]['status'], round_index=round_index))
        jsonl(ROOT/f'outputs/translation/round{round_index}.jsonl', rows)
        write(ROOT/f'outputs/translation/round{round_index}.complete.json', dict(complete=True, windows=len(rows),
              seconds=time.monotonic()-tick, forward=fperf, backward=bperf, active_gpus=1))
        print('TRANSLATION_ROUND_COMPLETE', round_index, len(rows), flush=True)

def quality_worker():
    from nli import NLI, calibration
    from copy_overlap_eval import compact
    t = time.monotonic(); nli = NLI(); calibration(nli)
    windows = read_jsonl(ROOT/'inputs/windows.jsonl'); original_windows = {w['window_id']: w for w in windows}
    papers = {p['document_id']: p for p in sources()}
    indices = {k: SourceIndex(p['fulltext']) for k, p in papers.items()}
    accepted = {}; details = []
    for round_index in [0, 1]:
        while not (ROOT/f'outputs/translation/round{round_index}.complete.json').exists(): time.sleep(2)
        candidates = read_jsonl(ROOT/f'outputs/translation/round{round_index}.jsonl')
        ready = [r for r in candidates if r['status'] == 'translated']
        pairs = [pair for r in ready for pair in [(original_windows[r['window_id']]['text'], r['text']),
                                                (r['text'], original_windows[r['window_id']]['text'])]]
        scores, nperf = nli.score(pairs)
        semantic = {r['window_id']: scores[2*i:2*i+2] for i, r in enumerate(ready)}
        rows = []
        for r in candidates:
            w = original_windows[r['window_id']]; after = r['text']; reasons = []
            critical = check(w['text'], after)
            left, right = semantic.get(r['window_id'], [None, None])
            overlap = indices[w['document_id']].fragment(after, str(w['path']), 'narrative')
            if r['status'] != 'translated': reasons.append('translation_failed')
            if not critical['passed']: reasons.append('number_unit_formula_or_operator_change')
            if not left or not right: reasons.append('nli_unassessable')
            elif min(left['entailment'], right['entailment']) < .9: reasons.append('nli_below_fixed_0.9')
            if overlap['longest_contiguous_match_words'] > 5: reasons.append('source_copy_exceeds_five')
            q = dict(window_id=w['window_id'], document_id=w['document_id'], condition=w['condition'],
                     path=w['path'], original=w['text'], candidate=after, german=r['german'],
                     round_index=round_index, accepted=not reasons, rejection_reasons=reasons,
                     critical_values=critical, nli_forward=left, nli_backward=right,
                     longest_copy_words=overlap['longest_contiguous_match_words'])
            rows.append(q); details.append(q)
            if q['accepted']: accepted[w['window_id']] = after
        jsonl(ROOT/f'outputs/quality/round{round_index}.jsonl', rows)
        write(ROOT/f'outputs/quality/round{round_index}.complete.json', dict(complete=True, windows=len(rows),
                                                                         accepted=sum(r['accepted'] for r in rows), nli=nperf))
        print('QUALITY_ROUND_COMPLETE', round_index, sum(r['accepted'] for r in rows), flush=True)
    grouped = defaultdict(list)
    for w in windows:
        if w['window_id'] in accepted: grouped[(w['condition'], w['document_id'])].append(w)
    changed = rolled_back = inserted = 0; audits = []; global_checks = []
    for condition in KU_CONDITIONS:
        for paper in sources():
            docid = paper['document_id']; row = load(ROOT/'outputs/generation'/condition/(docid+'.json'))
            original_result = row['result']; result = copy.deepcopy(original_result)
            groups = defaultdict(list)
            for w in grouped[(condition, docid)]: groups[tuple(w['path'])].append(w)
            count = 0
            for path, items in groups.items():
                value = get(result, path)
                for w in sorted(items, key=lambda x: x['start'], reverse=True):
                    assert w['fragment'] == get(original_result, path)
                    value = value[:w['start']]+accepted[w['window_id']].rstrip()+' '+value[w['end']:]; count += 1
                set_value(result, path, value)
            parsed = DocumentResult.from_dict(result)
            context = knowledge_unit_context(parsed)
            critical = check(row['judge_context'], context)
            rollback = not critical['passed']
            if rollback: result = original_result; context = row['judge_context']; count = 0; rolled_back += 1
            # Replacing only declared leaves leaves graph structure, names and metadata intact.
            before, after = copy.deepcopy(original_result), copy.deepcopy(result)
            for path, _ in descriptive_fields(original_result):
                set_value(before, path, '<descriptive-value>'); set_value(after, path, '<descriptive-value>')
            assert before == after
            final_check = check(row['judge_context'], context); assert final_check['passed']
            value = dict(row, result=result, judge_context=context, repair=dict(accepted_windows=count,
                         rolled_back=rollback, original_context_sha256=digest(row['judge_context']),
                         final_context_sha256=digest(context), critical_values=final_check))
            write(ROOT/'outputs/generation'/(condition+'_bt')/(docid+'.json'), value)
            changed += context != row['judge_context']; inserted += count
            global_checks.append(dict(condition=condition, document_id=docid, critical_values=final_check,
                                      accepted_windows=count, changed=context != row['judge_context'], rolled_back=rollback))
            for label, obj in [(condition, original_result), (condition+'_bt', result)]:
                audit = audit_summary(paper['fulltext'], audit_fields(obj), index=indices[docid]) if obj['knowledge_units'] else None
                audits.append(dict(condition=label, document_id=docid, source_sha256=paper['fulltext_sha256'], status=row['status'], audit=audit))
    jsonl(ROOT/'outputs/window_quality.jsonl', details); jsonl(ROOT/'outputs/document_quality.jsonl', global_checks)
    jsonl(ROOT/'outputs/copy_overlap_details.jsonl', audits)
    report = dict(audit_version=__import__('ngram_overlap').VERSION, complete=True, sources_untruncated=True,
                  allowed_consecutive_words=5, borderline=6, violation_from=7, qa_filtering=False,
                  conditions={c: compact([a for a in audits if a['condition'] == c]) for c in KU_CONDITIONS+[c+'_bt' for c in KU_CONDITIONS]})
    write(ROOT/'outputs/copy_overlap.json', report)
    write(ROOT/'outputs/quality_complete.json', dict(complete=True, windows=len(windows), attempts=len(details),
          changed_documents=changed, inserted_windows=inserted, rolled_back_documents=rolled_back,
          guarded_global_critical_failures=0, rejection_reasons=dict(Counter(x for d in details for x in d['rejection_reasons'])),
          elapsed_seconds=time.monotonic()-t, numeric_guard_is_conservative=True))
    print('QUALITY_COMPLETE', changed, 'changed documents', inserted, 'inserted windows', flush=True)

def qa_worker(lane):
    from qa_score import score
    protocol = load(ROOT/'inputs/protocol.json')
    conditions = protocol['prior_raw_conditions']+[c+'_bt' for c in KU_CONDITIONS]
    frozen = load(ROOT/'inputs/new20_questions.json'); qs = {p['document_id']: p['questions'] for p in frozen['papers']}
    assert digest((ROOT/'inputs/new20_questions.json').read_bytes()) == protocol['frozen_questions_sha256']
    slots = [(p, c) for c in conditions[lane::2] for p in sources()]
    server = None
    try:
        port = 21620+lane
        argv = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', MODELS['judge']['path'],
                '--served-model-name', 'base', '--host', '127.0.0.1', '--port', str(port), '--max-model-len', '32768',
                '--max-num-seqs', '4', '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '.90',
                '--generation-config', 'vllm', '--enable-prefix-caching', '--enable-chunked-prefill']
        log = (ROOT/f'logs/qa-server-{lane}.log').open('w')
        server = subprocess.Popen(argv, stdout=log, stderr=log, start_new_session=True)
        import urllib.request
        endpoint = f'http://127.0.0.1:{port}'
        for _ in range(300):
            if server.poll() is not None: raise RuntimeError('QA server startup failed')
            try:
                with urllib.request.urlopen(endpoint+'/health', timeout=2) as response:
                    if response.status == 200: break
            except OSError: time.sleep(2)
        else: raise RuntimeError('QA server startup timeout')
        backend = Backend(endpoint, 'base', 'qa', temperature=.5, json_output=False); pending = list(slots)
        while pending:
            progress = False
            for paper, condition in pending[:]:
                if condition not in {'original', 'no_context'} and not (ROOT/'outputs/generation'/condition/(paper['document_id']+'.json')).exists(): continue
                score(paper, qs[paper['document_id']], 'new20', condition, backend)
                pending.remove((paper, condition)); progress = True
                write(ROOT/f'outputs/qa_progress_{lane}.json', dict(completed=len(slots)-len(pending), total=len(slots)))
            if not progress: time.sleep(2)
        write(ROOT/f'outputs/qa_complete_{lane}.json', dict(complete=True, paper_condition_slots=len(slots)))
    finally:
        if server is not None:
            try: os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: server.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(server.pid, signal.SIGKILL); server.wait()

def run():
    assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    processes = []; start = time.monotonic()
    try:
        for gpu, role, extra in [(0, 'translation', []), (1, 'quality', []), (2, 'qa', ['--lane', '0']), (3, 'qa', ['--lane', '1'])]:
            cache = ROOT/'cache'/role/str(gpu); cache.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XDG_CACHE_HOME=str(cache), VLLM_CACHE_ROOT=str(cache/'vllm'),
                       TRITON_CACHE_DIR=str(cache/'triton'), TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),
                       FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'))
            log = (ROOT/f'logs/{role}-{gpu}.log').open('w')
            proc = subprocess.Popen([sys.executable, __file__, '--role', role]+extra, env=env, stdout=log, stderr=log, start_new_session=True)
            processes.append((role, proc, log))
        while any(p.poll() is None for _, p, _ in processes):
            failed = [(r, p.returncode) for r, p, _ in processes if p.poll() not in [None, 0]]
            write(ROOT/'outputs/worker_status.json', dict(job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start,
                  workers={r+str(i): p.poll() for i, (r, p, _) in enumerate(processes)}))
            if failed: raise RuntimeError('Owned worker failure: '+str(failed))
            time.sleep(3)
        write(ROOT/'outputs/compute_complete.json', dict(complete=True, job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start))
    finally:
        for role, proc, log in processes:
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--role', choices=['prepare', 'run', 'translation', 'quality', 'qa'], required=True)
    parser.add_argument('--lane', type=int, default=0); args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'translation': translate_worker, 'quality': quality_worker, 'qa': lambda: qa_worker(args.lane)}[args.role]()
