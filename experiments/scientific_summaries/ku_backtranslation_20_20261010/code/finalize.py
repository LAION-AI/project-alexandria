# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
"""Freeze factual QA, fidelity and copying results for the completed KU repair run."""
import csv
import datetime
import gzip
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from common import ROOT, digest, jsonl, load, sources, write
from ku_pipeline import KU_CONDITIONS, descriptive_fields, audit_fields

def author_names(source):
    authors = source.get('source_authors') or ''
    if isinstance(authors, str):
        try: authors = json.loads(authors)
        except (ValueError, TypeError): return [authors] if authors else []
    if not isinstance(authors, list): return []
    output = []
    for author in authors:
        if isinstance(author, str): output.append(author)
        elif isinstance(author, dict):
            middle = author.get('middle', [])
            if not isinstance(middle, list): middle = [middle]
            name = ' '.join(str(x) for x in [author.get('first', '')]+middle+[author.get('last', ''), author.get('suffix', '')] if x)
            if name: output.append(name)
    return output

def public_result(value):
    # The output object contains factual KUs and one-way source references,
    # never full source prose, emails, source archive paths or QA prompts.
    return {k: value[k] for k in ['document_id', 'source_sha256', 'kind', 'status', 'result', 'chunks_expected', 'chunks_generated', 'failed_chunks'] if k in value} | ({'repair': value['repair']} if 'repair' in value else {})

def main():
    import numpy as np
    assert load(ROOT/'outputs/compute_complete.json')['complete']
    protocol = load(ROOT/'inputs/protocol.json'); papers = sources()
    conditions = protocol['prior_raw_conditions']+[c+'_bt' for c in KU_CONDITIONS]
    counts = np.zeros((20, len(conditions)), int); invalid = {c: 0 for c in conditions}; records = []
    for i, p in enumerate(papers):
        for j, condition in enumerate(conditions):
            result = load(ROOT/'outputs/qa/new20'/condition/(p['document_id']+'.json'))
            assert result['source_sha256'] == p['fulltext_sha256'] and len(result['rows']) == 10
            counts[i, j] = sum(r['gold'] == r['prediction'] for r in result['rows'])
            invalid[condition] += sum(r['prediction'] is None for r in result['rows'])
            record = {k: result[k] for k in ['document_id', 'source_sha256', 'condition', 'context_sha256']}
            record['rows'] = [{k: r.get(k) for k in ['question_index', 'gold', 'prediction', 'generation_failed']} for r in result['rows']]
            records.append(record)
    rng = np.random.default_rng(20261010)
    bootstrap = counts[rng.integers(0, 20, (10000, 20))].mean(axis=1)/10
    scores = {c: dict(correct=int(counts[:,j].sum()), total=200, accuracy=float(counts[:,j].sum()/200),
                      invalid=invalid[c], confidence_interval_95=[float(x) for x in np.quantile(bootstrap[:,j], [.025,.975])]) for j,c in enumerate(conditions)}
    # Summary references are unchanged cached objects. Retain their already
    # completed full-source audits and verify source/content identity here.
    from copy_overlap_eval import compact
    audit_path=ROOT/'outputs/copy_overlap_details.jsonl'
    audits=[json.loads(x) for x in audit_path.read_text().splitlines()]
    summary_conditions=[c for c in protocol['prior_raw_conditions'] if c not in protocol['ku_conditions'] and c not in {'original','no_context'}]
    if not any(a['condition'] in summary_conditions for a in audits):
        reference=ROOT.parent/'scientific-gemma-ku500-eval20-20261010/inputs/reference_copy_overlap_details.jsonl'
        known={p['document_id']:p for p in papers}
        for line in reference.read_text().splitlines():
            a=json.loads(line)
            if a['cohort']!='new20' or a['condition'] not in summary_conditions:continue
            assert a['source_sha256']==known[a['document_id']]['fulltext_sha256']
            original=load(ROOT.parent/'scientific-ku-distillation-20261009/outputs/generation/new20'/a['condition']/(a['document_id']+'.json'))
            cached=load(ROOT/'outputs/generation'/a['condition']/(a['document_id']+'.json'))
            assert all(original.get(k)==cached.get(k) for k in ['summary','judge_context','status','source_sha256'])
            a.update(reused_unchanged_full_source_audit=True);audits.append(a)
        jsonl(audit_path,audits)
    assert len(audits)==520
    copy_report=load(ROOT/'outputs/copy_overlap.json')
    for c in summary_conditions:copy_report['conditions'][c]=compact([a for a in audits if a['condition']==c])
    copy_report['cached_summary_audits']=dict(conditions=summary_conditions,slots=120,source_and_cached_content_verified=True)
    write(ROOT/'outputs/copy_overlap.json',copy_report)
    paired = {}
    for c in KU_CONDITIONS:
        a, b = conditions.index(c+'_bt'), conditions.index(c)
        paired[c] = dict(point=scores[c+'_bt']['accuracy']-scores[c]['accuracy'],
                         confidence_interval_95=[float(x) for x in np.quantile(bootstrap[:,a]-bootstrap[:,b], [.025,.975])])
    quality = load(ROOT/'outputs/quality_complete.json')
    attempts = [json.loads(x) for x in (ROOT/'outputs/window_quality.jsonl').read_text().splitlines()]
    attempted = [x for x in attempts if x['nli_forward'] is not None]
    quality['raw_attempts_with_critical_flags'] = sum(not x['critical_values']['passed'] for x in attempted)
    quality['numeric_attempts'] = sum(x['critical_values']['numeric_content_present'] for x in attempted)
    quality['formula_attempts'] = sum(x['critical_values']['formula_content_present'] for x in attempted)
    quality['raw_numeric_signature_changes'] = sum(not x['critical_values']['checks']['numbers'] for x in attempted)
    quality['raw_formula_signature_changes'] = sum(not x['critical_values']['checks']['formulas'] for x in attempted)
    metadata = {}; all_public = []
    for c in KU_CONDITIONS:
        rows = [load(ROOT/'outputs/generation'/(c+'_bt')/(p['document_id']+'.json')) for p in papers]
        metadata[c] = dict(document_slots=20,
                           exact_document_title_matches=sum(r['result']['title'] == p['source'].get('source_title', '') for p,r in zip(papers,rows)),
                           nonempty_document_authors=sum(bool(r['result'].get('author')) for r in rows),
                           author_attributes_present=sum(any(any('author' in str(k).lower() for k in e['attributes']) for u in r['result']['knowledge_units'] for e in u['entities']) for r in rows),
                           explicit_style_attributes_present=sum(any(any('style' in str(k).lower() for k in e['attributes']) for u in r['result']['knowledge_units'] for e in u['entities']) for r in rows),
                           context_summary_present_units=sum(bool(u['context_summary']) for r in rows for u in r['result']['knowledge_units']),
                           units=sum(len(r['result']['knowledge_units']) for r in rows),
                           complete_papers=sum(r['status']=='generated' and r['chunks_generated']==r['chunks_expected'] for r in rows),
                           sentence_minhash_present_units=sum(bool(u['source'].get('sentence_minhash')) for r in rows for u in r['result']['knowledge_units']))
        for p,r in zip(papers,rows):
            if p['document_id'] not in protocol['demo_document_ids']: continue
            if c.split('_ku')[0] not in {'gemma12_r128','gemma4_r128','qwen'}: continue
            public = public_result(r); public.update(condition=c+'_bt', source_metadata=dict(
                title=p['source'].get('source_title',''), authors=author_names(p['source']),
                doi=p['source'].get('source_doi') or p['source'].get('oa_doi') or '',
                year=p['source'].get('source_year') or '', openalex_id=p['source'].get('openalex_id') or '',
                origin='Existing disjoint 20-paper benchmark cohort',
                metadata_added_for_attribution_only=True, metadata_not_used_to_change_QA_context=True))
            all_public.append(public)
    translation = [load(ROOT/f'outputs/translation/round{i}.complete.json') for i in [0,1]]
    seconds = sum(r['seconds'] for r in translation)
    allocation = {}; job = load(ROOT/'outputs/compute_complete.json')['job_id']
    output = subprocess.check_output(['sacct','-j',job,'--parsable2','--noheader','--format=JobIDRaw,State,ElapsedRaw,AllocTRES'], text=True)
    for line in output.splitlines():
        fields = line.split('|')
        if fields[0] == job:
            allocation = dict(job_id=job, state=fields[1], allocation_seconds=int(fields[2]), allocated_gpus=4,
                              reserved_gpu_hours=int(fields[2])*4/3600, alloc_tres=fields[3])
    metrics = dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   papers=20, questions_per_condition=200, primary_qa_slots=5600, scores=scores,
                   paired_backtranslation_minus_raw=paired, quality=quality, metadata=metadata,
                   translation=dict(rounds=translation, active_translation_seconds=seconds,
                                    active_translation_gpu_hours=seconds/3600, documents=200,
                                    document_versions_per_active_gpu_hour=200*3600/seconds,
                                    setup_semantic_checks_QA_and_idle_excluded=True), allocation=allocation,
                   protocol=protocol, no_qa_based_output_filtering=True)
    write(ROOT/'outputs/metrics.json', metrics)
    jsonl(ROOT/'outputs/public_demo_examples.jsonl', all_public)
    jsonl(ROOT/'outputs/qa_predictions.jsonl', records)
    with (ROOT/'outputs/copy_overlap.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['condition','document_id','status','longest_narrative_match','coverage_six_plus','longest_metadata_match'])
        writer.writeheader()
        for row in [json.loads(x) for x in (ROOT/'outputs/copy_overlap_details.jsonl').read_text().splitlines()]:
            categories = row['audit']['categories'] if row['audit'] else None
            writer.writerow(dict(condition=row['condition'],document_id=row['document_id'],status=row['status'],
                longest_narrative_match=categories['narrative']['longest_contiguous_match_words'] if categories else '',
                coverage_six_plus=categories['narrative']['covered_word_fraction_by_minimum_run']['6'] if categories else '',
                longest_metadata_match=categories['metadata']['longest_contiguous_match_words'] if categories else ''))
    print('FINALIZED',len(scores),'conditions;', len(all_public),'complete-or-explicitly-flagged example versions', flush=True)

if __name__ == '__main__': main()
