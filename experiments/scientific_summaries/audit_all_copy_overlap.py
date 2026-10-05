"""Audit preserved generated scientific summaries and returned intermediate attempts.

Run bulk analysis as a Slurm CPU step. Never read MCQs to select outputs.
"""
import argparse
from collections import defaultdict
import concurrent.futures
import csv
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
from ngram_overlap import SourceIndex,audit_summary,FIELDS,VERSION
from copy_overlap_eval import compact,markdown

BASE=Path('/e/fscratch/reformo/schuhmann1')
PINNED=BASE/'scientific-ornith-dflash-eval-97/inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a/experiments/scientific_summaries/data/testset.json'


def load(path):return json.loads(Path(path).read_text())


def parsed_summary(value):
    if isinstance(value,dict) and len(set(value)&set(FIELDS))>=3:return value
    if isinstance(value,str) and value.strip():
        text=value.strip()
        if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
        try:obj=json.loads(text)
        except ValueError:
            try:obj=json.loads(text[text.index('{'):text.rindex('}')+1])
            except (ValueError,TypeError):return text
        if isinstance(obj,dict) and len(set(obj)&set(FIELDS))>=3:return obj
    return None


def discover():
    tasks=defaultdict(list);roots=[]
    def add(group,path,docid,kind):tasks[docid].append((group,str(path),kind))
    teacher=BASE/'scientific-distillation-1000/outputs/documents';roots.append(str(teacher))
    for folder in sorted(teacher.iterdir()):
        if not folder.is_dir():continue
        if (folder/'final.json').exists():add('Qwen27 distillation / final corrected',folder/'final.json',folder.name,'saved')
        for p in sorted((folder/'calls').glob('*.json')):
            if p.name.startswith(('generation','structural_repair','semantic_repair')):
                phase='generation' if p.name.startswith('generation') else 'correction'
                add('Qwen27 distillation / returned '+phase+' attempts',p,folder.name,'call')
    for run in ['scientific-ornith-dflash-eval-97','scientific-ornith-35b-dflash-eval-97','scientific-gemma4-eval-97']:
        root=BASE/run/'outputs/cohorts';roots.append(str(root))
        for model in sorted(root.iterdir()):
            for folder in sorted((model/'documents').iterdir()):
                if not folder.is_dir():continue
                for p in sorted(folder.glob('*.json')):
                    if p.name=='document.json' or p.name.startswith('before-'):
                        add(run+'/'+model.name,p,folder.name,'historical')
                for p in sorted((folder/'calls').glob('*.json')):
                    add(run+'/'+model.name+' / returned attempts',p,folder.name,'call')
    native=BASE/'scientific-distillation-865-20261004/outputs/ornith/documents';roots.append(str(native))
    for folder in sorted(native.iterdir()):
        if not folder.is_dir():continue
        if (folder/'result.json').exists():add('Ornith9B distillation / final raw',folder/'result.json',folder.name,'saved')
        for p in sorted(folder.glob('attempt*.json')):
            if '.copy-overlap.' not in p.name:add('Ornith9B distillation / returned generation attempts',p,folder.name,'call')
    for run in ['scientific-distillation-865-20261004','scientific-gemma-r128-no-thinking-20261004','ornith-distillation-736-20261004','scientific-gemma-r128-fp8-20261005']:
        root=BASE/run/'outputs/evaluation/generation'
        if not root.exists():continue
        roots.append(str(root))
        for model in sorted(root.iterdir()):
            for folder in sorted(model.iterdir()):
                if not folder.is_dir():continue
                if (folder/'result.json').exists():add(run+'/'+model.name+' / saved output',folder/'result.json',folder.name,'saved')
                for p in sorted(folder.glob('attempt*.json')):
                    if '.copy-overlap.' not in p.name:add(run+'/'+model.name+' / returned attempts',p,folder.name,'call')
    return tasks,roots


def candidates(record,kind):
    if kind=='historical':
        for stage in ['raw','corrected']:
            value=record.get(stage)
            if isinstance(value,dict) and value.get('summary'):
                yield stage,value['summary'],value.get('judge_context'),'schema_object'
        return
    if kind=='saved':
        if record.get('summary'):yield '',record['summary'],record.get('judge_context',record.get('narrative')),'schema_object'
        return
    # Consume only model-returned content, never input messages or embedded drafts.
    response=record.get('response') or record.get('raw_response') or {}
    if not isinstance(response,dict):response={}
    choices=response.get('choices') or []
    content=choices[0].get('message',{}).get('content') if choices else record.get('content')
    if content is None and isinstance(record.get('record'),dict):content=record['record'].get('response')
    if content is None:
        value=record.get('result')
        if isinstance(value,dict):content=value.get('content')
    value=parsed_summary(content)
    if value is not None:
        # Old calls also contain QA/reviewer prose; those are not summary attempts.
        phase=str(record.get('phase',record.get('kind',(record.get('record') or {}).get('phase','')))).lower()
        if isinstance(value,str) and any(x in phase for x in ['assessment','review','judge','qa']):return
        yield '',value,None,'schema_object' if isinstance(value,dict) else 'unparsed_returned_text'


def task(item):
    docid,source,files=item;index=SourceIndex(source);found={};errors=[];no_output=0
    for group,path,kind in files:
        try:
            record=load(path)
            expected=record.get('source_sha256',record.get('fulltext_sha256'))
            if expected and expected!=index.sha256:raise ValueError('Stored source hash does not match the paper')
            outputs=list(candidates(record,kind))
            if not outputs:no_output+=1
            for stage,summary,context,representation in outputs:
                canonical=json.dumps(summary,sort_keys=True,ensure_ascii=False)
                fingerprint=hashlib.sha256(canonical.encode()).hexdigest()
                condition=group+(' / '+stage if stage else '')
                if representation=='unparsed_returned_text':condition+=' / unparsed text diagnostic'
                key=(condition,fingerprint)
                if key in found:found[key]['locations'].append(path);continue
                found[key]=dict(condition=condition,document_id=docid,source_sha256=index.sha256,
                    summary_sha256=fingerprint,representation=representation,locations=[path],
                    audit=audit_summary(source,summary,context,index=index))
        except Exception as e:errors.append(dict(document_id=docid,file=path,error=type(e).__name__+': '+str(e)))
    return list(found.values()),errors,no_output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args();start=time.monotonic()
    papers=load(PINNED)['papers']+load(BASE/'scientific-distillation-1000/inputs/papers.json')
    sources={}
    for p in papers:
        docid=p['document_id'];source=p['fulltext']
        if docid in sources:assert sources[docid]==source
        sources[docid]=source
        expected=p.get('fulltext_sha256',p.get('source_sha256'))
        if expected:assert hashlib.sha256(source.encode()).hexdigest()==expected
    tasks,roots=discover();missing=[docid for docid in tasks if docid not in sources]
    assert not missing, 'Missing full sources: '+str(missing[:20])
    rows=[];errors=[];non_summary=0
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    work=[(docid,sources[docid],files) for docid,files in sorted(tasks.items())]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(found,issues,excluded) in enumerate(pool.map(task,work),1):
            rows.extend(found);errors.extend(issues);non_summary+=excluded
            if i%100==0:print('Audited paper sources',i,'/',len(work),'summary views',len(rows),flush=True)
    groups=sorted({r['condition'] for r in rows})
    report=dict(audit_version=VERSION,complete=not errors,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-start,paper_sources=len(work),input_files=sum(len(v) for v in tasks.values()),
        unique_summary_views=len(rows),unique_source_summary_pairs=len({(r['source_sha256'],r['summary_sha256']) for r in rows}),
        duplicate_locations_within_condition=sum(len(r['locations'])-1 for r in rows),non_summary_or_empty_records=non_summary,
        source_roots=roots,errors=errors,allowed_consecutive_words=5,borderline_consecutive_words=6,
        violation_from_consecutive_words=7,conditions={g:compact([r for r in rows if r['condition']==g]) for g in groups},
        qa_or_training_records_filtered=False,reasoning_traces_not_summary_prose=True,
        scope='Preserved final, raw, corrected, recovery snapshots and returned generator/corrector attempts; capped throughput probes excluded; snapshots of running evaluations are identified by paths and scan time')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'per_summary.jsonl').open('w') as handle:
        for r in rows:handle.write(json.dumps(r,ensure_ascii=False)+'\n')
    flats=[]
    for r in rows:
        flat={k:r[k] for k in ['condition','document_id','source_sha256','summary_sha256','representation']}
        flat['locations']=len(r['locations'])
        for c in ['narrative','evidence','metadata']:
            value=r['audit']['categories'][c]
            flat.update({c+'_longest_match':value['longest_contiguous_match_words'],c+'_classification':value['classification'],
                c+'_coverage_6plus':value['covered_word_fraction_by_minimum_run']['6'],c+'_coverage_7plus':value['covered_word_fraction_by_minimum_run']['7']})
        flats.append(flat)
    with (out/'per_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flats[0]));writer.writeheader();writer.writerows(flats)
    (out/'RESULTS.md').write_text('# Scientific summaries: systematic source-copy audit\n\n'+
        f"{len(work)} paper sources; {len(rows)} deduplicated summary views across {len(groups)} conditions; "
        f"{report['unique_source_summary_pairs']} unique source/summary pairs. "
        'Intermediate returned attempts are separate from delivered final/raw/corrected views.\n\n'+markdown(report)+
        '\n## Scope and provenance\n\n'+report['scope']+'\n\n'+
        f"Read errors: {len(errors)}. JSONL evidence records include source/output hashes, paths, normalized offsets and longest matching excerpts.\n")
    print(json.dumps({k:report[k] for k in ['complete','paper_sources','input_files','unique_summary_views','unique_source_summary_pairs','elapsed_seconds']},indent=2),flush=True)
    if errors:raise RuntimeError('Audit has explicit read/parse errors; inspect report.json')


if __name__=='__main__':main()
