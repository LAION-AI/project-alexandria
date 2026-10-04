"""Matched Gemma baseline/r64/r128 generation and frozen 97-paper, 970-MCQ QA."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

from common import ROOT, SOURCE, load, write, digest, jsonl
from inference import Server, generate_paper

sys.path.insert(0,str(SOURCE/'code'))
from run import summary_system

PINNED=Path('/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a')
VENDOR=PINNED/'experiments/scientific_summaries'
sys.path.insert(0,str(PINNED/'src'))
sys.path.insert(0,str(VENDOR))
import evaluate
from project_alexandria.backends import OpenAICompatibleBackend

LABELS={'gemma12_base':'Gemma 4 12B IT / no LoRA, thinking enabled',
        'gemma12_r64':'Gemma 4 12B IT / rank 64, one epoch',
        'gemma12_r128':'Gemma 4 12B IT / rank 128, one epoch'}


def sha_text(text):return hashlib.sha256(text.encode()).hexdigest()


def bootstrap(values):
    rng=random.Random(250219413)
    draws=sorted(sum(values[rng.randrange(len(values))] for _ in values)/(10*len(values)) for _ in range(10000))
    return [draws[249],draws[9749]]


def cohort(server,label,papers):
    started=time.monotonic()
    prompts=load(ROOT/'inputs/reference_prompts.json')
    system=summary_system(prompts)
    def task(paper):
        source=dict(document_id=paper['document_id'],fulltext=paper['fulltext'],
                    source_sha256=paper['fulltext_sha256'])
        return generate_paper(server.endpoint,source,system,
            prompts['summary-user'].replace('{paper_text}',paper['fulltext']),
            ROOT/'outputs/evaluation/generation'/label/paper['document_id'],
            model='summary-model' if label=='gemma12_base' else 'trained',thinking=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(task,papers))
    jsonl(ROOT/'outputs/evaluation'/(label+'-summaries.jsonl'),rows)
    elapsed=time.monotonic()-started
    attempts=[]
    for paper in papers:
        for path in (ROOT/'outputs/evaluation/generation'/label/paper['document_id']).glob('attempt*.json'):
            attempts.append(load(path))
    complete_tokens=sum(a['response'].get('usage',{}).get('completion_tokens',0) for a in attempts)
    prompt_tokens=sum(a['response'].get('usage',{}).get('prompt_tokens',0) for a in attempts)
    write(ROOT/'outputs/evaluation'/(label+'-generation-performance.json'),dict(
        papers=len(rows),elapsed_seconds=elapsed,active_gpus=1,active_generation_gpu_hours=elapsed/3600,
        completion_tokens_all_attempts=complete_tokens,prompt_tokens_all_attempts=prompt_tokens,
        output_tokens_per_second=complete_tokens/elapsed,api_calls=len(attempts),
        mean_native_narrative_tokens=statistics.mean(r.get('native_narrative_tokens',0) for r in rows),
        restarts_and_server_setup_excluded=True,requests_resumed_from_disk=any(r['elapsed_seconds']>elapsed for r in rows)))
    return {row['document_id']:row for row in rows}


def validate(results,papers,contexts):
    totals={label:dict(correct=0,total=0,invalid=0) for label in LABELS}
    by_id={p['document_id']:p for p in papers}
    for document in results['documents']:
        paper=by_id[document['document_id']]
        assert document['questions']==paper['questions']
        assert document['fulltext_sha256']==paper['fulltext_sha256']
        for label,condition in document['conditions'].items():
            context=contexts[label][paper['document_id']]['judge_context']
            assert condition['context_sha256']==sha_text(context) and len(condition['rows'])==10
            for row,question in zip(condition['rows'],paper['questions']):
                assert row['gold']==question['answer'] and row['question_index']==question['question_index']
                if not condition.get('generation_failed'):
                    assert row['responses']['qwen_summary']['prompt_sha256']==sha_text(
                        evaluate.historical_answer_prompt(question['formatted_question'],context))
                prediction=next(iter(row['predictions'].values()))
                totals[label]['correct']+=prediction==row['gold']
                totals[label]['total']+=1
                totals[label]['invalid']+=prediction is None
    return totals


def main():
    path=VENDOR/'data/testset.json'
    assert digest(path)=='a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
    papers=load(path)['papers'];assert len(papers)==97
    original=load(VENDOR/'data/papers.json')
    training_ids={p['document_id'] for p in load(ROOT/'inputs/frozen_cohort.json')['papers']}
    assert not training_ids & {p['document_id'] for p in papers}
    for paper in papers:
        assert sha_text(paper['fulltext'])==paper['fulltext_sha256']
        assert evaluate.question_signature(evaluate.questions_for(original[paper['original_paper_index']],paper['original_paper_index']))==evaluate.question_signature(paper['questions'])
    for r in [64,128]:
        trained=load(ROOT/'outputs'/('train-r'+str(r))/'result.json')
        assert trained['completed_epochs']==1 and trained['source_papers']==865 and trained['lora_rank']==r
    output=ROOT/'outputs/evaluation';output.mkdir(exist_ok=True)
    write(output/'protocol.json',dict(frozen_testset_sha256=digest(path),papers=97,questions_per_condition=970,
        generator_prompt_sha256=sha_text(summary_system(load(ROOT/'inputs/reference_prompts.json'))),
        thinking=True,generation_temperature=1.0,generation_top_p=.95,generation_top_k=20,
        generation_max_tokens=24576,format_attempts=3,semantic_correction=False,
        primary_comparison='matched same-prompt same-thinking untrained Gemma versus each LoRA',
        historical_gemma86_60_not_a_matched_control=True,
        judge='Qwen/Qwen2.5-7B-Instruct',judge_revision='a09a35458c702b33eeacc393d103063234e8bc28',
        judge_dtype='BF16',judge_temperature=.5,judge_top_p=.95,judge_max_tokens=100,
        judge_frequency_penalty=1.05,judge_presence_penalty=1.05,judge_thinking=False,
        generation_failures_count_as_ten_wrong_answers=True,bootstrap_draws=10000,bootstrap_seed=250219413))
    servers=[]
    contexts={}
    try:
        def start(index):
            label=list(LABELS)[index]
            adapter=None if index==0 else ROOT/'outputs'/('train-r'+str([64,128][index-1]))/'adapter'
            return Server(ROOT/'models/gemma-4-12b-it',19200+index,'eval-'+label,parser='gemma4',gpu=index,adapter=adapter)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            servers=list(pool.map(start,range(3)))
            futures={label:pool.submit(cohort,server,label,papers) for label,server in zip(LABELS,servers)}
            contexts={label:future.result() for label,future in futures.items()}
    finally:
        for server in servers:server.close()
    qa_path=output/'qa-results.json'
    results=load(qa_path) if qa_path.exists() else dict(labels=LABELS,documents=[dict(document_id=p['document_id'],
        fulltext_sha256=p['fulltext_sha256'],questions=p['questions'],conditions={}) for p in papers])
    validate(results,papers,contexts)
    judge=Server(ROOT/'models/Qwen2.5-7B-Instruct',19203,'eval-fixed-judge',gpu=0,judge=True)
    try:
        backend=OpenAICompatibleBackend('qa-model',base_url=judge.endpoint+'/v1',api_key='',max_tokens=100,
            temperature=.5,concurrency=4,thinking=False,frequency_penalty=1.05,presence_penalty=1.05,timeout=600)
        evaluate.CONDITIONS=('qwen_summary',)
        by_id={d['document_id']:d for d in results['documents']}
        for paper in papers:
            document=by_id[paper['document_id']]
            for label in LABELS:
                if label in document['conditions']:continue
                generated=contexts[label][paper['document_id']]
                context=generated['judge_context'];tick=time.monotonic()
                if generated['status']=='generated':
                    proxy=dict(document_id=paper['document_id'],viewer_config=paper['source']['viewer_config'],
                        fulltext=paper['fulltext'],existing_summary=paper['existing_summary'],qwen_summary=context)
                    value=evaluate.run_document(proxy,paper['questions'],backend,context_limit=32768)
                    value['generation_failed']=False
                else:
                    value=dict(generation_failed=True,rows=[dict(question_index=q['question_index'],gold=q['answer'],
                        predictions={'qwen_summary':None},responses={'qwen_summary':dict(generation_failed=True,
                            error=generated.get('error'),prompt_sha256=sha_text(evaluate.historical_answer_prompt(q['formatted_question'],'')))})
                        for q in paper['questions']])
                value['context_sha256']=sha_text(context);value['elapsed_seconds']=time.monotonic()-tick
                document['conditions'][label]=value
                write(qa_path,results)
                print('QA',paper['document_id'],label,flush=True)
    finally:judge.close()
    totals=validate(results,papers,contexts)
    assert all(value['total']==970 for value in totals.values())
    per={label:[sum(next(iter(row['predictions'].values()))==row['gold'] for row in d['conditions'][label]['rows'])
                for d in results['documents']] for label in LABELS}
    scores={label:dict(name=LABELS[label],**totals[label],accuracy=totals[label]['correct']/970,
                ci95=bootstrap(per[label]),failed_generation_papers=sum(r['status']!='generated' for r in contexts[label].values()),
                mean_narrative_words=statistics.mean(len(r['judge_context'].split()) for r in contexts[label].values())) for label in LABELS}
    paired={label+'_minus_base':dict(difference=sum(a-b for a,b in zip(per[label],per['gemma12_base']))/970,
                ci95=bootstrap([a-b for a,b in zip(per[label],per['gemma12_base'])])) for label in ['gemma12_r64','gemma12_r128']}
    performance={label:load(output/(label+'-generation-performance.json')) for label in LABELS}
    training={str(r):load(ROOT/'outputs'/('train-r'+str(r))/'result.json') for r in [64,128]}
    report=dict(papers=97,questions=970,models=scores,paired_differences=paired,protocol=load(output/'protocol.json'),
                generation_performance=performance,training=training,
                verified_prompt_hashes=2910-sum(v['failed_generation_papers']*10 for v in scores.values()),
                actual_condition_answers=2910,external_training_overlap=0)
    write(output/'report.json',report)
    lines=['# Gemma 4 12B IT: Qwen generator-reasoning distillation','',
        '865 training papers; ranks 64 and 128, one epoch each. Summary generation only. '
        'Matched untrained baseline uses exactly the same prompt, thinking mode and sampling. '
        '97 frozen evaluation papers / 970 immutable MCQs per condition.','',
        '| Model | Correct / 970 | QA accuracy | 95% paper-bootstrap CI | Failed generation papers | Mean narrative words |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for v in scores.values():lines.append(f"| {v['name']} | {v['correct']} | {100*v['accuracy']:.2f}% | {100*v['ci95'][0]:.2f}–{100*v['ci95'][1]:.2f}% | {v['failed_generation_papers']} | {v['mean_narrative_words']:.0f} |")
    lines+=['','## Paired QA differences','']
    for label,v in paired.items():lines.append(f"- {label}: {100*v['difference']:+.2f} percentage points; paired CI {100*v['ci95'][0]:+.2f} to {100*v['ci95'][1]:+.2f}.")
    lines+=['','## Generation time and native summary length','',
        '| Model | Generation seconds | Completion tokens/s/GPU | Mean narrative tokens | API calls |',
        '| --- | ---: | ---: | ---: | ---: |']
    for label,v in performance.items():
        lines.append(f"| {LABELS[label]} | {v['elapsed_seconds']:.1f} | {v['output_tokens_per_second']:.1f} | {v['mean_native_narrative_tokens']:.1f} | {v['api_calls']} |")
    lines+=['','Completion throughput includes actual thinking and JSON output from all format attempts. '
        'Time includes generator requests and tokenization, excluding server startup and QA. '
        'Three model servers run concurrently on three GPUs; Slurm allocates one four-GPU node.','',
        '## One-epoch training','',
        '| Rank | Peak learning rate | Trainable parameters | Compute seconds | Worker seconds including setup/save | Compute GPU-hours (8 GPUs) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r,v in training.items():
        lines.append(f"| {r} | {v['learning_rate']} | {v['trainable_parameters']:,} | {v['compute_seconds']:.1f} | {v['total_worker_seconds']:.1f} | {8*v['compute_seconds']/3600:.2f} |")
    lines+=['','Both ranks were restarted from the base checkpoint at the same 2e-5 learning rate '
        'after rank 128 diverged during an initial 1e-4 attempt. Cancelled-attempt GPU time is '
        'recorded separately in the allocation accounting; it is not hidden in successful compute time.']
    lines+=['','The targets are original source-only Qwen generator outputs and their actual reasoning, '
        'not corrected final summaries with mismatched reasoning. The earlier 86.60% Gemma raw score '
        'used different prompt/thinking/sampling settings and is a historical reference, not the matched '
        'no-LoRA control. QA measures answerability under the fixed student and synthetic MCQs; it '
        'does not establish independent human factual superiority. Every paper and failed output is retained.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    write(output/'complete.json',dict(job_id=os.environ['SLURM_JOB_ID'],papers=97,conditions=3,scored_answers=2910,
         report=str(output/'RESULTS.md'),audit_passed=True))


if __name__=='__main__':main()
