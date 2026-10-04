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
        'gemma12_r64':'Gemma 4 12B IT / Ornith-distilled rank 64, one epoch',
        'gemma12_r128':'Gemma 4 12B IT / Ornith-distilled rank 128, one epoch'}


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


def finalize(results,papers,contexts,output):
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
    lines=['# Gemma 4 12B IT: Ornith generator-reasoning distillation','',
        '736 training papers; ranks 64 and 128, one epoch each. Summary generation only. '
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
        'The baseline output and QA responses are reused verbatim from the audited matched Qwen study. Two LoRA generators and two QA replicas share one four-GPU node; the baseline generation time is historical, not newly billed.','',
        '## One-epoch training','',
        '| Rank | Peak learning rate | Trainable parameters | Compute seconds | Worker seconds including setup/save | Compute GPU-hours (8 GPUs) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r,v in training.items():
        lines.append(f"| {r} | {v['learning_rate']} | {v['trainable_parameters']:,} | {v['compute_seconds']:.1f} | {v['total_worker_seconds']:.1f} | {8*v['compute_seconds']/3600:.2f} |")
    lines+=['','Both ranks train from the same base checkpoint at a 2e-5 peak learning rate. '
        'The earlier Qwen-teacher experiment had a cancelled 1e-4 attempt; it is not part of this Ornith study. GPU time is '
        'recorded separately in the allocation accounting; it is not hidden in successful compute time.']
    lines+=['','The targets are original source-only Ornith generator outputs and their actual reasoning, '
        'not corrected final summaries with mismatched reasoning. The earlier 86.60% Gemma raw score '
        'used different prompt/thinking/sampling settings and is a historical reference, not the matched '
        'no-LoRA control. QA measures answerability under the fixed student and synthetic MCQs; it '
        'does not establish independent human factual superiority. Every paper and failed output is retained. These targets are unreviewed raw Ornith outputs: all 736 failed the stricter source-evidence schema check. Summary length and failure rates must be considered alongside QA accuracy.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    write(output/'complete.json',dict(job_id=os.environ['SLURM_JOB_ID'],papers=97,conditions=3,scored_answers=2910,
         report=str(output/'RESULTS.md'),audit_passed=True))


def main():
    import shutil
    import signal
    import subprocess
    reference=Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004')
    path=VENDOR/'data/testset.json'
    assert digest(path)=='a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
    papers=load(path)['papers'];assert len(papers)==97
    original=load(VENDOR/'data/papers.json')
    training_ids={p['document_id'] for p in load(ROOT/'inputs/frozen_cohort.json')['papers']}
    assert len(training_ids)==736 and not training_ids & {p['document_id'] for p in papers}
    for paper in papers:
        assert sha_text(paper['fulltext'])==paper['fulltext_sha256']
        assert evaluate.question_signature(evaluate.questions_for(original[paper['original_paper_index']],paper['original_paper_index']))==evaluate.question_signature(paper['questions'])
    for r in [64,128]:
        trained=load(ROOT/'outputs'/('train-r'+str(r))/'result.json')
        assert trained['completed_epochs']==1 and trained['source_papers']==736 and trained['lora_rank']==r
        assert trained['teacher']['model']=='ornith-ai/Ornith-1.5-9B'
    output=ROOT/'outputs/evaluation';output.mkdir(exist_ok=True)
    protocol=dict(frozen_testset_sha256=digest(path),papers=97,questions_per_condition=970,
        generator_prompt_sha256=sha_text(summary_system(load(ROOT/'inputs/reference_prompts.json'))),
        thinking=True,generation_temperature=1.0,generation_top_p=.95,generation_top_k=20,
        generation_max_tokens=24576,format_attempts=3,semantic_correction=False,
        primary_comparison='matched same-prompt same-thinking untrained Gemma versus each Ornith-trained LoRA',
        historical_gemma86_60_not_a_matched_control=True,
        judge='Qwen/Qwen2.5-7B-Instruct',judge_revision='a09a35458c702b33eeacc393d103063234e8bc28',
        judge_dtype='BF16',judge_temperature=.5,judge_top_p=.95,judge_max_tokens=100,
        judge_frequency_penalty=1.05,judge_presence_penalty=1.05,judge_thinking=False,
        generation_failures_count_as_ten_wrong_answers=True,bootstrap_draws=10000,bootstrap_seed=250219413,
        reused_baseline=True,baseline_reference=str(reference/'outputs/evaluation'),
        baseline_qa_sha256=digest(reference/'outputs/evaluation/qa-results.json'),
        teacher='ornith-ai/Ornith-1.5-9B',training_papers=736)
    old_protocol=load(reference/'outputs/evaluation/protocol.json')
    for key in ['frozen_testset_sha256','generator_prompt_sha256','thinking','generation_temperature',
                'generation_top_p','generation_top_k','generation_max_tokens','format_attempts',
                'semantic_correction','judge','judge_revision','judge_dtype','judge_temperature',
                'judge_top_p','judge_max_tokens','judge_frequency_penalty','judge_presence_penalty','judge_thinking']:
        assert protocol[key]==old_protocol[key], 'Cached baseline must use identical evaluation settings'
    assert load(reference/'outputs/evaluation/complete.json')['audit_passed']
    write(output/'protocol.json',protocol)
    for paper in papers:
        folder=output/'generation/gemma12_base'/paper['document_id'];folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(reference/'outputs/evaluation/generation/gemma12_base'/paper['document_id']/'result.json',folder/'result.json')
    shutil.copy2(reference/'outputs/evaluation/gemma12_base-summaries.jsonl',output/'gemma12_base-summaries.jsonl')
    base_perf=load(reference/'outputs/evaluation/gemma12_base-generation-performance.json')
    base_perf.update(reused_baseline=True,baseline_generation_not_billed_in_this_study=True)
    write(output/'gemma12_base-generation-performance.json',base_perf)
    checkpoint=output/'qa-live-results.json'
    if not checkpoint.exists():
        old_qa=load(reference/'outputs/evaluation/qa-results.json')
        results=dict(labels=LABELS,documents=[dict(document_id=d['document_id'],fulltext_sha256=d['fulltext_sha256'],
            questions=d['questions'],conditions={'gemma12_base':d['conditions']['gemma12_base']}) for d in old_qa['documents']])
        write(checkpoint,results)
    servers=[];qa=None
    qa_log=(ROOT/'logs/qa-'+os.environ['SLURM_JOB_ID']+'.log').open('a')
    try:
        env=dict(os.environ,QA_GPU_INDICES='0,3')
        qa=subprocess.Popen([sys.executable,str(ROOT/'code/live_qa.py')],env=env,stdout=qa_log,stderr=qa_log,start_new_session=True)
        def start(index):
            label=['gemma12_r64','gemma12_r128'][index]
            adapter=ROOT/'outputs'/('train-r'+str([64,128][index]))/'adapter'
            return Server(ROOT/'models/gemma-4-12b-it',19201+index,'eval-'+label,parser='gemma4',gpu=1+index,adapter=adapter)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(start,i) for i in range(2)]:servers.append(future.result())
            futures=[pool.submit(cohort,server,label,papers) for server,label in zip(servers,['gemma12_r64','gemma12_r128'])]
            for future in futures:future.result()
        for server in servers:server.close()
        servers=[]
        assert qa.wait(timeout=7200)==0, 'QA worker failed; inspect the owned QA log'
        complete=load(output/'complete.json')
        assert complete['audit_passed'] and complete['scored_answers']==2910
    finally:
        for server in servers:server.close()
        if qa and qa.poll() is None:
            os.killpg(qa.pid,signal.SIGTERM)
            try:qa.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(qa.pid,signal.SIGKILL);qa.wait(timeout=10)
        qa_log.close()


if __name__=='__main__':main()
