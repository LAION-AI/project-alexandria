"""Score completed matched paper triples while the three generators continue."""
import concurrent.futures
import json
import os
import time

from common import ROOT,load,write,digest
from inference import Server
from evaluate_loras import LABELS,VENDOR,evaluate,OpenAICompatibleBackend,sha_text,validate,finalize


def main():
    output=ROOT/'outputs/evaluation'
    if (output/'complete.json').exists():
        assert load(output/'complete.json')['scored_answers']==2910
        return
    gpu_indices=[int(i) for i in os.environ.get('QA_GPU_INDICES','0,1,2,3').split(',')]
    replicas=len(gpu_indices)
    testset=VENDOR/'data/testset.json'
    assert digest(testset)=='a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
    papers=load(testset)['papers'];assert len(papers)==97
    training_ids={p['document_id'] for p in load(ROOT/'inputs/frozen_cohort.json')['papers']}
    assert not training_ids&{p['document_id'] for p in papers}
    path=output/'qa-live-results.json'
    results=load(path) if path.exists() else dict(labels=LABELS,documents=[dict(
        document_id=p['document_id'],fulltext_sha256=p['fulltext_sha256'],questions=p['questions'],conditions={}) for p in papers])
    documents={d['document_id']:d for d in results['documents']}
    protocol=load(output/'protocol.json')
    protocol.update(qa_execution='Identical pinned BF16 QA replicas; matched paper triples scored as soon as their summaries complete',
                    qa_replicas=replicas,qa_gpu_indices=gpu_indices,qa_concurrency_per_replica=4,qa_job_id=os.environ['SLURM_JOB_ID'],
                    generation_parameters_changed=False)
    write(output/'protocol.json',protocol)
    servers=[]
    try:
        def start(i):
            return Server(ROOT/'models/Qwen2.5-7B-Instruct',19500+i,'eval-live-judge'+str(i),gpu=i,judge=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as pool:
            for future in [pool.submit(start,i) for i in gpu_indices]:servers.append(future.result())
        backends=[OpenAICompatibleBackend('qwen25',base_url=server.endpoint+'/v1',api_key='',max_tokens=100,
                    temperature=.5,concurrency=4,thinking=False,frequency_penalty=1.05,presence_penalty=1.05,timeout=600) for server in servers]
        evaluate.CONDITIONS=('qwen_summary',)
        def task(paper,index):
            generated={label:load(output/'generation'/label/paper['document_id']/'result.json') for label in LABELS}
            document=documents[paper['document_id']]
            additions={}
            for label in LABELS:
                if label in document['conditions']:continue
                row=generated[label];context=row['judge_context'];tick=time.monotonic()
                if row['status']=='generated':
                    proxy=dict(document_id=paper['document_id'],viewer_config=paper['source']['viewer_config'],
                               fulltext=paper['fulltext'],existing_summary=paper['existing_summary'],qwen_summary=context)
                    value=evaluate.run_document(proxy,paper['questions'],backends[index],context_limit=32768)
                    value['generation_failed']=False
                else:
                    value=dict(generation_failed=True,rows=[dict(question_index=q['question_index'],gold=q['answer'],
                        predictions={'qwen_summary':None},responses={'qwen_summary':dict(generation_failed=True,
                            error=row.get('error'),prompt_sha256=sha_text(evaluate.historical_answer_prompt(q['formatted_question'],'')))}) for q in paper['questions']])
                value.update(context_sha256=sha_text(context),elapsed_seconds=time.monotonic()-tick)
                additions[label]=value
            return paper['document_id'],additions
        pending=list(papers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=replicas) as pool:
            while pending:
                # A checkpoint resumes whole matched paper triples, without selecting
                # papers using scores, questions, or gold answers.
                batch=[p for p in pending if all((output/'generation'/label/p['document_id']/'result.json').exists() for label in LABELS)][:replicas]
                if not batch:
                    time.sleep(5);continue
                for future in concurrent.futures.as_completed([pool.submit(task,p,i) for i,p in enumerate(batch)]):
                    docid,additions=future.result()
                    documents[docid]['conditions'].update(additions)
                    write(path,results)
                    pending=[p for p in pending if p['document_id']!=docid]
                    print('QA completed matched paper',docid,'remaining',len(pending),flush=True)
        while not all((output/(label+'-generation-performance.json')).exists() for label in LABELS):time.sleep(5)
        contexts={label:{p['document_id']:load(output/'generation'/label/p['document_id']/'result.json') for p in papers} for label in LABELS}
        totals=validate(results,papers,contexts)
        assert all(t['total']==970 for t in totals.values())
        write(output/'qa-results.json',results)
        finalize(results,papers,contexts,output)
    finally:
        for server in servers:server.close()


if __name__=='__main__':main()
