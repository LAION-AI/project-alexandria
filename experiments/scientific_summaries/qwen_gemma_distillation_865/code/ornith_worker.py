"""Generate on exactly the frozen Qwen cohort using Ornith 1.5 9B, native AR."""
import concurrent.futures
import os
import shutil
import sys
import time

from common import ROOT, SOURCE, load, write
from inference import Server, generate_paper

sys.path.insert(0,str(SOURCE/'code'))
from run import summary_system


def main():
    worker=int(os.environ['SLURM_PROCID']);world=int(os.environ['SLURM_NTASKS'])
    job_id=os.environ['SLURM_JOB_ID']
    offset=int(os.environ.get('ORNITH_ATTEMPT_OFFSET','0'))
    assert world==8
    cohort=load(ROOT/'inputs/frozen_cohort.json')
    sources={p['document_id']:p for p in load(SOURCE/'inputs/papers.json')}
    papers=[sources[p['document_id']] for p in cohort['papers'][worker::world]]
    pending=[paper for paper in papers if not (ROOT/'outputs/ornith/documents'/paper['document_id']/'result.json').exists()]
    if not pending:
        write(ROOT/'outputs/ornith'/('worker'+str(worker)+'.json'),dict(worker=worker,papers=len(papers),
              generated=len(papers),failed=0,seconds=0,new_papers=0,job_id=os.environ['SLURM_JOB_ID']))
        return
    prompts=load(ROOT/'inputs/reference_prompts.json');system=summary_system(prompts)
    server=Server(ROOT/'models/ornith-1.5-9b',19100+worker,'ornith-worker'+str(worker),parser='qwen3')
    start=time.monotonic()
    try:
        def task(paper):
            folder=ROOT/'outputs/ornith/documents'/paper['document_id']
            for retry_round in range(3):
                # Source-only format retries; no QA answers or semantic review feedback.
                budget=16384 if retry_round==0 else 12288
                try:
                    result=generate_paper(server.endpoint,paper,system,
                        prompts['summary-user'].replace('{paper_text}',paper['fulltext']),
                        folder,thinking_token_budget=budget,attempt_offset=offset+3*retry_round)
                except (OSError,ValueError) as e:
                    result=dict(document_id=paper['document_id'],status='generation_failed',
                                error='Request failure: '+str(e),judge_context='',job_id=job_id,
                                thinking_token_budget=budget,attempt_offset=offset+3*retry_round)
                    write(folder/'result.json',result)
                if result['status']=='generated' or retry_round==2:break
                archive=ROOT/'outputs/ornith_retry_attempts'/job_id/paper['document_id']/('round'+str(retry_round))
                archive.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(folder),str(archive))
            print(paper['document_id'],result['status'],flush=True)
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:results=list(pool.map(task,pending))
        all_results=[load(ROOT/'outputs/ornith/documents'/paper['document_id']/'result.json') for paper in papers]
        record=dict(worker=worker,papers=len(papers),
            generated=sum(r['status']=='generated' for r in all_results),failed=sum(r['status']!='generated' for r in all_results),
            model='ornith-ai/Ornith-1.5-9B',model_revision='489cb97981b8654bcfcf30ce1f94ed1b62e07b53',
            precision='BF16',speculative_decoding=False,thinking=True,concurrency=16,
            thinking_token_budget=16384,retry_thinking_token_budget=12288,max_source_only_calls_per_paper=9,
            attempt_offset=offset,new_papers=len(pending),retained_existing_papers=len(papers)-len(pending),
            seconds=time.monotonic()-start,job_id=job_id)
        write(ROOT/'outputs/ornith'/('worker'+str(worker)+'.json'),record)
        write(ROOT/'outputs/ornith/phases'/job_id/('worker'+str(worker)+'.json'),record)
    finally:server.close()


if __name__=='__main__':main()
