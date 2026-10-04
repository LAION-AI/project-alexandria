"""Generate on exactly the frozen Qwen cohort using Ornith 1.5 9B, native AR."""
import concurrent.futures
import os
import sys
import time

from common import ROOT, SOURCE, load, write
from inference import Server, generate_paper

sys.path.insert(0,str(SOURCE/'code'))
from run import summary_system


def main():
    worker=int(os.environ['SLURM_PROCID']);world=int(os.environ['SLURM_NTASKS'])
    assert world==8
    cohort=load(ROOT/'inputs/frozen_cohort.json')
    sources={p['document_id']:p for p in load(SOURCE/'inputs/papers.json')}
    papers=[sources[p['document_id']] for p in cohort['papers'][worker::world]]
    prompts=load(ROOT/'inputs/reference_prompts.json');system=summary_system(prompts)
    server=Server(ROOT/'models/ornith-1.5-9b',19100+worker,'ornith-worker'+str(worker),parser='qwen3')
    start=time.monotonic()
    try:
        def task(paper):
            result=generate_paper(server.endpoint,paper,system,
                prompts['summary-user'].replace('{paper_text}',paper['fulltext']),
                ROOT/'outputs/ornith/documents'/paper['document_id'])
            print(paper['document_id'],result['status'],flush=True)
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:results=list(pool.map(task,papers))
        write(ROOT/'outputs/ornith'/('worker'+str(worker)+'.json'),dict(worker=worker,papers=len(papers),
            generated=sum(r['status']=='generated' for r in results),failed=sum(r['status']!='generated' for r in results),
            model='ornith-ai/Ornith-1.5-9B',model_revision='489cb97981b8654bcfcf30ce1f94ed1b62e07b53',
            precision='BF16',speculative_decoding=False,thinking=True,concurrency=16,
            seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID']))
    finally:server.close()


if __name__=='__main__':main()
