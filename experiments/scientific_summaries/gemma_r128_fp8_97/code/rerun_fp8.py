"""Repeat only the unmeasured FP8 arm after fixing a generator/judge port collision."""
import concurrent.futures as cf
import importlib.metadata
import os
from pathlib import Path
import threading
import benchmark as b
from common import ROOT,load,write,digest

PRIOR=Path('/e/fscratch/reformo/schuhmann1/scientific-gemma-r128-no-thinking-20261004')


def main():
    assert load(PRIOR/'outputs/evaluation/complete.json')['audit_passed']
    assert load(PRIOR/'outputs/evaluation/report.json')['models']['no_thinking_merged']['correct']==911
    path=b.VENDOR/'data/testset.json'
    assert digest(path)=='a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
    papers=load(path)['papers'];probes=load(ROOT/'inputs/optimization_papers.json')
    training={p['document_id'] for p in load(ROOT/'inputs/frozen_cohort.json')['papers']}
    assert len(papers)==97 and sum(len(p['questions']) for p in papers)==970
    assert {p['document_id'] for p in probes}<=training
    assert not training&{p['document_id'] for p in papers}
    runtime=load(ROOT/'inputs/runtime.json')
    for name in ['vllm','torch','transformers','tokenizers']:
        assert importlib.metadata.version(name)==runtime['packages'][name]
    for p in papers:assert b.sha(p['fulltext'])==p['fulltext_sha256']
    prompts=load(ROOT/'inputs/reference_prompts.json')
    protocol=load(b.OUT/'protocol.json')
    protocol.update(job_id=os.environ.get('SLURM_JOB_ID'),fp8_repeat=True,
        repeat_reason='First FP8 server used the QA port 19603; health/reset requests hit the existing judge. FP8 compatibility was not measured.',
        cached_no_thinking_bf16_reference=str(PRIOR),cached_conditions_reused=['thinking_reference','no_thinking_matched','no_thinking_merged'],
        cached_qa_sha256=digest(PRIOR/'outputs/evaluation/qa-results.json'),port_assignment=dict(generator=19600,judge=19603),
        source_copy_overlap_audit_required=True,copy_overlap_word_limit=5,copy_overlap_six_words_borderline=True)
    write(b.OUT/'protocol.json',protocol)
    b.STATES['no_thinking_matched']=dict(state='complete',reused=True)
    b.STATES['no_thinking_merged']=dict(state='complete',reused=True,
        config=load(PRIOR/'outputs/optimization/no_thinking_merged.json')['selection']['config'])
    b.state('no_thinking_fp8',state='prepared')
    done=threading.Event();errors=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        qa=pool.submit(b.qa_loop,papers,done)
        try:
            b.tuned('no_thinking_fp8',0,papers,probes,prompts,[
                dict(max_batched_tokens=16384,quantization='fp8_per_tensor'),
                dict(max_batched_tokens=16384,quantization='fp8_per_tensor',kv_cache_dtype='fp8')])
        except Exception as e:errors.append(repr(e))
        finally:done.set()
        qa.result()
    write(ROOT/'outputs/runtime_errors.json',dict(errors=errors))
    b.finalize(papers)
    from publish import publish
    publish()


if __name__=='__main__':main()
