"""Fixed historical MCQ prompts/parser; continuous batching without repair feedback."""
import copy
import os
import sys
import time
from common import ROOT, PINNED, REFERENCE, ARMS, load, read_jsonl, write, sha
sys.path.insert(0, str(PINNED.parents[1] / 'src'))
from project_alexandria.experiments.mcq import historical_answer_prompt, extract_historical_choice
from project_alexandria.experiments.reproduce import JUDGE_SYSTEM_PROMPT

class Judge:
    def __init__(self):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.params = SamplingParams(temperature=.5, top_p=.95, max_tokens=100,
                                    frequency_penalty=1.05, presence_penalty=1.05)
        path = ROOT.parent / 'scientific-ornith-dflash-eval-97/models/Qwen2.5-7B-Instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = LLM(model=str(path), dtype='bfloat16', max_model_len=32768,
            max_num_seqs=64, max_num_batched_tokens=16384, gpu_memory_utilization=.88,
            enable_prefix_caching=True, enable_chunked_prefill=True,
            generation_config='vllm', disable_log_stats=True)

    def answer(self, prompts):
        ids = [self.tokenizer.apply_chat_template([
            dict(role='system', content=JUDGE_SYSTEM_PROMPT), dict(role='user', content=p)],
            tokenize=True, add_generation_prompt=True) for p in prompts]
        assert all(len(p)+100 <= 32768 for p in ids), 'Judge context overflow: no truncation allowed'
        rows = [dict(prompt_sha256=sha(p), prompt_tokens=len(t), attempts=[], prediction=None,
                     context_limit=32768) for p,t in zip(prompts,ids)]
        pending = list(range(len(ids)))
        for attempt in range(5):
            if not pending: break
            outputs = self.model.generate([dict(prompt_token_ids=ids[i]) for i in pending],
                                          sampling_params=self.params, use_tqdm=False)
            retry=[]
            for i,output in zip(pending,outputs):
                text=output.outputs[0].text
                rows[i]['attempts'].append(text)
                rows[i]['prediction']=extract_historical_choice(text)
                if rows[i]['prediction'] is None: retry.append(i)
            pending=retry
        return rows

def main():
    cohort=read_jsonl(ROOT/'inputs/cohort.jsonl')
    test={p['document_id']:p for p in load(PINNED/'data/testset.json')['papers']}
    cached={d['document_id']:d for d in load(REFERENCE/'outputs/evaluation/qa-results.json')['documents']}
    judge=None;pending=list(ARMS)
    write(ROOT/'outputs/qa_protocol.json',dict(judge_model='Qwen/Qwen2.5-7B-Instruct',
        temperature=.5,top_p=.95,max_tokens=100,frequency_penalty=1.05,presence_penalty=1.05,
        historical_prompt_and_parser=True,invalid_answer_retries=4,max_num_seqs=64,
        max_num_batched_tokens=16384,context_limit=32768,source_and_summary_truncation=False,
        unchanged_contexts_reuse_exact_cached_responses=True,qa_does_not_control_repairs=True,
        native_vllm_batching=True,sampling_schedule_differs_from_previous_http_concurrency4=True,
        sampled_judge_results_have_sampling_noise=True))
    while pending:
        for arm in pending[:]:
            output=ROOT/'outputs/qa'/arm
            if (output/'complete.json').exists(): pending.remove(arm);continue
            quality=ROOT/'outputs/quality'/arm
            if not (quality/'complete.json').exists(): continue
            tick=time.monotonic();repaired={r['uid']:r for r in read_jsonl(quality/'summaries.jsonl')}
            documents=[];slots=[];prompts=[]
            for c in cohort:
                p=test[c['document_id']];r=repaired[c['uid']]
                baseline=copy.deepcopy(cached[c['document_id']]['conditions'][c['original_condition']])
                assert len(baseline['rows'])==10 and cached[c['document_id']]['questions']==p['questions']
                for row,q in zip(baseline['rows'],p['questions']):
                    assert row['gold']==q['answer'] and row['question_index']==q['question_index']
                    assert row['responses']['qwen_summary']['prompt_sha256']==sha(historical_answer_prompt(q['formatted_question'],c['judge_context']))
                unchanged=sha(r['judge_context'])==sha(c['judge_context'])
                doc=dict(uid=c['uid'],document_id=c['document_id'],original_condition=c['original_condition'],
                         source_sha256=c['source_sha256'],original=baseline,
                         repaired=copy.deepcopy(baseline) if unchanged else None,
                         unchanged_context_reused=unchanged,guarded_context_sha256=sha(r['judge_context']))
                documents.append(doc)
                if not unchanged:
                    for q in p['questions']:
                        prompts.append(historical_answer_prompt(q['formatted_question'],r['judge_context']))
                        slots.append((len(documents)-1,q))
            if prompts:
                if judge is None:
                    setup=time.monotonic();judge=Judge()
                    write(ROOT/'outputs/setup/judge.json',dict(model_load_seconds=time.monotonic()-setup))
                records=judge.answer(prompts)
                for (index,q),record in zip(slots,records):
                    doc=documents[index]
                    if doc['repaired'] is None:doc['repaired']=dict(rows=[],generation_failed=False)
                    record.update(question_index=q['question_index'],condition='qwen_summary')
                    doc['repaired']['rows'].append(dict(question_index=q['question_index'],gold=q['answer'],
                        predictions={'qwen_summary':record['prediction']},responses={'qwen_summary':record}))
            for d in documents:
                assert len(d['repaired']['rows'])==10
            write(output/'qa-results.json',dict(arm=arm,documents=documents))
            write(output/'complete.json',dict(complete=True,summary_versions=200,unique_papers=97,
                evaluated_questions=2000,new_question_requests=len(prompts),
                reused_question_responses=2000-len(prompts),seconds=time.monotonic()-tick))
            print('QA COMPLETE',arm,len(prompts),'new questions',flush=True);pending.remove(arm)
        if pending:time.sleep(2)

if __name__=='__main__':main()
