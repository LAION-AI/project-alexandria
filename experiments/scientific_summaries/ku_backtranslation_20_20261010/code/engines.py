# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
import time
from common import ROOT,load
class TranslateGemma:
    def __init__(self):
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        self.SamplingParams=SamplingParams
        path=ROOT/'models/translategemma'
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.stop_ids=load(path/'generation_config.json')['eos_token_id']
        self.model=LLM(model=str(path),dtype='bfloat16',tensor_parallel_size=1,max_model_len=2048,
            max_num_seqs=512,max_num_batched_tokens=16384,gpu_memory_utilization=.88,
            enable_prefix_caching=True,enable_chunked_prefill=True,language_model_only=True,
            disable_log_stats=True,generation_config='vllm',seed=20261005)

    def reset(self):
        return self.model.reset_prefix_cache()

    def translate(self,texts,direction,batch,beam=1):
        tick=time.monotonic();source,target=direction.split('_');rows=[None]*len(texts);prompts=[];valid=[]
        for i,text in enumerate(texts):
            messages=[dict(role='user',content=[dict(type='text',source_lang_code=source,target_lang_code=target,text=text)])]
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            assert isinstance(ids,list) and all(isinstance(t,int) for t in ids), 'Need integer prompt token IDs'
            if len(ids)+768>2048:rows[i]=dict(text=text,status='input_overflow_not_translated',input_tokens=len(ids),output_tokens=0)
            else:valid.append((i,ids))
        valid.sort(key=lambda x:len(x[1]))
        for start in range(0,len(valid),batch):
            group=valid[start:start+batch]
            results=self.model.generate([dict(prompt_token_ids=ids) for _,ids in group],
                sampling_params=self.SamplingParams(temperature=0,max_tokens=768,seed=20261005,
                                                     stop_token_ids=self.stop_ids),use_tqdm=False)
            for (i,ids),result in zip(group,results):
                emitted=result.outputs[0]
                rows[i]=dict(text=emitted.text,status='translated' if emitted.finish_reason=='stop' and emitted.text.strip() else 'output_length_or_empty',
                             input_tokens=len(ids),output_tokens=len(emitted.token_ids),finish_reason=emitted.finish_reason)
        seconds=time.monotonic()-tick;tokens=sum(r['output_tokens'] for r in rows)
        return rows,dict(seconds=seconds,output_tokens=tokens,input_tokens=sum(r['input_tokens'] for r in rows),
                         output_tokens_per_second=tokens/max(seconds,1e-9),batch_size=batch,num_beams=1,
                         actual_largest_batch=min(batch,len(valid)),inputs=len(texts),
                         successful_outputs=sum(r['status']=='translated' for r in rows),
                         includes_tokenization_and_complete_outputs=True)

