"""Length-bucketed GPU translation with full outputs and recorded batch sweeps."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
from common import ROOT,ARMS,load,read_jsonl,write,jsonl,sha


class Marian:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        self.torch=torch;torch.set_num_threads(8)
        self.models={};self.tokenizers={}
        for direction in ['en_de','de_en']:
            path=ROOT/'models'/('windy_'+direction)
            self.tokenizers[direction]=AutoTokenizer.from_pretrained(path,local_files_only=True)
            self.models[direction]=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True,
                dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()

    def translate(self,texts,direction,batch,beam=1):
        tick=time.monotonic();torch=self.torch;tok=self.tokenizers[direction];model=self.models[direction]
        encoded=[tok.encode(t,add_special_tokens=True) for t in texts]
        rows=[None]*len(texts);valid=[]
        for i,ids in enumerate(encoded):
            if len(ids)>512:rows[i]=dict(text=texts[i],status='input_overflow_not_translated',input_tokens=len(ids),output_tokens=0)
            else:valid.append(i)
        valid.sort(key=lambda i:len(encoded[i]))
        torch.cuda.reset_peak_memory_stats()
        for start in range(0,len(valid),batch):
            indices=valid[start:start+batch]
            features=tok.pad([dict(input_ids=encoded[i]) for i in indices],padding=True,return_tensors='pt')
            features={k:v.to('cuda') for k,v in features.items()}
            with torch.inference_mode():
                generated=model.generate(**features,num_beams=beam,do_sample=False,max_new_tokens=511,
                                         use_cache=True,renormalize_logits=True)
            sequences=generated.cpu().tolist();outputs=tok.batch_decode(generated,skip_special_tokens=True)
            for i,ids,text in zip(indices,sequences,outputs):
                emitted=ids[1:]
                count=emitted.index(tok.eos_token_id)+1 if tok.eos_token_id in emitted else len(emitted)
                status='translated' if count<511 and text.strip() else 'output_length_or_empty'
                rows[i]=dict(text=text,status=status,input_tokens=len(encoded[i]),output_tokens=count)
        torch.cuda.synchronize()
        seconds=time.monotonic()-tick;tokens=sum(r['output_tokens'] for r in rows)
        return rows,dict(seconds=seconds,output_tokens=tokens,input_tokens=sum(r['input_tokens'] for r in rows),
                         output_tokens_per_second=tokens/max(seconds,1e-9),batch_size=batch,num_beams=beam,
                         actual_largest_batch=min(batch,len(valid)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         inputs=len(texts),successful_outputs=sum(r['status']=='translated' for r in rows),
                         includes_tokenization_and_complete_outputs=True)


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


def optimize(engine,arm,beam):
    path=ROOT/'outputs/optimization'/(arm+'.json')
    if path.exists() and load(path).get('state')=='selected':return load(path)['selected_batch_size']
    probes=read_jsonl(ROOT/'inputs/probes.jsonl');texts=[p['text'] for p in probes]
    engine.translate(texts[:4],'en_de',4,beam)
    engine.translate(texts[:4],'de_en',4,beam)
    results=[]
    for batch in [64,128,256,512]:
        if hasattr(engine,'reset'):engine.reset()
        german,forward=engine.translate(texts,'en_de',batch,beam)
        if hasattr(engine,'reset'):engine.reset()
        returned,back=engine.translate([r['text'] for r in german],'de_en',batch,beam)
        total=forward['seconds']+back['seconds'];tokens=forward['output_tokens']+back['output_tokens']
        value=dict(batch_size=batch,forward=forward,backward=back,roundtrip_seconds=total,
                   successful_roundtrip_count=sum(a['status']==b['status']=='translated' for a,b in zip(german,returned)),
                   roundtrip_output_tokens_per_second=tokens/total)
        results.append(value);write(path,dict(state='tuning',results=results,uses_heldout_qa=False))
        print('BATCH',arm,batch,round(tokens/total,2),'tokens/s',flush=True)
    chosen=max(results,key=lambda r:(r['successful_roundtrip_count'],r['roundtrip_output_tokens_per_second']))
    write(path,dict(state='selected',selected_batch_size=chosen['batch_size'],results=results,
                    uses_heldout_qa=False,probe_count=len(texts),
                    selection_priority=['complete_roundtrip_count','roundtrip_output_tokens_per_second'],
                    probe_document_ids=sorted({p['document_id'] for p in probes})))
    return chosen['batch_size']


def run_arm(engine,arm,beam):
    folder=ROOT/'outputs/translation'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    windows=read_jsonl(ROOT/'inputs/windows.jsonl');batch=optimize(engine,arm,beam)
    if hasattr(engine,'reset'):engine.reset()
    tick=time.monotonic()
    german,forward=engine.translate([w['text'] for w in windows],'en_de',batch,beam)
    jsonl(folder/'forward.jsonl',[dict(r,window_id=w['window_id']) for w,r in zip(windows,german)])
    valid=[i for i,r in enumerate(german) if r['status']=='translated']
    if hasattr(engine,'reset'):engine.reset()
    translated,back=engine.translate([german[i]['text'] for i in valid],'de_en',batch,beam)
    backward={i:r for i,r in zip(valid,translated)}
    rows=[]
    for i,w in enumerate(windows):
        result=backward.get(i,dict(text=w['text'],status='forward_failed_not_translated',input_tokens=0,output_tokens=0))
        rows.append(dict(result,window_id=w['window_id'],uid=w['uid'],original_text=w['text'],german=german[i]['text'],
                         forward_status=german[i]['status']))
    jsonl(folder/'roundtrip.jsonl',rows)
    elapsed=time.monotonic()-tick
    perf=dict(arm=arm,summary_versions=200,unique_papers=97,window_count=len(windows),
              successful_roundtrips=sum(r['status']=='translated' for r in rows),
              batch_size=batch,num_beams=beam,roundtrip_seconds=elapsed,
              forward=forward,backward=back,active_gpus=1,active_translation_gpu_hours=elapsed/3600,
              summaries_per_hour=200*3600/elapsed,windows_per_second=len(windows)/elapsed,
              combined_native_output_tokens_per_second=(forward['output_tokens']+back['output_tokens'])/elapsed,
              native_tokens_not_comparable_between_model_tokenizers=True,
              setup_tuning_and_quality_checks_excluded=True,quality_checked_separately=True,
              single_round=True,pivot='de',job_id=os.environ.get('SLURM_JOB_ID'))
    write(folder/'performance.json',perf)
    write(folder/'complete.json',dict(complete=True,summary_versions=200,window_count=len(windows)))
    print('TRANSLATION COMPLETE',arm,round(elapsed,3),'seconds',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['windy','translategemma'],required=True)
    family=parser.parse_args().family;start=time.monotonic()
    engine=Marian() if family=='windy' else TranslateGemma()
    write(ROOT/'outputs/setup'/(family+'.json'),dict(model_load_seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID')))
    if family=='windy':
        run_arm(engine,'windy_greedy',1);run_arm(engine,'windy_beam4',4)
    else:run_arm(engine,'translategemma',1)


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        traceback.print_exc()
        # An unhandled exception can otherwise wait indefinitely for vLLM children.
        import sys
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
