"""CPU preflight of native chat templates on source-only inputs; no MCQ access."""
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from common import ROOT,PROMPT,load,write,source_user,correction_user,digest

def check(model):
    directory=ROOT/'models'/('gemma-4-E4B-it' if model=='gemma4_e4b' else 'gemma-4-12B-it')
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    rows=[]
    for label,items in [('evaluation_generation',load(ROOT/'inputs/eval_sources_only.json')),
                        ('tuning_generation',load(ROOT/'inputs/throughput_sources_only.json')),
                        ('tuning_correction',load(ROOT/'inputs/throughput_sources_only.json'))]:
        for paper in items:
            user=correction_user(paper,paper['summary'],dict(instruction='Check every narrative statement against the source and preserve supported detail.')) if label.endswith('correction') else source_user(paper)
            messages=[dict(role='system',content=PROMPT),dict(role='user',content=user)]
            ids=tokenizer.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,tokenize=True)
            if hasattr(ids,'keys'):ids=ids['input_ids']
            if ids and isinstance(ids[0],list):ids=ids[0]
            budget=16000 if label=='evaluation_generation' else 512
            if len(ids)+budget>65536:raise ValueError('Untruncated context does not fit '+model+' '+paper['document_id'])
            rows.append(dict(document_id=paper['document_id'],phase=label,input_tokens=len(ids),output_budget=budget,user_prompt_sha256=digest(user)))
    result=dict(model=model,passed=True,thinking=False,context_limit=65536,
        max_initial_generation_tokens=max(r['input_tokens'] for r in rows if r['phase']=='evaluation_generation'),rows=rows)
    print(model,'PREFLIGHT PASSED',result['max_initial_generation_tokens'],flush=True)
    return result

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(check,('gemma4_e4b','gemma4_12b')))
    write(ROOT/'inputs/token_preflight.json',dict(passed=True,models=results))
