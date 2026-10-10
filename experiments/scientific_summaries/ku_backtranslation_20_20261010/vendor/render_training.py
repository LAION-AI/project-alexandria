"""Render real teacher reasoning with each student's own chat template; never truncate."""
import argparse
import copy
import json
import os
from pathlib import Path


def render(tok, messages):
    thinking=bool(messages[-1].get('reasoning_content'))
    kwargs={'enable_thinking':thinking,'preserve_thinking':True,'reasoning_effort':'medium'}
    text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,**kwargs)
    prefix=tok.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,**kwargs)
    transformation=None
    if not text.startswith(prefix):
        common=os.path.commonprefix([text,prefix])
        base=tok.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=False,**kwargs)
        suffix=prefix[len(common):]
        # Gemma's no-thinking inference prefix includes an empty thought channel
        # which its complete-dialogue template omits for an answer-only assistant.
        # Insert exactly that native prefix fragment, with no change to the answer.
        if thinking or len(common)<len(base) or suffix not in ('<|channel>thought\n<channel|>','<think>\n\n</think>\n\n'):
            raise ValueError('Cannot derive a reliable assistant loss boundary from this template')
        text=prefix+text[len(common):]
        transformation='inserted_native_empty_thinking_prefix'
    if thinking and messages[-1]['reasoning_content'] not in text:
        raise ValueError('Student template dropped teacher reasoning')
    return text,prefix,transformation


def main():
    from transformers import AutoTokenizer
    ap=argparse.ArgumentParser()
    ap.add_argument('--tokenizer',required=True,help='Local pinned Qwen3.5-9B or Gemma-4-12B-it tokenizer snapshot')
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-length',type=int,default=65536)
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    counts={'written':0,'too_long':0}
    with args.input.open() as source,args.output.open('w') as out:
        for line in source:
            row=json.loads(line);messages=copy.deepcopy(row['messages'])
            thinking=bool(messages[-1].get('reasoning_content'))
            text,prefix,transformation=render(tok,messages)
            ids=tok.encode(text,add_special_tokens=False)
            if len(ids)>args.max_length:
                counts['too_long']+=1;continue
            # Offsets avoid token-boundary assumptions when masking prompt tokens.
            encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            labels=[token if start>=len(prefix) and end>start else -100
                    for token,(start,end) in zip(encoded['input_ids'],encoded['offset_mapping'])]
            if all(v==-100 for v in labels):
                raise ValueError('No assistant target tokens')
            out.write(json.dumps({'document_id':row['document_id'],'domain':row['domain'],'split':row['split'],
                      'task':row['task'],'text':text,'input_ids':encoded['input_ids'],'labels':labels,
                      'prompt_characters':len(prefix),'reasoning_in_target':thinking,
                      'template_transformation':transformation},ensure_ascii=False)+'\n')
            counts['written']+=1
    args.output.with_suffix('.stats.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts))


if __name__=='__main__':main()
