import argparse,json,math
from pathlib import Path
from common import ROOT
from ngram_overlap import SourceIndex,fragments,audit_summary,tokenize
from reference_validator import narrative_context
from passages import patch
import insertions as editor

def main():
    p=argparse.ArgumentParser(description='Apply the frozen jointly planned insertion system to a saved summary.')
    p.add_argument('--summary',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--mode',choices=['system_greedy','system_joint','system_core'],default='system_core');args=p.parse_args()
    editor.init();source=args.source.read_text();index=SourceIndex(source);saved=json.loads(args.summary.read_text());original=saved.get('summary',saved)
    before=audit_summary(source,original,index=index,expanded_diagnostic=False);budget=math.floor(before['categories']['narrative']['words']*.03);changes={};traces=[]
    for text,path,kind in fragments(original):
        if kind!='narrative':continue
        after,plans,_=editor.repair(text,index,args.mode,max_add_words=budget);assert editor.verify_trace(text,after,plans,index)
        if text!=after:
            changes[path,text]=after;traces.extend(dict(t,path=path) for t in plans);budget-=len(tokenize(after)[1])-len(tokenize(text)[1])
    summary=patch(original,changes);context=narrative_context(summary);assert editor.scientific_guard(narrative_context(original),context)
    audit=audit_summary(source,summary,index=index);rollback=audit['categories']['narrative']['longest_contiguous_match_words']>before['categories']['narrative']['longest_contiguous_match_words']
    if rollback:summary=original;context=narrative_context(original);audit=audit_summary(source,original,index=index)
    value=dict(summary=summary,judge_context=context,audit=audit,traces=traces,source_sha256=index.sha256,rolled_back=rollback,mode=args.mode)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
