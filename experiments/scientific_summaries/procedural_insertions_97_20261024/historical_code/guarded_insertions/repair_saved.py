"""Apply the measured optional-insertion stage to one saved structured summary."""
import argparse,json,time
from pathlib import Path
from common import ROOT
from ngram_overlap import SourceIndex,fragments,audit_summary
from reference_validator import narrative_context,validate_summary
from passages import patch
import insertions as editor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--mode',choices=['grammar_insert','expanded_insert'],default='grammar_insert')
    p.add_argument('--nltk-data',type=Path);args=p.parse_args()
    if args.nltk_data:
        import nltk
        nltk.data.path.insert(0,str(args.nltk_data))
    editor.init();source=args.source.read_text();index=SourceIndex(source)
    saved=json.loads(args.summary.read_text());original=saved.get('summary',saved)
    tick=time.process_time();changes={};traces=[]
    for text,path,kind in fragments(original):
        if kind!='narrative':continue
        after,edits=editor.repair(text,index,args.mode);assert editor.verify_trace(text,after,edits)
        if after!=text:
            changes[path,text]=after;traces.append(dict(path=path,before=text,after=after,edits=edits))
    summary=patch(original,changes);context=narrative_context(summary)
    assert editor.guard(narrative_context(original),context)
    protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
    assert protected(original)==protected(summary)
    try:validate_summary(json.dumps(original,ensure_ascii=False),source);valid=True
    except (ValueError,TypeError,KeyError):valid=False
    if valid:validate_summary(json.dumps(summary,ensure_ascii=False),source)
    audit=audit_summary(source,summary,index=index);before=audit_summary(source,original,index=index,expanded_diagnostic=False)
    n=audit['categories']['narrative'];b=before['categories']['narrative']
    rolled_back=(n['longest_contiguous_match_words']>b['longest_contiguous_match_words'] or n['words']>b['words']*1.03
                 or any(not editor.temperature_scope_safe(t['after']) for t in traces if any(e['rule']=='explicit_temperature' for e in t['edits'])))
    if rolled_back:summary=original;context=narrative_context(original);audit=audit_summary(source,original,index=index)
    value=dict(summary=summary,judge_context=context,audit=audit,source_sha256=index.sha256,mode=args.mode,traces=traces,rolled_back=rolled_back,cpu_seconds_excluding_setup=time.process_time()-tick)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
