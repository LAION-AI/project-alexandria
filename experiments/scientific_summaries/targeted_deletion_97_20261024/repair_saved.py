"""Apply one measured CPU deletion mode to a structured scientific summary."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
from common import ROOT
from ngram_overlap import SourceIndex,fragments,audit_summary
from passages import patch
from reference_validator import narrative_context,validate_summary
from critical_values import check
from lexicon import immutable

MODES={
    'fillers':('deletion_original',0),
    'articles_legacy':('deletion_original',1),
    'articles_guarded':('deletion_case_guard',1),
    'articles_soft_modifiers':('deletion_original',2),
    'articles_modifiers':('deletion_original',3),
    'modifiers_only':('deletion_modifiers_only',3),
    'percent_only':('number_style','percent_only'),
    'numbers_percent':('number_style','numbers_percent'),
    'numbers_percent_guarded':('number_style_guarded','numbers_percent_guarded'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--mode',choices=MODES,default='articles_guarded')
    parser.add_argument('--nltk-data',type=Path)
    args=parser.parse_args()
    saved=json.loads(args.summary.read_text());original=saved.get('summary',saved)
    source=args.source.read_text();index=SourceIndex(source)
    module,level=MODES[args.mode];editor=importlib.import_module(module)
    numeric=args.mode in {'percent_only','numbers_percent','numbers_percent_guarded'}
    if args.nltk_data:
        import nltk
        nltk.data.path.insert(0,str(args.nltk_data))
    editor.init();start=time.process_time();changes={};traces=[]
    for text,path,category in fragments(original):
        if category!='narrative':continue
        after,edits=editor.repair(text,index,level)
        if numeric:assert editor.verify_trace(text,after,edits)
        if text!=after:
            changes[path,text]=after
            traces.append(dict(path=path,before=text,after=after,edits=edits))
    baseline=narrative_context(original);rolled_back=False;error=None
    try:
        summary=patch(original,changes);context=narrative_context(summary)
        if numeric:
            assert editor.invariant_non_numeric(baseline,context)
        else:
            assert immutable(baseline,context),'Scientific guard changed'
            assert check(json.dumps(original,ensure_ascii=False),json.dumps(summary,ensure_ascii=False))['passed']
        protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
        assert protected(original)==protected(summary),'Evidence or metadata changed'
        try:validate_summary(json.dumps(original,ensure_ascii=False),source);valid=True
        except (ValueError,TypeError,KeyError):valid=False
        if valid:validate_summary(json.dumps(summary,ensure_ascii=False),source)
        audit=audit_summary(source,summary,index=index)
        before=audit_summary(source,original,index=index,expanded_diagnostic=False)
        assert audit['categories']['narrative']['longest_contiguous_match_words']<=before['categories']['narrative']['longest_contiguous_match_words']
    except (AssertionError,ValueError,TypeError,KeyError) as exc:
        summary=original;context=baseline;rolled_back=True;error=repr(exc)
        audit=audit_summary(source,summary,index=index)
    output=dict(summary=summary,judge_context=context,mode=args.mode,source_sha256=index.sha256,
        audit=audit,rolled_back=rolled_back,error=error,traces=traces,cpu_seconds_excluding_setup=time.process_time()-start)
    if numeric:output['numeric_preservation']='Exact supported Decimal value/sign/precision roundtrip per changed span; protected formula and nonnumeric text. The old digit-only signature is deliberately replaced for digit-word changes.'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
