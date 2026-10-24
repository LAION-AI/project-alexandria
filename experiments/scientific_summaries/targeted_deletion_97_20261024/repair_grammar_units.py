"""Apply the measured identifier-safe numeric/grammar/unit CPU pipeline.

Input for reproducing the experiment is a saved protected-article summary.
The initial Gemma repair, phrase bank and article stage are upstream.
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
import number_style_identifiers as numeric
import grammar_units as grammar
from ngram_overlap import SourceIndex,fragments,audit_summary
from passages import patch
from reference_validator import narrative_context,validate_summary


def apply_stage(summary,index,editor,mode):
    changes={};traces=[]
    for text,path,kind in fragments(summary):
        if kind!='narrative':continue
        after,edits=editor.repair(text,index,mode)
        assert editor.verify_trace(text,after,edits)
        if text!=after:
            changes[path,text]=after
            traces.append(dict(path=path,before=text,after=after,edits=edits))
    return patch(summary,changes),traces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--mode',choices=['identifier_guard','optional_that','unit_aliases','combined'],default='combined')
    parser.add_argument('--nltk-data',type=Path)
    args=parser.parse_args()
    if args.nltk_data:
        import nltk
        nltk.data.path.insert(0,str(args.nltk_data))
    grammar.init()
    source=args.source.read_text();index=SourceIndex(source)
    saved=json.loads(args.summary.read_text());original=saved.get('summary',saved)
    start=time.process_time()
    intermediate,numeric_traces=apply_stage(original,index,numeric,'numbers_percent_guarded')
    assert numeric.invariant_non_numeric(narrative_context(original),narrative_context(intermediate))
    summary=intermediate;grammar_traces=[]
    if args.mode!='identifier_guard':summary,grammar_traces=apply_stage(intermediate,index,grammar,args.mode)
    context=narrative_context(summary)
    assert grammar.invariant(narrative_context(intermediate),context)
    protected=lambda s:[(t,p,k) for t,p,k in fragments(s) if k!='narrative']
    assert protected(original)==protected(summary)
    try:validate_summary(json.dumps(original,ensure_ascii=False),source);valid=True
    except (ValueError,TypeError,KeyError):valid=False
    if valid:validate_summary(json.dumps(summary,ensure_ascii=False),source)
    audit=audit_summary(source,summary,index=index)
    baseline=audit_summary(source,original,index=index,expanded_diagnostic=False)
    assert audit['categories']['narrative']['longest_contiguous_match_words']<=baseline['categories']['narrative']['longest_contiguous_match_words']
    result=dict(summary=summary,judge_context=context,mode=args.mode,source_sha256=index.sha256,
                audit=audit,numeric_traces=numeric_traces,grammar_unit_traces=grammar_traces,
                rolled_back=False,cpu_seconds_excluding_setup=time.process_time()-start,
                input_scope='Starts after protected-article stage; numeric stage is recomputed with identifier protection.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
