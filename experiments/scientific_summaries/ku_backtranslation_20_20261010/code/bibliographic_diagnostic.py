# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
"""Separate embedded bibliography in a supplemental, QA-independent copy diagnostic.

The historical all-factual-string audit is unchanged for matched comparisons.
This diagnostic reuses its existing per-fragment match counts; it does not
rewrite outputs, alter acceptance, inspect QA predictions or add source text.
"""
import copy
import re
import statistics
from common import ROOT, load, read_jsonl, write
from ku_pipeline import KU_CONDITIONS

BIB=re.compile(r'author|title|doi|citation|reference|bibliograph|publication_year',re.I)

def classified_strings(result):
    out=[];title=result.get('title','').strip().casefold()
    def strings(v,metadata=False):
        if isinstance(v,str):out.append((v,metadata or bool(title and v.strip().casefold()==title)))
        elif isinstance(v,dict):
            for k,x in v.items():
                bib=metadata or bool(BIB.search(str(k)))
                out.append((str(k),bib));strings(x,bib)
        elif isinstance(v,list):
            for x in v:strings(x,metadata)
        elif v is not None:out.append((str(v),metadata))
    for u in result['knowledge_units']:
        strings(u['context_summary'])
        for e in u['entities']:
            strings(e['name']);strings(e['entity_type']);strings(e.get('aliases',[]));strings(e['attributes'])
            for r in e['relationships']:strings(r['predicate']);strings(r['target']);strings(r['attributes'])
    return out

def categories(rows, category):
    fragments=[x for x in rows if x['category']==category]
    words=sum(x['words'] for x in fragments)
    return dict(words=words,longest_contiguous_match_words=max((x['longest_contiguous_match_words'] for x in fragments),default=0),
                covered_word_fraction_six_plus=sum(x['covered_words_by_minimum_run']['6'] for x in fragments)/max(1,words))

def main():
    summaries=[]
    for record in read_jsonl(ROOT/'outputs/copy_overlap_details.jsonl'):
        if record['condition'].removesuffix('_bt') not in KU_CONDITIONS or not record['audit']:continue
        value=load(ROOT/'outputs/generation'/record['condition']/(record['document_id']+'.json'))
        fields=classified_strings(value['result']);fragments=copy.deepcopy(record['audit']['fragments']);count=0
        for fragment in fragments:
            match=re.fullmatch(r'narrative\[(\d+)\]',fragment['path'])
            if match and fields[int(match.group(1))][1]:fragment['category']='metadata';count+=1
        summaries.append(dict(condition=record['condition'],document_id=record['document_id'],
                              source_sha256=record['source_sha256'],reclassified_fragments=count,
                              narrative=categories(fragments,'narrative'),metadata=categories(fragments,'metadata')))
    conditions={}
    for c in KU_CONDITIONS+[x+'_bt' for x in KU_CONDITIONS]:
        selected=[x for x in summaries if x['condition']==c]
        conditions[c]={}
        for category in ['narrative','metadata']:
            items=[r[category] for r in selected]
            conditions[c][category]=dict(audited_documents=len(items),
                strict_five_word_pass=sum(x['longest_contiguous_match_words']<=5 for x in items),
                borderline_six_only=sum(x['longest_contiguous_match_words']==6 for x in items),
                violation_seven_plus=sum(x['longest_contiguous_match_words']>=7 for x in items),
                maximum_contiguous_match_words=max(x['longest_contiguous_match_words'] for x in items),
                mean_covered_fraction_six_plus=statistics.mean(x['covered_word_fraction_six_plus'] for x in items))
    write(ROOT/'outputs/bibliographic_copy_diagnostic.json',dict(
          complete=True,scope='Supplemental bibliography-separated diagnostic; historical primary audit and QA unchanged',
          classification='Attribute keys matching author/title/doi/citation/reference/bibliography/publication_year and their descendant values; exact document-title strings embedded in names/targets. Document metadata remains metadata.',
          no_new_generation=True,no_qa_used=True,reuses_frozen_primary_per_fragment_matches=True,
          conditions=conditions,per_document=summaries))
    print('Bibliographic overlap separated for',len(summaries),'unchanged KU document versions.')

if __name__=='__main__':main()
