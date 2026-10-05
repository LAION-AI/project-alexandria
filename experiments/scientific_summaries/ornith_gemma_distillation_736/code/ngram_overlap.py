"""Exact normalized word-overlap audit against the complete paper source.

Never uses MCQs or gold answers and never filters evaluation papers.
Five words are allowed; six are borderline; seven or more are flagged.
"""
import hashlib
import re
import unicodedata
from collections import defaultdict

VERSION='2.1'
WORD=re.compile(r"[^\W_]+(?:['’][^\W_]+)*",re.UNICODE)
FIELDS=('title','authors','field_subfield','type_of_paper','executive_summary',
        'research_context','research_question_and_hypothesis','methodological_details',
        'procedures_and_architectures','key_results','interpretation_and_theoretical_implications',
        'contradictions_and_limitations','claims','data_and_code_availability',
        'robustness_and_ablation_notes','ethical_considerations','key_figures_tables',
        'top_influential_citations','three_takeaways')
STRUCTURAL=set(FIELDS)|{'description','supporting_evidence','contradicting_evidence','implications',
    'narrative','statement','text','claim','summary','evidence','quote','quotes','source','source_quote',
    'evidence_quote','citation','justification','reason','reference','references','citations','authors',
    'page','pages','section','figure','table','value','finding','result','method','limitation','takeaway',
    'title','author','doi','year','url','content','explanation','support','contradiction','proof'}
EVIDENCE={'evidence','quote','quotes','source','source_quote','evidence_quote','proof'}
METADATA={'title','authors','field_subfield','type_of_paper','author','doi','year','url','reference','references','citation','citations'}


def normalize(text):
    # Compatibility accents such as U+02DC can expand to a space plus a
    # combining mark. Preserve original word boundaries during normalization.
    def compatibility(match):
        char=match.group();value=unicodedata.normalize('NFKC',char)
        return value if char.isspace() else ''.join(c for c in value if not c.isspace())
    text=re.sub(r'[^\x00-\x7f]',compatibility,text)
    text=unicodedata.normalize('NFKC',text).replace('\u00ad','')
    text=re.sub(r'(?<=\w)[-‐‑]\s*\n\s*(?=\w)','',text)
    return text.casefold().replace('’',"'")


def tokenize(text,expanded=False):
    normalized=normalize(text)
    if expanded:
        spans=[(m.start(),m.end()) for m in WORD.finditer(normalized)]
        return normalized,[normalized[a:b] for a,b in spans],spans
    words=[];spans=[]
    # Match the existing <=5-word quote rule: whitespace-delimited words.
    # Strip punctuation inside each word without splitting chemical terms,
    # decimal numbers or mathematical expressions into artificial extra words.
    for match in re.finditer(r'\S+',normalized):
        word=''.join(c for c in match.group() if c.isalnum())
        if word:words.append(word);spans.append((match.start(),match.end()))
    return normalized,words,spans


def merged_length(intervals):
    end=0;total=0
    for a,b in sorted(intervals):
        if b>end:total+=b-max(a,end);end=b
    return total


class SourceIndex:
    def __init__(self,text,expanded=False):
        self.sha256=hashlib.sha256(text.encode()).hexdigest()
        self.source_text=text;self.expanded=expanded;self.expanded_index=None
        self.normalized,self.words,self.positions=tokenize(text,expanded=expanded)
        self.five=defaultdict(list)
        for i in range(len(self.words)-4):self.five[tuple(self.words[i:i+5])].append(i)
        self.short={}

    def grams(self,n):
        if n not in self.short:self.short[n]={tuple(self.words[i:i+n]) for i in range(len(self.words)-n+1)}
        return self.short[n]

    def fragment(self,text,path,category):
        normalized,words,positions=tokenize(text,expanded=self.expanded);runs=[]
        for i in range(len(words)-4):
            for j in self.five.get(tuple(words[i:i+5]),()):
                # A maximal aligned run is extended only once, avoiding quadratic
                # work for long copied paragraphs. Hash hits are exact tuple matches.
                if i and j and words[i-1]==self.words[j-1]:continue
                length=5
                while i+length<len(words) and j+length<len(self.words) and words[i+length]==self.words[j+length]:length+=1
                runs.append((i,i+length,j,j+length))
        longest=max((b-a for a,b,_,_ in runs),default=0)
        if not longest:
            for n in range(min(4,len(words)),0,-1):
                if any(tuple(words[i:i+n]) in self.grams(n) for i in range(len(words)-n+1)):
                    longest=n;break
        coverage={str(n):merged_length([(a,b) for a,b,_,_ in runs if b-a>=n]) for n in [5,6,7]}
        coverage['3']=merged_length([(i,i+3) for i in range(len(words)-2) if tuple(words[i:i+3]) in self.grams(3)])
        # One canonical source occurrence per identical target span.
        unique={}
        for a,b,j,k in runs:
            unique.setdefault((a,b),(j,k))
        examples=[]
        for (a,b),(j,k) in sorted(unique.items(),key=lambda x:(-(x[0][1]-x[0][0]),x[0][0])):
            if b-a<6:continue
            examples.append(dict(words=b-a,summary_word_start=a,summary_word_end=b,
                source_word_start=j,source_word_end=k,
                source_normalized_char_start=self.positions[j][0],source_normalized_char_end=self.positions[k-1][1],
                summary_normalized_char_start=positions[a][0],summary_normalized_char_end=positions[b-1][1],
                excerpt=normalized[positions[a][0]:positions[min(b,a+30)-1][1]],excerpt_clipped=b-a>30))
        return dict(path=path,category=category,words=len(words),longest_contiguous_match_words=longest,
            covered_words_by_minimum_run=coverage,matching_runs_6plus=len(examples),matching_runs_7plus=sum(x['words']>=7 for x in examples),
            longest_examples=examples[:12],evidence_length_over_five=category=='evidence' and len(text.split())>5)


def fragments(summary,context=None):
    output=[]
    def add(text,path,category):
        if isinstance(text,str) and text.strip():output.append((text,path,category))
    def walk(value,path,category):
        if isinstance(value,str):add(value,path,category)
        elif isinstance(value,list):
            for i,item in enumerate(value):walk(item,f'{path}[{i}]',category)
        elif isinstance(value,dict):
            for key,item in value.items():
                if key in EVIDENCE:walk(item,path+'.'+key,'evidence')
                elif key in METADATA:walk(item,path+'.'+key,'metadata')
                elif key in STRUCTURAL:walk(item,path+'.'+key,category)
                else:
                    # Schema-v4 stores prose as a dictionary key and the quote as
                    # its value. Preserve malformed nested values without omission.
                    add(key,path+'.<statement>',category)
                    walk(item,path+'.<evidence>','evidence' if category!='metadata' else 'metadata')
    if isinstance(summary,dict):
        for key,value in summary.items():
            if key=='top_influential_citations':
                if not isinstance(value,list):
                    walk(value,key,'metadata')
                    continue
                for i,citation in enumerate(value if isinstance(value,list) else []):
                    if isinstance(citation,dict):
                        for k,v in citation.items():
                            if k=='quotes':walk(v,f'{key}[{i}].quotes','evidence')
                            elif k in STRUCTURAL:walk(v,f'{key}[{i}].{k}','metadata' if k in METADATA else 'narrative')
                            else:add(k,f'{key}[{i}].<reference>','metadata');walk(v,f'{key}[{i}].<justification>','narrative')
                continue
            walk(value,key,'metadata' if key in METADATA else 'narrative')
    elif isinstance(summary,str):add(summary,'summary','narrative')
    elif context:add(context,'judge_context','narrative')
    if not output and context:add(context,'judge_context','narrative')
    return output


def audit_summary(source,summary,context=None,index=None,expanded_diagnostic=True):
    index=index or SourceIndex(source)
    assert index.sha256==hashlib.sha256(source.encode()).hexdigest()
    rows=[index.fragment(text,path,category) for text,path,category in fragments(summary,context)]
    categories={}
    for category in ['narrative','evidence','metadata']:
        selected=[r for r in rows if r['category']==category]
        words=sum(r['words'] for r in selected);longest=max((r['longest_contiguous_match_words'] for r in selected),default=0)
        covered={n:sum(r['covered_words_by_minimum_run'][n] for r in selected) for n in ['3','5','6','7']}
        categories[category]=dict(words=words,longest_contiguous_match_words=longest,
            classification='violation_7plus' if longest>=7 else 'borderline_6' if longest==6 else 'within_5',
            strict_five_word_limit_passed=longest<=5,six_word_tolerance_passed=longest<=6,
            covered_words_by_minimum_run=covered,covered_word_fraction_by_minimum_run={n:v/words if words else 0 for n,v in covered.items()},
            evidence_fragments_over_five_words=sum(r['evidence_length_over_five'] for r in selected))
    substantive=max(categories[c]['longest_contiguous_match_words'] for c in ['narrative','evidence'])
    all_fields=max(c['longest_contiguous_match_words'] for c in categories.values())
    result=dict(audit_version=VERSION,source_sha256=index.sha256,source_words=len(index.words),
        normalization='Unicode NFKC; casefold; whitespace-delimited words with punctuation stripped within words; punctuation-only tokens excluded; soft-hyphens removed; PDF line-end word hyphenation joined',
        allowed_consecutive_words=5,borderline_consecutive_words=6,violation_from_consecutive_words=7,
        substantive_classification='violation_7plus' if substantive>=7 else 'borderline_6' if substantive==6 else 'within_5',
        all_fields_classification='violation_7plus' if all_fields>=7 else 'borderline_6' if all_fields==6 else 'within_5',
        all_fields_strict_five_word_limit_passed=all_fields<=5,all_fields_six_word_tolerance_passed=all_fields<=6,
        categories=categories,fragments=rows,metadata_reported_separately=True,
        no_cross_field_or_cross_fragment_matches=True,lexical_overlap_does_not_establish_plagiarism=True,
        generation_or_qa_filtering=False)
    if expanded_diagnostic:
        if index.expanded_index is None:index.expanded_index=SourceIndex(source,expanded=True)
        diagnostic=audit_summary(source,summary,context,index=index.expanded_index,expanded_diagnostic=False)
        result['expanded_word_diagnostic']=dict(
            normalization='Separate punctuation-split Unicode alphanumeric tokens; numeric/formula components and hyphenated terms may become multiple tokens',
            categories=diagnostic['categories'],substantive_classification=diagnostic['substantive_classification'],
            interpretation='Conservative punctuation-robust diagnostic; not the five-whitespace-word quote-limit decision')
    return result
