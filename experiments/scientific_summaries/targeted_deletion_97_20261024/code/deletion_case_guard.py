"""Targeted word deletion. Finite candidates, immutable scientific-value guards.

No QA, model scores, or gold answers are consulted by deployment.
"""
import re
from collections import Counter
from common import ROOT
from ngram_overlap import tokenize, merged_length
from lexicon import immutable
from critical_values import LATEX
from procedural import PROTECTED, COLLOCATIONS

FILLERS = [
    ('framing', r'\b(?:in this paper|in this study|in this work|in this article)\s*,\s*'),
    ('note', r'\bit is (?:worth noting|interesting to note) that\s+'),
    ('back_reference', r'\bas (?:noted|mentioned|described|discussed) (?:above|earlier)\s*,\s*'),
    ('discourse', r'\b(?:indeed|interestingly|notably)\s*,\s*'),
    ('in_fact', r'\bin fact\s*,\s*'),
]
SOFT = set('interesting notable remarkable detailed comprehensive useful helpful straightforward'.split())
SCOPE = set('not never no without neither nor only also even still already yet just almost nearly approximately about roughly exactly mostly mainly primarily partially partly predominantly typically generally usually often sometimes always rarely seldom possibly potentially probably certainly apparently seemingly relatively comparatively significantly statistically substantially considerably slightly marginally fully completely necessarily sufficiently locally globally simultaneously respectively individually separately independently directly indirectly causally positively negatively inversely however thus therefore hence then otherwise consequently instead nevertheless first last next other same former latter each every any some most least many few more less accurately efficiently effectively particularly especially truly very highly enough too rather well together'.split())
PROTECTED_WORDS = PROTECTED | SCOPE | {w for p in COLLOCATIONS for w in p.split()}


def init():
    global nltk, treebank, splitter
    import nltk as module
    nltk = module
    nltk.data.path.insert(0, str(ROOT/'resources/nltk_data'))
    treebank = nltk.tokenize.TreebankWordTokenizer()
    splitter = nltk.tokenize.PunktTokenizer('english')
    # Load/tagger once in the process initializer.
    nltk.pos_tag(['The', 'method', 'worked', '.'])


def state(text, index):
    words = tokenize(text)[1]
    six = index.grams(6)
    hits = [i for i in range(len(words)-5) if tuple(words[i:i+6]) in six]
    return words, hits, merged_length((i,i+6) for i in hits)


def spans(text, level):
    result = []
    sentence_starts = {a for a,b in splitter.span_tokenize(text)}
    for rule, pattern in FILLERS:
        for m in re.finditer(pattern, text, re.I):
            # Restrict discourse/metatext removal to sentence starts.
            if m.start() in sentence_starts:
                result.append((m.start(),m.end(),'filler_'+rule,0,'phrase'))
    if level == 0:
        return result
    positions = list(treebank.span_tokenize(text))
    tagged = nltk.pos_tag([text[a:b] for a,b in positions])
    for k,((a,b),(word,tag)) in enumerate(zip(positions,tagged)):
        lower = word.casefold()
        if not re.fullmatch('[A-Za-z]+', word):
            continue
        if tag == 'DT' and lower in {'a','an','the'}:
            if word != lower and a not in sentence_starts:continue
            if k+1>=len(tagged) or not tagged[k+1][1].startswith(('NN','JJ')):continue
            # Established Latin expressions and capitalized proper-name noun
            # phrases are left intact; a/an/the remain an experimental arm.
            if re.match(r'\s+(?:priori|posteriori)\b',text[b:],re.I):continue
            if lower == 'the' and re.match(r'\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)',text[b:]):continue
            result.append((a,b,'article_'+lower,1,tag))
        elif level >= 2 and tag == 'JJ' and lower in SOFT:
            result.append((a,b,'soft_modifier_'+lower,2,tag))
        elif level >= 3 and tag in {'JJ','JJR','JJS','RB','RBR','RBS'} and lower not in PROTECTED_WORDS:
            # No comparative/superlative deletion: these express a relation.
            if tag not in {'JJ','RB'}:continue
            if word[0].isupper() and a not in sentence_starts:continue
            result.append((a,b,'pos_'+tag,3,tag))
    return result


def remove(text, a, b):
    starts = {x for x,y in splitter.span_tokenize(text)}
    initial = a in starts and text[a:a+1].isupper()
    # Remove following horizontal whitespace while preserving punctuation,
    # paragraph breaks, and every remaining lexical token.
    end = b
    while end < len(text) and text[end] in ' \t':end += 1
    if end == b and a and text[a-1] in ' \t':a -= 1
    tail = text[end:]
    if initial:
        match = re.match(r'([a-z]{2,})\b',tail)
        if match:tail = tail[:1].upper()+tail[1:]
    return (text[:a]+tail).strip()


def repair(text, index, level):
    original = text
    initial_words,_,_ = state(original,index)
    budget = max(1,int(len(initial_words)*.15))
    edits=[]
    for _ in range(64):
        words,hits,covered=state(text,index)
        if not hits:break
        raw=[]
        for m in re.finditer(r'\S+',text):
            values=tokenize(m.group())[1]
            if len(values)>1:return text,edits
            if values:raw.append((m.start(),m.end(),values[0]))
        if [r[2] for r in raw]!=words:return text,edits
        match_spans=[(raw[i][0],raw[i+5][1]) for i in hits]
        protected=[m.span() for m in LATEX.finditer(text)]
        protected += [m.span() for m in re.finditer(r'`[^`]*`|"[^"\n]*"|“[^”\n]*”',text)]
        candidates=[]
        for a,b,rule,priority,pos in spans(text,level):
            if not any(a<d and b>c for c,d in match_spans):continue
            if any(a<d and b>c for c,d in protected):continue
            after=remove(text,a,b)
            new_words,new_hits,new_covered=state(after,index)
            deleted=len(initial_words)-len(new_words)
            if not new_words or deleted>budget or deleted<=0:continue
            if len(new_hits)>=len(hits) or new_covered>=covered:continue
            if new_covered/len(new_words)>=covered/len(words):continue
            if not immutable(original,after):continue
            if tokenize(text,expanded=True)[1]==tokenize(after,expanded=True)[1]:continue
            candidates.append(dict(rule=rule,pos=pos,start=a,end=b,deleted=text[a:b],before=text,after=after,
                covered_before=covered,covered_after=new_covered,words_before=len(words),words_after=len(new_words),
                gain=covered-new_covered,hit_gain=len(hits)-len(new_hits),priority=priority))
        if not candidates:break
        best=min(candidates,key=lambda c:(-c['gain'],c['priority'],-c['hit_gain'],c['words_before']-c['words_after'],c['start']))
        edits.append(best);text=best['after']
    return text,edits
