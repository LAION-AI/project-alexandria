"""Bounded function-word insertions with exact scientific-content preservation.

All candidates are strict character insertions, with explicit rule provenance.
Matching is against the original full source. Original-word coverage must fall;
merely increasing summary length never qualifies as an improvement.
"""
import re,math
from collections import Counter
from common import ROOT,named_values,negations
from critical_values import signature
from ngram_overlap import tokenize
import number_style as numeric
import grammar_units as grammar

REPORTING=grammar.REPORTING
METHOD_PARTICIPLES=set('measured estimated computed calculated determined evaluated assessed tested analyzed analysed collected obtained generated extracted detected identified quantified synthesized prepared amplified calibrated fitted simulated modeled modelled optimized solved filtered mapped constructed recorded produced validated processed isolated labeled labelled sequenced'.split())
RELATIVE_HEADS='method|methods|model|models|procedure|procedures|data|samples|measurements|results|approach|approaches|algorithm|algorithms|dataset|datasets'
RELATIVE_VERBS='used|developed|tested|analyzed|analysed|measured|obtained|collected|reported|evaluated|proposed|implemented|observed|described'
DOCUMENT_NOUNS='study|paper|work|analysis|experiment|method|approach|model|dataset|procedure|result|studies|papers|analyses|experiments|methods|approaches|models|datasets|procedures|results'
FORBIDDEN=set('maybe perhaps possibly probably clearly certainly definitely indeed generally typically usually often always never only merely especially notably significantly substantially partly primarily mainly basically essentially'.split())


def temperature_scope_safe(text):
    # Expanding a shared preposition into a temperature noun phrase must not
    # capture a coordinated pressure/field/etc.: at400K and1000atm.
    value=r'[-−]?\d+(?:\.\d+)?'
    other_unit=r'(?:atm|bar|mbar|torr|psi|Pa|kPa|MPa|GPa|T|mT|Hz|kHz|MHz|GHz|V|mV|A|mA|W|kW|mol|mmol|M|mM|µM|μM)'
    pattern=r'\bat a temperature of[ \t]+'+value+r'[ \t]*(?:°[CF]|K)[ \t]*(?:,|and|or)[ \t]*'+value+r'[ \t]*'+other_unit+r'(?!\w)'
    return not re.search(pattern,text)


def init():
    global nltk,splitter,word_tokenizer
    grammar.init();nltk=grammar.nltk;splitter=grammar.splitter
    word_tokenizer=nltk.tokenize.TreebankWordTokenizer()


def guard(before,after):
    if before==after:return True
    return (signature(before)==signature(after)
            and named_values(before)==named_values(after)
            and negations(before)==negations(after)
            and Counter(numeric.MODALS.findall(before.lower()))==Counter(numeric.MODALS.findall(after.lower())))


def spans_and_words(text):
    spans=[];words=[]
    for m in re.finditer(r'\S+',text):
        values=tokenize(m.group())[1]
        if len(values)>1:return None
        if values:spans.append(m.span());words.append(values[0])
    if words!=tokenize(text)[1]:return None
    return spans,words


def covered_positions(words,index):
    grams=index.grams(6);hits=[];covered=set()
    for i in range(len(words)-5):
        if tuple(words[i:i+6]) in grams:
            hits.append(i);covered.update(range(i,i+6))
    return hits,covered


def candidates(text,mode):
    result=[];protected=numeric.protected_spans(text)
    def add(position,insertion,rule):
        if any(a<position<b for a,b in protected):return
        assert not set(re.findall(r'\b\w+\b',insertion.casefold())) & FORBIDDEN
        result.append(dict(start=position,end=position,replacement=insertion,rule=rule))
    # Insert a complementizer only if the already tested optional-that grammar
    # recognizes it as an explicit-subject finite reporting clause.
    for m in re.finditer(r'\b('+ '|'.join(sorted(REPORTING,key=lambda s:(-len(s),s)))+r')[ \t]+',text,re.I):
        p=m.end()
        if re.match(r'that\b',text[p:]):continue
        left_words=word_tokenizer.tokenize(text[max(0,m.start()-100):m.start()])+[m.group(1)]
        tagged=nltk.pos_tag(left_words+word_tokenizer.tokenize(text[p:p+150])[:18])
        if tagged[len(left_words)-1][1] not in {'VBD','VBP','VBZ','VBN','VB'}:continue
        proposed=text[:p]+'that '+text[p:]
        if any(c['start']==p and c['end']==p+5 for c in grammar.optional_that_candidates(proposed)):
            add(p,'that ','reporting_complementizer_that')
    # A narrow object-relative pattern: noun + subject + transitive verb with
    # no second object. The omitted relative marker can be restored.
    pattern=r'\b(?:'+RELATIVE_HEADS+r')[ \t]+(?P<subject>we|they|the authors)[ \t]+(?:'+RELATIVE_VERBS+r')(?=[ \t]*(?:[,.;:]|$|\b(?:in|for|with|from|during|previously|here)\b))'
    for m in re.finditer(pattern,text):add(m.start('subject'),'that ','object_relative_that')
    # Require a noun phrase; exclude relative 'all that is...' and idiomatic
    # degree/connective forms 'all the more', 'all the same', 'all the while'.
    for m in re.finditer(r'\b(?:all|half)[ \t]+(?P<det>the|these|those)\b',text,re.I):
        following=word_tokenizer.tokenize(text[m.end():])[:3]
        if not following or following[0].lower() in {'more','less','better','worse','same','while','way'}:continue
        if nltk.pos_tag(following)[0][1] not in {'NN','NNS','NNP','NNPS','JJ','CD'}:continue
        add(m.start('det'),'of ','quantifier_of')
    # Only a directly preceding methodological past participle is accepted;
    # avoid NP-attached 'patients using...' and prepositions like 'without'.
    for m in re.finditer(r'\b(?P<verb>'+ '|'.join(sorted(METHOD_PARTICIPLES))+r')[ \t]+(?P<using>using)\b',text):
        add(m.start('using'),'by ','method_by_using')
    # Unambiguous purpose in an explicit passive instrument-use construction.
    for m in re.finditer(r'\b(?:was|were|is|are|be|been|being)[ \t]+used[ \t]+(?P<to>to)[ \t]+(?P<verb>[a-z]+)\b',text,re.I):
        if m.group('to')!='to':continue
        word=m.group('verb')
        if nltk.pos_tag(['to',word])[1][1]=='VB':add(m.start('to'),'in order ','purpose_in_order')
    # Explicit durations, leaving number spelling, sign and precision intact.
    number_pattern=r'(?:\d+(?:\.\d+)?|'+ '|'.join(numeric.SMALL)+r'|'+ '|'.join(numeric.TENS[2:])+r')'
    for m in re.finditer(r'\b(?:for|over|after|within)[ \t]+(?P<value>'+number_pattern+r')[ \t]+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b',text,re.I):
        add(m.start('value'),'a period of ','explicit_duration_period')
    for m in re.finditer(r'\bat[ \t]+(?P<value>[-−]?\d+(?:\.\d+)?)[ \t]*(?:°[CF]|K)(?!\w)',text,re.I):
        # K must remain uppercase (kelvin); lowercase k may be another symbol.
        if m.group()[-1]=='k':continue
        add(m.start('value'),'a temperature of ','explicit_temperature')
    for m in re.finditer(r'\bin[ \t]+(?P<year>(?:19|20)\d{2})(?![\w–—./-])',text,re.I):
        # A following cohort noun ('in 2000 patients') is not a calendar year.
        tail=text[m.end():]
        if not re.match(r'[ \t]*(?:[,.;:)]|$|\b(?:from|during|after|before|when|where)\b)',tail,re.I):continue
        add(m.start('year'),'the year ','explicit_calendar_year')
    if mode=='expanded_insert':
        # Deictic 'this/these' already selects the referent. 'Particular' adds
        # emphasis; its scientific-content neutrality is separately reviewed.
        for m in re.finditer(r'\b(?:this|these)[ \t]+(?P<noun>'+DOCUMENT_NOUNS+r')\b',text,re.I):
            add(m.start('noun'),'particular ','deictic_particular')
        # The existing 'such as' already explicitly marks exemplification.
        for m in re.finditer(r'\bsuch as(?=[ \t]+[A-Za-z0-9])',text):
            following=word_tokenizer.tokenize(text[m.end():])[:4]
            if not following or nltk.pos_tag(following)[0][1] not in {'NN','NNS','NNP','NNPS','JJ','DT','CD'}:continue
            if not re.match(r'[ \t]+(?:for example|for instance)',text[m.end():]):
                add(m.end(),', for example,','redundant_example_cue')
    return result


def repair(text,index,mode):
    original=text;parsed=spans_and_words(text)
    if not parsed:return text,[]
    original_words=parsed[1];origins=list(range(len(original_words)));edits=[]
    budget=min(8,max(1,math.ceil(len(original_words)/50)))
    for _ in range(budget):
        parsed=spans_and_words(text)
        if not parsed:break
        spans,words=parsed;hits,covered=covered_positions(words,index)
        if not hits:break
        old_original={origins[i] for i in covered if origins[i] is not None}
        matches=[(spans[i][0],spans[i+5][1]) for i in hits];options=[]
        for c in candidates(text,mode):
            p=c['start']
            # Insert strictly inside a currently copied span. An append at its
            # edge cannot break the existing original six-word sequence.
            if not any(a<p<b for a,b in matches):continue
            if c['rule'] in {'deictic_particular','redundant_example_cue'} and any(e['rule']==c['rule'] for e in edits):continue
            after=text[:p]+c['replacement']+text[p:]
            new_words=tokenize(after)[1];inserted=tokenize(c['replacement'])[1]
            k=len(tokenize(text[:p])[1])
            if new_words!=words[:k]+inserted+words[k:]:continue
            if len(new_words)>max(len(original_words)+5,int(len(original_words)*1.08)):continue
            new_origins=origins[:k]+[None]*len(inserted)+origins[k:]
            new_hits,new_covered=covered_positions(new_words,index)
            new_original={new_origins[i] for i in new_covered if new_origins[i] is not None}
            # All three constraints matter. A larger denominator alone, or
            # breaking a run only among previously inserted words, is rejected.
            if len(new_original)>=len(old_original) or len(new_covered)>=len(covered) or len(new_hits)>=len(hits):continue
            if len(new_covered)/len(new_words)>=len(covered)/len(words):continue
            if not guard(original,after):continue
            options.append(dict(c,before=text,after=after,origins_after=new_origins,
                                inserted_words=len(inserted),covered_before=len(covered),covered_after=len(new_covered),
                                original_covered_before=len(old_original),original_covered_after=len(new_original),
                                gain=len(old_original)-len(new_original)))
        if not options:break
        # Prefer copied-original-word gain per inserted word, then total gain.
        best=min(options,key=lambda e:(-e['gain']/e['inserted_words'],-e['gain'],e['inserted_words'],e['start']))
        origins=best.pop('origins_after');text=best['after'];edits.append(best)
    assert guard(original,text)
    assert [tokenize(text)[1][i] for i,x in enumerate(origins) if x is not None]==original_words
    return text,edits


def verify_trace(before,after,edits):
    current=before
    for e in edits:
        p=e['start'];assert e['end']==p and e['before']==current
        assert current[:p]+e['replacement']+current[p:]==e['after']
        assert any(c['start']==p and c['replacement']==e['replacement'] and c['rule']==e['rule'] for c in candidates(current,'expanded_insert'))
        assert e['covered_after']<e['covered_before'] and e['original_covered_after']<e['original_covered_before']
        assert guard(current,e['after'])
        current=e['after']
    assert current==after
    return True
