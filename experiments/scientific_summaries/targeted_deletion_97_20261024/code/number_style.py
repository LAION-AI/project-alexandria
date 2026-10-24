"""Controlled exact-value spelling changes outside mathematical expressions."""
import re
from decimal import Decimal,InvalidOperation
from collections import Counter
from common import ROOT,named_values,negations
from critical_values import LATEX,RELATION,OPERATION,FUNCTION,signature
from ngram_overlap import tokenize,merged_length

SMALL='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
TENS='zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
SCALES={'billion':10**9,'million':10**6,'thousand':10**3}
VALUES={w:i for i,w in enumerate(SMALL)}|{w:i*10 for i,w in enumerate(TENS) if i>=2}
NUMWORDS=set(VALUES)|set(SCALES)|{'hundred','and','point'}
LEX='|'.join(sorted(NUMWORDS,key=lambda x:(-len(x),x)))
WORD_NUMBER=re.compile(r'(?<![\w-])(?:minus\s+)?(?:'+LEX+r')(?:(?:[ \t-]+|,[ \t]*)(?:'+LEX+r'))*(?![\w-])',re.I)
DIGIT_NUMBER=re.compile(r'(?<![\w.+-])[-−]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?!\w|\.\d)')
PERCENT=re.compile(r'[ \t]*(?:%|percent\b|per[ \t]+cent\b)',re.I)
MODALS=re.compile(r'\b(?:may|might|can|cannot|could|must|should|shall|will|would)\b',re.I)


def init():
    global nltk,splitter
    import nltk as module
    nltk=module;nltk.data.path.insert(0,str(ROOT/'resources/nltk_data'))
    splitter=nltk.tokenize.PunktTokenizer('english');nltk.pos_tag(['The','sample','.'])


def spell_integer(n):
    if n<20:return SMALL[n]
    if n<100:return TENS[n//10]+('-'+SMALL[n%10] if n%10 else '')
    if n<1000:return SMALL[n//100]+' hundred'+(' and '+spell_integer(n%100) if n%100 else '')
    for word,scale in SCALES.items():
        if n>=scale:return spell_integer(n//scale)+' '+word+(' '+spell_integer(n%scale) if n%scale else '')
    raise ValueError('Unsupported integer')


def spell_decimal(value):
    if not value.is_finite() or abs(value)>=10**12 or value.as_tuple().exponent < -6:raise ValueError('Unsupported magnitude or precision')
    text=format(abs(value),'f');whole,_,fraction=text.partition('.')
    out=spell_integer(int(whole))
    if fraction:out+=' point '+' '.join(SMALL[int(c)] for c in fraction)
    if value.is_signed():out='minus '+out
    if len(out.split())>12:raise ValueError('Excessive spelling length')
    return out


def small_value(words):
    if not words:raise ValueError('Empty group')
    if len(words)==1 and words[0] in VALUES:return VALUES[words[0]]
    if len(words)==2 and words[0] in TENS[2:] and words[1] in SMALL[1:10]:return VALUES[words[0]]+VALUES[words[1]]
    if len(words)>=2 and words[0] in SMALL[1:10] and words[1]=='hundred':
        rest=words[2:]
        if rest[:1]==['and']:rest=rest[1:]
        return VALUES[words[0]]*100+(small_value(rest) if rest else 0)
    raise ValueError('Ambiguous English cardinal')


def parse_words(text):
    words=text.lower().replace('-',' ').replace(',',' ').split();negative=words[:1]==['minus']
    if negative:words=words[1:]
    fraction=None
    if 'point' in words:
        if words.count('point')!=1:raise ValueError('Multiple decimal points')
        p=words.index('point');fraction=words[p+1:];words=words[:p]
        if not fraction or any(w not in SMALL[:10] for w in fraction):raise ValueError('Ambiguous decimal digits')
    total=0;group=[];last=10**15
    for word in words:
        if word in SCALES:
            scale=SCALES[word]
            if scale>=last:raise ValueError('Invalid scale order')
            total+=small_value(group)*scale;group=[];last=scale
        else:group.append(word)
    if group[:1]==['and'] and total:group=group[1:]
    if group:total+=small_value(group)
    elif not total:raise ValueError('Empty cardinal')
    result=Decimal(('-' if negative else '')+str(total)+('.'+''.join(str(VALUES[w]) for w in fraction) if fraction else ''))
    if abs(result)>=10**12 or result.as_tuple().exponent < -6:raise ValueError('Unsupported number')
    return result


def same_number(a,b):
    return a==b and a.as_tuple().exponent==b.as_tuple().exponent and a.is_signed()==b.is_signed()


def protected_spans(text):
    spans=[m.span() for p in [LATEX,RELATION,OPERATION,FUNCTION] for m in p.finditer(text)]
    spans += [m.span() for m in re.finditer(r'\S+[ \t]*[=<>≤≥≠±≈∝×÷^–−][ \t]*\S+',text)]
    spans += [m.span() for m in re.finditer(r'`[^`]*`|"[^"\n]*"|“[^”\n]*”|\[[\d\s,;–−-]+\]|\([\d\s,;–−+*/%=.<>-]+\)',text)]
    return spans


def state(text,index):
    words=tokenize(text)[1];grams=index.grams(6)
    hits=[i for i in range(len(words)-5) if tuple(words[i:i+6]) in grams]
    return words,hits,merged_length((i,i+6) for i in hits)


def candidates(text,mode):
    protected=protected_spans(text);result=[]
    for kind,regex in [('digits',DIGIT_NUMBER),('words',WORD_NUMBER)]:
        for m in regex.finditer(text):
            a,b=m.span();surface=m.group();left=text[max(0,a-32):a];right=text[b:]
            if a and text[a-1] in '-−' or b<len(text) and text[b] in '-−':continue
            if re.search(r'\b(?:fig(?:ure)?|table|eq(?:uation)?|section|sec|appendix|chapter|version)\.?[ \t]*$',left,re.I):continue
            if kind=='words' and surface not in [surface.lower(),surface.lower().capitalize()]:continue
            try:value=Decimal(surface.replace(',','').replace('−','-')) if kind=='digits' else parse_words(surface)
            except (ValueError,InvalidOperation):continue
            if abs(value)>=10**12 or value.as_tuple().exponent < -6:continue
            if value==int(value) and 1900<=value<=2100 and re.search(r'\b(?:in|since|during|year|published|before|after)[ \t]*$',left,re.I):continue
            unit=PERCENT.match(right);unit_end=b+unit.end() if unit else b
            if any(a<d and unit_end>c for c,d in protected):continue
            if kind=='words' and not unit:
                nxt=re.match(r'[ \t]+([A-Za-z]+)',right)
                if not nxt or nltk.pos_tag([nxt.group(1)])[0][1] not in {'NN','NNS'}:continue
            if unit and kind=='digits':
                old_unit=text[b:unit_end]
                flipped=' percent' if '%' in old_unit else '%'
                result.append(dict(start=b,end=unit_end,replacement=flipped,rule='percent_symbol_to_word' if '%' in old_unit else 'percent_word_to_symbol',
                    number_before=str(value),number_after=str(value),decimal_places=-value.as_tuple().exponent,unit='percent',quantity_start=a,quantity_end=unit_end))
            if mode=='percent_only':continue
            try:
                replacement=spell_decimal(value) if kind=='digits' else format(value,'f')
                back=parse_words(replacement) if kind=='digits' else Decimal(replacement)
                if not same_number(value,back):continue
            except (ValueError,InvalidOperation):continue
            if a==0 and surface[:1].isupper():replacement=replacement[:1].upper()+replacement[1:]
            numeric_replacement=replacement
            if unit:replacement += ' percent' if kind=='digits' else '%'
            rule=('digits_to_words' if kind=='digits' else 'words_to_digits')+('_percent' if unit else '')
            result.append(dict(start=a,end=unit_end if unit else b,replacement=replacement,rule=rule,
                numeric_before_surface=surface,numeric_after_surface=numeric_replacement,
                number_before=str(value),number_after=str(back),decimal_places=-value.as_tuple().exponent,unit='percent' if unit else None,quantity_start=a,quantity_end=unit_end))
    return result


def invariant_non_numeric(before,after):
    if named_values(before)!=named_values(after) or negations(before)!=negations(after):return False
    if Counter(MODALS.findall(before.lower()))!=Counter(MODALS.findall(after.lower())):return False
    a=signature(before);b=signature(after)
    return all(a[k]==b[k] for k in ['formulas','mathematical_variables','operators'])


def repair(text,index,mode):
    original=text;edits=[];initial_words=len(tokenize(text)[1])
    for _ in range(64):
        words,hits,covered=state(text,index)
        if not hits:break
        raw=[]
        for m in re.finditer(r'\S+',text):
            value=tokenize(m.group())[1]
            if len(value)>1:return text,edits
            if value:raw.append((m.start(),m.end(),value[0]))
        if [x[2] for x in raw]!=words:return text,edits
        matches=[(raw[i][0],raw[i+5][1]) for i in hits];options=[]
        for c in candidates(text,mode):
            a,b=c['start'],c['end']
            if not any(c['quantity_start']<d and c['quantity_end']>e for e,d in matches):continue
            after=text[:a]+c['replacement']+text[b:]
            new_words,new_hits,new_covered=state(after,index)
            if not new_words or len(new_words)>max(initial_words+4,int(initial_words*1.2)):continue
            if new_covered>=covered or new_covered/len(new_words)>=covered/len(words) or len(new_hits)>=len(hits):continue
            if not invariant_non_numeric(original,after):continue
            # The proposed edit alters only the audited numeric/unit span. Its
            # value is checked with Decimal, including decimal-place precision.
            if after[:a]+text[a:b]+after[a+len(c['replacement']):]!=text:raise AssertionError('Edit provenance changed nonnumeric text')
            options.append(dict(c,before=text,after=after,old=text[a:b],covered_before=covered,covered_after=new_covered,gain=covered-new_covered))
        if not options:break
        best=min(options,key=lambda c:(-c['gain'],len(c['replacement']),c['start']))
        edits.append(best);text=best['after']
    assert invariant_non_numeric(original,text)
    return text,edits


def verify_trace(before,after,edits):
    current=before
    for e in edits:
        assert e['before']==current
        a,b=e['start'],e['end'];assert current[a:b]==e['old']
        assert current[:a]+e['replacement']+current[b:]==e['after']
        assert not any(a<d and b>c for c,d in protected_spans(current))
        if e['rule'].startswith('digits_to_words'):
            x=Decimal(e['numeric_before_surface'].replace(',','').replace('−','-'));y=parse_words(e['numeric_after_surface'])
            assert same_number(x,y)
            if e['unit']=='percent':assert e['replacement'].lower().endswith(' percent')
        elif e['rule'].startswith('words_to_digits'):
            assert same_number(parse_words(e['numeric_before_surface']),Decimal(e['numeric_after_surface']))
            if e['unit']=='percent':assert e['replacement'].endswith('%')
        else:
            units=lambda s:re.sub(r'\s+',' ',s.strip().lower())
            assert {units(e['old']),units(e['replacement'])} in [{'%', 'percent'},{'%','per cent'}]
        assert invariant_non_numeric(current,e['after'])
        current=e['after']
    assert current==after
    return True
