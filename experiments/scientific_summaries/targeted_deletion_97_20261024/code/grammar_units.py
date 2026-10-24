"""Conservative complementizer deletion and exact unit aliases, CPU only.

The small rule set is frozen before QA. A proposal must remove copied word
positions and six-gram hits; padding the denominator is insufficient. This is
notation/grammar editing, not a guarantee of scientific semantic equivalence.
"""
import re
from decimal import Decimal
from collections import Counter
import number_style_identifiers as numeric
from common import named_values,negations
from critical_values import signature,LATEX
from ngram_overlap import tokenize

REPORTING=set('show shows showed shown demonstrate demonstrates demonstrated indicate indicates indicated suggest suggests suggested report reports reported find finds found confirm confirms confirmed conclude concludes concluded reveal reveals revealed establish establishes established observe observes observed'.split())
# Case-sensitive SI symbols, with no magnitude conversion. Ambiguous M/m and
# temperature scales are excluded. Full names are recognized case-insensitively.
UNIT_ROWS=[
 ('mg','milligram'),('g','gram'),('kg','kilogram'),('µg','microgram'),('μg','microgram'),('ng','nanogram'),
 ('mm','millimeter'),('cm','centimeter'),('nm','nanometer'),('km','kilometer'),('µm','micrometer'),('μm','micrometer'),
 ('mL','milliliter'),('µL','microliter'),('μL','microliter'),('L','liter'),
 ('ms','millisecond'),('s','second'),('µs','microsecond'),('μs','microsecond'),
 ('Hz','hertz'),('kHz','kilohertz'),('MHz','megahertz'),('GHz','gigahertz'),
 ('Pa','pascal'),('kPa','kilopascal'),('MPa','megapascal'),('GPa','gigapascal'),
 ('eV','electronvolt'),('keV','kiloelectronvolt'),('MeV','megaelectronvolt'),('GeV','gigaelectronvolt')]
SYMBOLS=dict(UNIT_ROWS)
NAMES={}
for symbol,name in UNIT_ROWS:
    canonical=symbol.replace('μ','µ')
    for form in {name,name+'s' if not name.endswith('hertz') else name}:
        NAMES[form]=(canonical,name)
UNIT_RE=re.compile(r'(?<![\w])(?:'+ '|'.join(sorted(set(SYMBOLS)|set(NAMES),key=lambda s:(-len(s),s)))+r')(?![\w])')
FORMULA_SIG_KEYS=['formulas','mathematical_variables','operators']


def init():
    global nltk,splitter
    numeric.init();nltk=numeric.nltk;splitter=numeric.splitter


def containing_sentence(text,start):
    for a,b in splitter.span_tokenize(text):
        if a<=start<b:return a,b
    return 0,len(text)


def optional_that_candidates(text):
    tokenizer=nltk.tokenize.TreebankWordTokenizer()
    for m in re.finditer(r'\bthat[ \t]+',text):
        a,b=m.span();sa,sb=containing_sentence(text,a)
        prior=re.search(r'\b([A-Za-z]+)[ \t]+$',text[sa:a])
        if not prior or prior.group(1).lower() not in REPORTING:continue
        # The first word must be an explicit subject, not an extraposition or
        # demonstrative use: 'show that result' does not introduce a clause.
        spans=list(tokenizer.span_tokenize(text[b:sb]))[:18]
        tokens=[text[b+x:b+y] for x,y in spans]
        if not tokens:continue
        tags=nltk.pos_tag(tokens)
        if tags[0][1] not in {'DT','PRP','NN','NNS','NNP','NNPS'}:continue
        subject_seen=False;finite=False
        for word,tag in tags:
            if word in {',',';',':','(',')','.'} or tag in {'WDT','WP','WP$','WRB'}:break
            if tag.startswith('NN') or tag=='PRP':subject_seen=True
            if tag in {'VBD','VBP','VBZ','MD'}:
                finite=subject_seen;break
            if tag not in {'DT','PRP','NN','NNS','NNP','NNPS','JJ','JJR','JJS','CD','POS','RB','RBR','RBS','CC'}:break
        if not finite:continue
        if any(a<d and b>c for c,d in numeric.protected_spans(text)):continue
        yield dict(start=a,end=b,replacement='',rule='optional_reporting_that',old=text[a:b])


def preceding_quantity(text,start):
    # Match the entire adjacent numeric token/cardinal, never an arbitrary
    # suffix of 'one hundred'. Preserve sign and decimal-place precision.
    matches=[]
    for kind,pattern in [('digits',numeric.DIGIT_NUMBER),('words',numeric.WORD_NUMBER)]:
        for m in pattern.finditer(text[:start]):
            if re.fullmatch(r'[ \t]+',text[m.end():start]):matches.append((m,kind))
    if len(matches)!=1:return None
    m,kind=matches[0]
    try:value=Decimal(m.group().replace(',','').replace('−','-')) if kind=='digits' else numeric.parse_words(m.group())
    except (ValueError,ArithmeticError):return None
    if numeric.identifier_context(text,m.start(),m.end()):return None
    if m.start() and text[m.start()-1] in '#/^\\-−':return None
    return m,value,kind


def unit_candidates(text):
    for m in UNIT_RE.finditer(text):
        a,b=m.span();surface=m.group();quantity=preceding_quantity(text,a)
        if not quantity:continue
        qm,value,kind=quantity
        # Compound units, exponents, attached ratios, and mathematical sentences
        # retain their exact spelling. Do not expand just half of mg/mL.
        if re.match(r'[ \t]*[/^−–-]|[⁰¹²³⁴⁵⁶⁷⁸⁹]',text[b:]):continue
        sa,sb=containing_sentence(text,a);sentence=text[sa:sb]
        if LATEX.search(sentence) or re.search(r'[=<>≤≥≠±≈∝×÷^]|[A-Za-zα-ωΑ-Ω][\wα-ωΑ-Ω]*\s*\(',sentence):continue
        if any(qm.start()<d and b>c for c,d in numeric.protected_spans(text)):continue
        if surface in SYMBOLS:
            name=SYMBOLS[surface];canonical=surface.replace('μ','µ')
            replacement=name if abs(value)==1 or name.endswith('hertz') else name+'s'
            rule='unit_symbol_to_name'
        else:
            canonical,name=NAMES[surface.lower()];replacement=canonical;rule='unit_name_to_symbol'
        yield dict(start=a,end=b,replacement=replacement,old=surface,rule=rule,
                   quantity_surface=qm.group(),quantity_start=qm.start(),quantity_end=qm.end(),
                   number=str(value),decimal_places=value.as_tuple().exponent,canonical_unit=canonical)


def canonical_unit_text(text):
    # Used for a whole-fragment invariant in addition to exact edit provenance.
    # Replaces only adjacent quantity+unit contexts allowed by the same guards.
    changes=[]
    for e in unit_candidates(text):
        changes.append((e['start'],e['end'],e['canonical_unit']))
    for a,b,replacement in reversed(changes):text=text[:a]+replacement+text[b:]
    return text


def invariant(before,after):
    a=canonical_unit_text(before);b=canonical_unit_text(after)
    if named_values(a)!=named_values(b) or negations(a)!=negations(b):return False
    if Counter(numeric.MODALS.findall(a.lower()))!=Counter(numeric.MODALS.findall(b.lower())):return False
    old,new=signature(a),signature(b)
    return old==new


def candidates(text,mode):
    result=[]
    if mode in {'optional_that','combined'}:result.extend(optional_that_candidates(text))
    if mode in {'unit_aliases','combined'}:result.extend(unit_candidates(text))
    return result


def repair(text,index,mode):
    original=text;edits=[];initial_words=len(tokenize(text)[1])
    for _ in range(64):
        words,hits,covered=numeric.state(text,index)
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
            if not any(a<d and b>e for e,d in matches):continue
            after=text[:a]+c['replacement']+text[b:]
            new_words,new_hits,new_covered=numeric.state(after,index)
            if not new_words or len(new_words)>max(initial_words+4,int(initial_words*1.1)):continue
            if new_covered>=covered or new_covered/len(new_words)>=covered/len(words) or len(new_hits)>=len(hits):continue
            if not invariant(original,after):continue
            options.append(dict(c,before=text,after=after,covered_before=covered,covered_after=new_covered,gain=covered-new_covered))
        if not options:break
        best=min(options,key=lambda c:(-c['gain'],len(c['replacement']),c['start']))
        text=best['after'];edits.append(best)
    assert invariant(original,text)
    return text,edits


def verify_trace(before,after,edits):
    current=before
    for e in edits:
        assert e['before']==current
        a,b=e['start'],e['end'];assert current[a:b]==e['old']
        assert current[:a]+e['replacement']+current[b:]==e['after']
        allowed=candidates(current,'combined')
        assert any(c['start']==a and c['end']==b and c['replacement']==e['replacement'] and c['rule']==e['rule'] for c in allowed)
        assert invariant(current,e['after'])
        if e['rule'].startswith('unit_'):
            original_quantity=current[e['quantity_start']:e['quantity_end']]
            assert original_quantity==e['quantity_surface']
            assert current[:a]+e['old']+current[b:]==current
        current=e['after']
    assert current==after
    return True
