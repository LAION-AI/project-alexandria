"""Typed redundant elaborations and component-wise joint insertion planning.

No sentence is regenerated. Original characters/word order and scientific
quantities remain. Parenthetical unit names are typed notation explanations;
acronym glosses come from an unambiguous definition in this paper's full source.
All final proposals are audited against that same complete source.
"""
import re,math,itertools,statistics
from decimal import Decimal,InvalidOperation
from collections import Counter,defaultdict
import legacy_insertions as legacy
import number_style as numeric
import grammar_units as grammar
from common import named_values,negations
from critical_values import signature,LATEX,RELATION,OPERATION,FUNCTION
from ngram_overlap import tokenize
from reference_aliases import ALIASES

STOP={'the','a','an','of','and','or','for','to','in','on','with','by','from'}
TEXTUAL={'shown','reported','described','presented','listed','displayed','summarized','illustrated','given'}
ACTION={'measured','estimated','computed','calculated','determined','evaluated','assessed','tested','analyzed','analysed','collected','obtained','generated','extracted','detected','identified','quantified','synthesized','prepared','amplified','calibrated','fitted','simulated','modeled','modelled','optimized','solved','filtered','mapped','constructed','recorded','produced','validated','processed','isolated','labeled','labelled','sequenced'}
GERUNDS=set('measuring estimating computing calculating evaluating testing analyzing analysing collecting generating extracting detecting quantifying synthesizing preparing amplifying calibrating fitting simulating modeling modelling optimizing solving filtering mapping constructing recording processing heating cooling stirring incubating annealing drying centrifuging training averaging integrating differentiating sampling imaging scanning'.split())
TOOLS=set('method methods model models algorithm algorithms instrument instruments detector detectors sensor sensors microscope microscopy spectrometer spectroscopy software package packages library libraries code program programs apparatus tool tools technique techniques platform platforms protocol protocols framework frameworks computer computers HPLC NMR MRI SEM TEM XRD Python MATLAB R'.split())
HEADS={'result':'singular','sample':'singular','measurement':'singular','method':'singular','model':'singular','value':'singular','experiment':'singular','observation':'singular','study':'singular','procedure':'singular','algorithm':'singular','dataset':'singular','figure':'singular','table':'singular','parameter':'singular','estimate':'singular','approach':'singular','system':'singular',
       'results':'plural','samples':'plural','measurements':'plural','methods':'plural','models':'plural','values':'plural','experiments':'plural','observations':'plural','studies':'plural','procedures':'plural','algorithms':'plural','datasets':'plural','figures':'plural','tables':'plural','parameters':'plural','estimates':'plural','approaches':'plural','systems':'plural'}
# Finite exact aliases. No rescaling or inference of an ambiguous physical role.
UNIT_ROWS=grammar.UNIT_ROWS+[
 ('°C','degree Celsius'),('°F','degree Fahrenheit'),('K','kelvin'),
 ('h','hour'),('min','minute'),('sec','second'),('µs','microsecond'),('μs','microsecond'),
 ('mol','mole'),('mmol','millimole'),('atm','atmosphere'),('bar','bar'),('mbar','millibar'),
 ('mol%','mole percent'),('wt%','weight percent'),('vol%','volume percent'),('at%','atomic percent'),
 ('T','tesla'),('mT','millitesla'),('V','volt'),('mV','millivolt'),('W','watt'),('kW','kilowatt')]
UNIT_NAMES=dict(UNIT_ROWS)
UNIT_RE=re.compile(r'(?<![A-Za-zµμ])(?:'+'|'.join(sorted(UNIT_NAMES,key=lambda s:(-len(s),s)))+r')(?![\w])')
DIMENSION={**{u:'temperature' for u in ['°C','°F','K']},**{u:'pressure' for u in ['Pa','kPa','MPa','GPa','atm','bar','mbar']},
           **{u:'frequency' for u in ['Hz','kHz','MHz','GHz']},**{u:'energy' for u in ['eV','keV','MeV','GeV']},
           **{u:'voltage' for u in ['V','mV']},**{u:'power' for u in ['W','kW']},**{u:'magnetic field' for u in ['T','mT']}}
TIME_UNITS={'h','min','sec','s','ms','µs','μs'}
SUFFIX_NUMBER=re.compile(r'(?<![\w.+-])[-−]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)$')
EXTRA_FORBIDDEN=legacy.FORBIDDEN|{'apparently','presumably','necessarily','roughly','approximately','almost','exactly','solely','exclusively'}


def init():
    global nltk,splitter,word_tokenizer
    legacy.init();nltk=legacy.nltk;splitter=legacy.splitter;word_tokenizer=legacy.word_tokenizer


def definition_inventory(source):
    """PDF-aware forward/reverse definitions plus source-grounded aliases."""
    # A wrapped PDF line is not a semantic boundary. Paragraph punctuation is
    # retained. Only explicit line-end hyphenation is joined.
    source=re.sub(r'(?<=\w)[-‐‑][ \t]*\n[ \t]*(?=\w)','',source).replace('\n',' ')
    definitions=defaultdict(dict);provenance={}
    def aligns(expansion,abbr):
        words=re.findall(r'[A-Za-z]+(?:-[A-Za-z]+)*',expansion)
        if not words or words[0].lower() in STOP or expansion.lower()==abbr.lower():return False
        if any(w.lower()==abbr.lower() for w in words):return False
        forms=[{w[0].upper(),''.join(p[0].upper() for p in w.split('-'))} for w in words if w.lower() not in STOP]
        if len(forms)<=8 and any(''.join(parts)==abbr for parts in itertools.product(*forms)):return True
        # DNA/RNA require letters internal to 'deoxyribonucleic/ribonucleic'.
        # Restrict this relaxation to>=3 letters and>=2 word-initial matches.
        if len(abbr)<3:return False
        long=expansion.upper();j=len(long)-1;initial_matches=0
        for i in range(len(abbr)-1,-1,-1):
            while j>=0 and (long[j]!=abbr[i] or i==0 and j>0 and long[j-1].isalpha()):j-=1
            if j<0:return False
            if j==0 or not long[j-1].isalpha():initial_matches+=1
            j-=1
        return initial_matches>=max(2,math.ceil(len(abbr)/2))
    def register(abbr,expansion,kind):
        expansion=' '.join(expansion.split())
        if not 1<=len(tokenize(expansion)[1])<=5:return
        if not aligns(expansion,abbr):return
        key=expansion.lower();definitions[abbr][key]=expansion;provenance[abbr,key]=kind
    for m in re.finditer(r'\(([A-Z][A-Z0-9-]{1,9})\)',source):
        abbr=m.group(1)
        if not re.fullmatch(r'[A-Z]{2,8}',abbr):continue
        tail=re.split(r'[.!?;:]',source[max(0,m.start()-180):m.start()])[-1]
        tokens=list(re.finditer(r'[A-Za-z]+(?:-[A-Za-z]+)*',tail))
        possible=[]
        for n in range(1,min(10,len(tokens))+1):
            selected=tokens[-n:]
            expansion=tail[selected[0].start():selected[-1].end()].strip()
            if len(tokenize(expansion)[1])>5:continue
            words=[t.group() for t in selected]
            if words[0].lower() in STOP:continue
            if aligns(expansion,abbr):possible.append(expansion)
        if possible:
            expansion=min(possible,key=lambda s:(len(tokenize(s)[1]),len(s)))
            register(abbr,expansion,'explicit_forward')
    for m in re.finditer(r'\b([A-Z]{2,8})[ \t]*\(([A-Za-z][A-Za-z \t-]{2,100})\)',source):
        register(m.group(1),m.group(2),'explicit_reverse')
    # A reference expansion is eligible only if that exact term, allowing
    # hyphen/space spelling and a plural final noun, is in this same paper.
    flat=re.sub(r'[^a-z]+',' ',source.lower())
    for abbr,expansions in ALIASES.items():
        if abbr in definitions:continue
        if not re.search(r'\b'+re.escape(abbr)+r'\b',source):continue
        grounded=[]
        for expansion in expansions:
            phrase=re.sub(r'[^a-z]+',' ',expansion.lower()).strip()
            if re.search(r'\b'+re.escape(phrase)+r's?\b',flat):grounded.append(expansion)
        # Near-synonymous MSE expansions collapse to one canonical term; all
        # genuinely different grounded senses (e.g.DFT) remain excluded.
        if abbr=='MSE' and grounded:grounded=grounded[:1]
        if len(grounded)==1:
            expansion=grounded[0];key=expansion.lower();definitions[abbr][key]=expansion;provenance[abbr,key]='reference_term_present_in_source'
    return {abbr:dict(expansion=next(iter(values.values())),kind=provenance[abbr,next(iter(values))]) for abbr,values in definitions.items() if len(values)==1}


def source_glossary(source):
    return {abbr:value['expansion'] for abbr,value in definition_inventory(source).items()}


def prepare_index(index):
    if not hasattr(index,'insertion_glossary'):
        index.insertion_glossary_provenance=definition_inventory(index.source_text)
        index.insertion_glossary={abbr:value['expansion'] for abbr,value in index.insertion_glossary_provenance.items()}


def raw_number_before(text,start):
    prefix=text[:start];m=SUFFIX_NUMBER.search(prefix.rstrip(' \t'))
    if m:
        a,b=m.span()
        try:value=Decimal(m.group().replace(',','').replace('−','-'))
        except InvalidOperation:return None
        return a,b,value
    result=grammar.preceding_quantity(text,start)
    return (result[0].start(),result[0].end(),result[1]) if result else None


def unit_quantities(text):
    for m in UNIT_RE.finditer(text):
        a,b=m.span();unit=m.group();quantity=raw_number_before(text,a)
        if not quantity:continue
        qa,qb,value=quantity;gap=text[qb:a]
        if gap.strip() or numeric.identifier_context(text,qa,qb):continue
        if qa and text[qa-1] in '#/^\\-−':continue
        if re.match(r'[ \t]*(?:[/^%·⋅⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺−–-]|\.(?=[A-Za-zµμ])|\bper\b)',text[b:]):continue
        adjacent=re.match(r'[ \t]+([^ \t,.;:()]+)',text[b:])
        if adjacent and any(re.match(re.escape(u)+r'(?!\w)',adjacent.group(1)) for u in UNIT_NAMES):continue
        # Whitespace is required for single-letter units to avoid orbitals,
        # labels and names such as1s, 3T and2020s.
        if len(unit)==1 and not gap:continue
        left=text[max(0,qa-140):qa].lower()
        right=text[b:b+70].lower()
        # h can be Planck's constant, g a gravitational acceleration, and K/L
        # technical labels. Explain these symbols only in a positive context.
        if unit in {'h','s','sec'} and not re.search(r'\b(?:for|after|before|within|over|duration|time|lasting|lasted|incubat\w*|heat\w*|cool\w*|stir\w*)\b',left):continue
        if unit=='g':
            if re.search(r'gravitation|gravity|centrifug|acceleration|g-force|orbital',left+right):continue
            if not re.search(r'mass|weigh|contain|add|dissolv|load|yield|dose|prepar|sample|compound|amount|mix|solut|reagent|tissue',left):continue
        if unit=='K' and not re.search(r'\bat[ \t]*$|temperature|thermal|kelvin|heat|cool|anneal|incubat',left):continue
        if unit=='L' and not re.search(r'volume|solution|water|ethanol|solvent|mix|add|contain|flow|liter',left):continue
        if len(unit)==1 and unit in {'T','V','W'}:
            cues={'T':r'magnet|field|tesla','V':r'voltage|potential|volt|bias','W':r'power|watt'}
            if not re.search(cues[unit],left):continue
        yield dict(start=qa,end=b,number_end=qb,unit_start=a,value=value,unit=unit)


def scoped_quantity_safe(text,q,dimension):
    tail=text[q['end']:]
    connector=re.match(r'[ \t]*(?:,|and|or)[ \t]*',tail)
    if connector:
        offset=q['end']+connector.end()
        for second in unit_quantities(text):
            if second['start']==offset:
                return DIMENSION.get(second['unit'])==dimension
    return True


def scientific_guard(before,after):
    """Strict inserted-word provenance plus numeric/role protections.

    An explicitly typed unit gloss after K can resemble a function K(kelvin)
    to the generic regex. Original formula spans are separately protected by
    insertion-position exclusions; no inserted text may contain math symbols.
    """
    if before==after:return True
    a,b=signature(before),signature(after)
    keys=['numbers','quantities','mathematical_variables','operators']
    return (all(a[k]==b[k] for k in keys) and named_values(before)==named_values(after)
            and negations(before)==negations(after)
            and Counter(w for w in re.findall(r'\b\w+\b',before.lower()) if w in EXTRA_FORBIDDEN)==Counter(w for w in re.findall(r'\b\w+\b',after.lower()) if w in EXTRA_FORBIDDEN)
            and Counter(numeric.MODALS.findall(before.lower()))==Counter(numeric.MODALS.findall(after.lower())))


def candidates(text,index):
    prepare_index(index);result=[];protected=numeric.protected_spans(text)
    def add(p,insertion,rule,key=None,quantity=None):
        if any(a<p<b for a,b in protected):return
        if re.search(r'\d|[α-ωΑ-Ω=<>≤≥≠±≈∝×÷^/\\]',insertion):return
        if set(re.findall(r'\b\w+\b',insertion.lower())) & EXTRA_FORBIDDEN:return
        # Avoid nested explanatory parentheses and quoted labels.
        if rule in {'unit_name_gloss','source_acronym_gloss'}:
            prefix=text[:p]
            if prefix.count('(')>prefix.count(')') or re.match(r'[ \t]*\(',text[p:]):return
        after=text[:p]+insertion+text[p:]
        if not scientific_guard(text,after):return
        result.append(dict(start=p,end=p,replacement=insertion,rule=rule,key=key,quantity=quantity))
    # Preserve all previously checked narrow grammar rules except a dangerous
    # shared-temperature scope. Experimental emphasis/example cues are omitted.
    quantities=list(unit_quantities(text))
    for c in legacy.candidates(text,'grammar_insert'):
        after=text[:c['start']]+c['replacement']+text[c['start']:]
        if c['rule']=='explicit_temperature' and not legacy.temperature_scope_safe(after):continue
        quantity=next(((q['start'],q['end']) for q in quantities if q['start']==c['start']),None)
        add(c['start'],c['replacement'],c['rule'],quantity=quantity)
    for q in quantities:
        a,b,unit=q['start'],q['end'],q['unit'];value=q['value']
        if any(a<d and b>c for c,d in protected):continue
        name=UNIT_NAMES[unit]
        if name.startswith('degree '):name=('degree ' if abs(value)==1 else 'degrees ')+name[7:]
        elif abs(value)!=1 and not name.endswith(('hertz','siemens','percent')):name+='s'
        add(b,' ('+name+')','unit_name_gloss',key='unit:'+unit,quantity=(a,b))
        left=text[max(0,a-30):a]
        if re.search(r'\bat[ \t]+$',left,re.I) and unit in DIMENSION:
            dimension=DIMENSION[unit]
            if scoped_quantity_safe(text,q,dimension):
                article='an' if dimension=='energy' else 'a'
                add(a,article+' '+dimension+' of ','typed_'+dimension.replace(' ','_'),quantity=(a,b))
        if unit in TIME_UNITS and re.search(r'\b(?:for|over|after|within)[ \t]+$',left,re.I):
            add(a,'a period of ','abbreviated_duration',quantity=(a,b))
    for abbr,expansion in index.insertion_glossary.items():
        if expansion.lower() in text.lower():continue
        for m in re.finditer(r'\b'+re.escape(abbr)+r'\b',text):
            add(m.end(),' ('+expansion+')','source_acronym_gloss',key='acronym:'+abbr)
    # Explicit purpose constructions retain their existing purpose relation.
    for m in re.finditer(r'\b(?:designed|developed|intended|constructed|implemented|employed|utilized)[ \t]+(?P<to>to)[ \t]+([a-z]+)\b',text,re.I):
        if m.group('to')=='to' and nltk.pos_tag(['to',m.group(2)])[1][1]=='VB':
            add(m.start('to'),'in order ','purpose_design_in_order')
    for m in re.finditer(r'\b(?:used|designed|intended|employed|utilized)[ \t]+for[ \t]+(?P<action>[a-z]+)\b',text,re.I):
        if m.group('action') in GERUNDS:add(m.start('action'),'the purpose of ','purpose_for_gerund')
    for m in re.finditer(r'\b(?P<prep>by|through|before|after)[ \t]+(?P<action>[a-z]+)\b',text):
        if m.group('action') not in GERUNDS:continue
        prep=m.group('prep');words='means of ' if prep=='by' else 'the process of '
        add(m.start('action'),words,'gerund_'+prep+'_elaboration')
    for m in re.finditer(r'\bduring[ \t]+(?P<event>(?:the |an? )?(?:study|experiment|training|testing|evaluation|measurement|heating|cooling|incubation|annealing|recording|simulation)\b)',text,re.I):
        add(m.start('event'),'the course of ','during_course')
    # Instrumental with/via requires a methodological participle and an explicit
    # tool noun. Exclude accompaniment, error tolerances and patient attributes.
    for m in re.finditer(r'\b(?P<verb>[a-z]+)[ \t]+(?P<prep>with|via)[ \t]+(?P<np>[^,.;:\n]{1,110})',text):
        if m.group('verb') not in ACTION:continue
        words=word_tokenizer.tokenize(m.group('np'))[:7]
        if any(w.lower() in {'error','uncertainty','precision','accuracy','confidence','resolution','tolerance','patient','patients','subject','subjects'} for w in words[:4]):continue
        tagged=nltk.pos_tag(words);head=None
        for word,tag in tagged:
            if tag in {'IN','TO','CC','VB','VBD','VBP','VBZ'} or tag=='VBG' and word.lower() not in {'scanning','imaging'}:break
            if tag.startswith('NN'):head=word
        if head and (head in TOOLS or head.lower() in TOOLS):
            add(m.start('np'),'the use of ','instrument_'+m.group('prep'))
    # Restore the marker in both-of demonstratives with one plural NP, avoiding
    # 'both X and Y', which has a different coordination structure.
    for m in re.finditer(r'\bboth[ \t]+(?P<det>these|those|the)[ \t]+(?P<np>[^,.;:\n]{1,80})',text,re.I):
        words=word_tokenizer.tokenize(m.group('np'))[:6]
        if any(w.lower() in {'and','or'} for w in words):continue
        if any(t=='NNS' for _,t in nltk.pos_tag(words)):add(m.start('det'),'of ','both_of_plural_np')
    # Expand reduced textual relatives using fixed noun agreement. Action
    # relatives additionally require a past-tense containing sentence.
    head_pattern='|'.join(sorted(HEADS,key=lambda s:(-len(s),s)))
    for m in re.finditer(r'\b(?P<head>'+head_pattern+r')[ \t]+(?P<part>[a-z]+)[ \t]+(?P<prep>in|by|with|using|from|for|at|above|below|here)\b',text,re.I):
        head=m.group('head').lower();part=m.group('part').lower();plural=HEADS[head]=='plural'
        if part in TEXTUAL:aux='are' if plural else 'is';rule='textual_reduced_relative'
        elif part in ACTION:
            sa,sb=grammar.containing_sentence(text,m.start())
            tokens=word_tokenizer.tokenize(text[sa:sb]);tagged=nltk.pos_tag(tokens)
            finite=[(w,t) for w,t in tagged if t in {'VBD','VBP','VBZ','MD'} and w.lower()!=part]
            if not finite or finite[0][1]!='VBD' or any(t=='MD' for _,t in finite):continue
            aux='were' if plural else 'was';rule='past_reduced_relative'
        else:continue
        add(m.start('part'),'that '+aux+' ',rule)
    unique={}
    for c in result:unique.setdefault((c['start'],c['replacement']),c)
    return list(unique.values())


def render(original,plan):
    out=original
    for c in sorted(plan,key=lambda x:x['start'],reverse=True):out=out[:c['start']]+c['replacement']+out[c['start']:]
    return out


def compatible(plan):
    if len({c['start'] for c in plan})!=len(plan):return False
    keys=[c['key'] for c in plan if c.get('key')]
    if any(n>2 for key,n in Counter(keys).items()):return False
    quantities=[tuple(c['quantity']) for c in plan if c.get('quantity')]
    # One explanation of any one physical quantity; do not stack its dimension
    # prefix with a redundant unit-name parenthesis.
    return len(quantities)==len(set(quantities))


def plan_state(original,index,plan,original_words):
    after=render(original,plan);words=tokenize(after)[1];origins=list(range(len(original_words)))
    current=original
    for c in sorted(plan,key=lambda x:x['start'],reverse=True):
        k=len(tokenize(original[:c['start']])[1]);added=len(tokenize(c['replacement'])[1])
        origins=origins[:k]+[None]*added+origins[k:]
    if len(words)!=len(origins):return None
    if [words[i] for i,o in enumerate(origins) if o is not None]!=original_words:return None
    hits,covered=legacy.covered_positions(words,index)
    original_covered=sum(origins[i] is not None for i in covered)
    return dict(after=after,words=len(words),hits=len(hits),covered=len(covered),original_covered=original_covered,added=len(words)-len(original_words))


def repair(text,index,mode,max_add_words=None):
    parsed=legacy.spans_and_words(text)
    stats=dict(candidates=0,eligible_candidates=0,components=0,evaluated_plans=0,joint_plans=0,zero_gain_intermediate_plans=0,rule_candidates={})
    if not parsed:return text,[],stats
    spans,words=parsed;hits,covered=legacy.covered_positions(words,index)
    if not hits:return text,[],stats
    all_candidates=candidates(text,index);stats['candidates']=len(all_candidates)
    if mode=='system_core':
        all_candidates=[c for c in all_candidates if c['rule'] not in {'source_acronym_gloss','past_reduced_relative','textual_reduced_relative'}]
        stats['candidates']=len(all_candidates)
    stats['rule_candidates']=dict(Counter(c['rule'] for c in all_candidates))
    eligible=[]
    for c in all_candidates:
        k=len(tokenize(text[:c['start']])[1])
        if any(h<k<h+6 for h in hits):eligible.append(c)
    stats['eligible_candidates']=len(eligible)
    if not eligible:return text,[],stats
    budget=min(32,max(8,math.ceil(len(words)*.20)))
    if max_add_words is not None:budget=min(budget,max_add_words)
    initial=plan_state(text,index,[],words);cache={():initial}
    def state(ids):
        key=tuple(sorted(ids))
        if key in cache:return cache[key]
        plan=[eligible[i] for i in key]
        if not compatible(plan):cache[key]=None;return None
        if sum(len(tokenize(c['replacement'])[1]) for c in plan)>budget:cache[key]=None;return None
        value=plan_state(text,index,plan,words);cache[key]=value;stats['evaluated_plans']+=1
        return value
    def improves(a,b):
        return a and a['original_covered']<b['original_covered'] and a['covered']<b['covered'] and a['hits']<b['hits'] and a['covered']/a['words']<b['covered']/b['words']
    def rank(ids):
        s=state(ids)
        return (s['original_covered'],s['covered'],s['hits'],s['added'],tuple(ids))
    chosen=();current=initial
    # Connected components of originally covered word positions. This retains
    # neutral intermediate steps inside long runs instead of letting unrelated
    # easy six-word runs consume the whole search beam.
    intervals=[]
    for i in hits:
        if intervals and i<=intervals[-1][1]:intervals[-1]=(intervals[-1][0],max(intervals[-1][1],i+6))
        else:intervals.append((i,i+6))
    for a,b in intervals:
        options=[i for i,c in enumerate(eligible) if a<len(tokenize(text[:c['start']])[1])<b]
        if not options:continue
        stats['components']+=1
        if mode=='system_greedy':
            while True:
                valid=[tuple(sorted(chosen+(i,))) for i in options if i not in chosen and improves(state(chosen+(i,)),current)]
                if not valid:break
                best=min(valid,key=lambda ids:((rank(ids)[0]-current['original_covered'])/max(1,state(ids)['added']-current['added']),rank(ids)))
                chosen=best;current=state(chosen)
        else:
            beam=[chosen];best=chosen;best_state=current
            # Six insertions cover the longest25-word source runs in this
            # cohort when sufficient safe locations exist; search is bounded.
            for depth in range(min(6,len({eligible[i]['start'] for i in options}))):
                expanded=set()
                for ids in beam:
                    for i in options:
                        if i not in ids:
                            key=tuple(sorted(ids+(i,)));s=state(key)
                            if s:expanded.add(key)
                if not expanded:break
                beam=sorted(expanded,key=rank)[:24]
                for ids in beam:
                    s=state(ids)
                    if improves(s,current) and (best==chosen or rank(ids)<rank(best)):
                        best=ids;best_state=s
            if best!=chosen:
                new_ids=[i for i in best if i not in chosen]
                if len(new_ids)>1:
                    stats['joint_plans']+=1
                    if any(not improves(state(chosen+(i,)),current) for i in new_ids):stats['zero_gain_intermediate_plans']+=1
                chosen=best;current=best_state
    if not chosen or not improves(current,initial):return text,[],stats
    plan=[eligible[i] for i in chosen]
    after=current['after']
    if not scientific_guard(text,after):return text,[],stats
    # Original formula/code/quote spans remain literally contiguous.
    for regex in [LATEX,RELATION,OPERATION,FUNCTION]:
        for m in regex.finditer(text):
            assert not any(m.start()<c['start']<m.end() for c in plan)
    trace=dict(before=text,after=after,edits=plan,metrics_before=initial,metrics_after=current,search=stats)
    return after,[trace],stats


def verify_trace(before,after,traces,index=None):
    if not traces:return before==after
    assert len(traces)==1;trace=traces[0]
    assert trace['before']==before and trace['after']==after
    assert render(before,trace['edits'])==after and compatible(trace['edits'])
    assert scientific_guard(before,after)
    if index is not None:
        catalog={(c['start'],c['replacement'],c['rule']) for c in candidates(before,index)}
        assert all((c['start'],c['replacement'],c['rule']) in catalog for c in trace['edits'])
    return True
