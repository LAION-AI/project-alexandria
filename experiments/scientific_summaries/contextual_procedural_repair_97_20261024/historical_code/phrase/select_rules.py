"""Freeze candidate allowlists using training documents only, before new97 QA."""
import re
from collections import Counter,defaultdict
from common import ROOT, load, read_jsonl, jsonl, write, digest


GOOD={'unchanged_or_equivalent','awkward_only'}


def collect():
    return [c for lane in range(3) for c in read_jsonl(ROOT/f'outputs/reviews/training-{lane}.jsonl')]


def select(refined=False):
    rows=collect();by=defaultdict(list)
    for c in rows:
        if c['kind']=='candidate_rule':by[c['rule']].append(c)
    adjustments=load(ROOT/'inputs/luna_refinements.json') if refined and (ROOT/'inputs/luna_refinements.json').exists() else {}
    blocked=adjustments.get('add_blocked_context_regex',{})
    disabled=set(adjustments.get('disable_rule_ids',[]))
    output=[];scores={}
    for original in load(ROOT/'inputs/candidate_rules.json')['rules']:
        rule=dict(original);rule['training_refinement_blocked_context_regex']=blocked.get(rule['id'],[])
        for pattern in rule['blocked_context_regex']+rule['training_refinement_blocked_context_regex']:re.compile(pattern,re.I)
        cards=[c for c in by[rule['id']] if not any(re.search(p,c['before'],re.I) for p in blocked.get(rule['id'],[]))]
        split={s:[c for c in cards if c['split']==s] for s in ['derivation','validation']}
        counts={s:dict(Counter(c['judgment']['severity'] for c in cs)) for s,cs in split.items()}
        def passes(cs,minimum,rate):
            return (len({c['document_id'] for c in cs})>=minimum
                and sum(c['judgment']['severity'] in GOOD for c in cs)/max(1,len(cs))>=rate
                and not any(c['judgment']['severity'] in {'major_meaning_change','unintelligible','unparsed'} for c in cs))
        approved=(rule['id'] not in disabled and passes(split['derivation'],5,.95) and passes(split['validation'],3,.90))
        scores[rule['id']]=dict(approved=approved,counts=counts,documents={s:len({c['document_id'] for c in cs}) for s,cs in split.items()})
        if approved:output.append(rule)
    version='v2' if refined else 'v1'
    value=dict(version=version,rules=output,all_rule_scores=scores,
        selection='Derivation >=5 documents, >=95% equivalent/awkward; validation >=3 documents, >=90%; zero major/unintelligible/unparsed in either. Modifications use only derivation failures, not QA gold.',
        refinements=adjustments if refined else None)
    write(ROOT/f'inputs/learned_rules_{version}.json',value)
    write(ROOT/f'outputs/rule_selection_{version}.json',value)
    print('LEARNED_ALLOWLIST',version,len(output),flush=True)
    return value


def feedback():
    rows=collect()
    failures=[c for c in rows if c['kind']=='candidate_rule' and c['split']=='derivation' and c['judgment']['severity'] not in GOOD]
    jsonl(ROOT/'outputs/luna_derivation_failures.jsonl',failures)
    jsonl(ROOT/'outputs/luna_dictionary_derivation_failures.jsonl',[c for c in rows if c['kind']!='candidate_rule' and c['split']=='derivation' and c['judgment']['severity'] not in GOOD])
    diagnostic={kind:dict(Counter(c['judgment']['severity'] for c in rows if c['kind']==kind)) for kind in ['existing_guarded_dictionary','naive_dictionary_diagnostic']}
    write(ROOT/'outputs/dictionary_diagnostic.json',diagnostic)
    select(False)
    write(ROOT/'outputs/awaiting_luna_refinement.json',dict(complete=True,derivation_failures=len(failures),
        instructions='Only derivation failures are supplied for refinement; validation labels and new97 outputs must not guide new contexts. Add block regexes or disable rules, never silently replace QA-tested outputs.'))


if __name__=='__main__':
    import sys
    feedback() if sys.argv[1]=='feedback' else select(True)
