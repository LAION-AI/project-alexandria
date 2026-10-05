"""Evaluate actual raw candidates; accept conservative repairs independently of QA."""
from collections import Counter,defaultdict
import json
import os
import time
from common import ROOT,ARMS,load,read_jsonl,write,jsonl,sha
from critical_values import check,VERSION as CRITICAL_VERSION
from passages import assemble
from ngram_overlap import SourceIndex,audit_summary,fragments
from copy_overlap_eval import audit_conditions
from reference_validator import narrative_context,validate_summary


class NLI:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        torch.set_num_threads(8);self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/nli',local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(ROOT/'models/nli',local_files_only=True,
            dtype=torch.float16).to('cuda').eval()
        self.entailment=next(int(k) for k,v in self.model.config.id2label.items() if v.lower()=='entailment')
        self.contradiction=next(int(k) for k,v in self.model.config.id2label.items() if v.lower()=='contradiction')

    def score(self,pairs):
        tick=time.monotonic();encoded=[self.tokenizer(a,b,add_special_tokens=True,truncation=False) for a,b in pairs]
        values=[None]*len(pairs);valid=[i for i,v in enumerate(encoded) if len(v['input_ids'])<=512]
        valid.sort(key=lambda i:len(encoded[i]['input_ids']))
        for start in range(0,len(valid),32):
            indices=valid[start:start+32]
            batch=self.tokenizer.pad([encoded[i] for i in indices],padding=True,return_tensors='pt')
            batch={k:v.to('cuda') for k,v in batch.items()}
            with self.torch.inference_mode():scores=self.model(**batch).logits.float().softmax(-1).cpu().tolist()
            for i,s in zip(indices,scores):values[i]=dict(entailment=s[self.entailment],contradiction=s[self.contradiction])
        return values,dict(seconds=time.monotonic()-tick,pairs=len(pairs),scored_pairs=len(valid),
                           overlength_pairs_not_truncated=len(pairs)-len(valid),batch_size=32)


def calibration(nli):
    cases=[('The treatment reduced mortality.','Mortality decreased following the treatment.',True),
           ('The result was not statistically significant.','The result was statistically significant.',False),
           ('The treatment reduced mortality.','The treatment increased mortality.',False),
           ('The sample contained 20 patients.','The sample contained 200 patients.',False),
           ('The mass was 5 mg.','The mass was 5 g.',False),
           ('The value satisfies x < 5.','The value satisfies x > 5.',False),
           ('The study suggests a possible association.','The study proves a causal relationship.',False)]
    pairs=[pair for a,b,_ in cases for pair in [(a,b),(b,a)]];scores,perf=nli.score(pairs)
    rows=[]
    for i,(a,b,equivalent) in enumerate(cases):
        left,right=scores[2*i:2*i+2];accept=left and right and min(left['entailment'],right['entailment'])>=.9
        rows.append(dict(before=a,after=b,known_equivalent=equivalent,bidirectional_nli_accepts=bool(accept),
                         forward=left,backward=right,critical_values=check(a,b)))
    write(ROOT/'outputs/nli_control_examples.json',dict(cases=rows,performance=perf,
          interpretation='Small diagnostic controls, not a calibrated scientific-domain accuracy estimate; cutoff fixed before heldout QA'))


def evaluate_arm(arm,nli):
    output=ROOT/'outputs/quality'/arm;output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists() and load(output/'complete.json').get('critical_values_version')==CRITICAL_VERSION:return
    tick=time.monotonic();cohort=read_jsonl(ROOT/'inputs/cohort.jsonl');windows=read_jsonl(ROOT/'inputs/windows.jsonl')
    candidates=read_jsonl(ROOT/'outputs/translation'/arm/'roundtrip.jsonl')
    assert [r['window_id'] for r in candidates]==[w['window_id'] for w in windows]
    indices=[i for i,r in enumerate(candidates) if r['status']=='translated']
    pairs=[pair for i in indices for pair in [(windows[i]['text'],candidates[i]['text']),(candidates[i]['text'],windows[i]['text'])]]
    scores,performance=nli.score(pairs);nli_scores={i:scores[2*j:2*j+2] for j,i in enumerate(indices)}
    byuid=defaultdict(list);sources={c['uid']:SourceIndex(c['fulltext']) for c in cohort};details=[]
    for i,(w,r) in enumerate(zip(windows,candidates)):
        original=w['text'];translated=r['text'] if r['status']=='translated' else original
        critical=check(original,translated);forward,backward=nli_scores.get(i,[None,None])
        overlap=sources[w['uid']].fragment(translated,w['path'],'narrative')
        reasons=[]
        if r['status']!='translated':reasons.append(r['status'])
        if not critical['passed']:reasons.append('critical_values_changed_or_unverified')
        if not forward or not backward:reasons.append('nli_pair_unassessable')
        elif min(forward['entailment'],backward['entailment'])<.9:reasons.append('nli_equivalence_below_fixed_cutoff')
        if overlap['longest_contiguous_match_words']>5:reasons.append('source_copy_run_exceeds_five')
        accepted=not reasons
        row=dict(window_id=w['window_id'],uid=w['uid'],original_document_id=w['document_id'],path=w['path'],
                 original=original,german=r['german'],backtranslated=translated,translation_status=r['status'],
                 critical_values=critical,nli_forward=forward,nli_backward=backward,
                 candidate_source_copy_longest=overlap['longest_contiguous_match_words'],accepted=accepted,rejection_reasons=reasons)
        details.append(row);byuid[w['uid']].append((w,translated,accepted))
    jsonl(output/'window_quality.jsonl',details)
    summaries=[];contexts={'original':{},'raw_backtranslation':{},'guarded_backtranslation':{}};papers=[]
    for c in cohort:
        items=byuid[c['uid']];wins=[i[0] for i in items];texts=[i[1] for i in items];accept=[i[2] for i in items]
        raw=assemble(c['summary'],wins,texts);guarded=assemble(c['summary'],wins,texts,accept)
        raw_context=narrative_context(raw);guarded_context=narrative_context(guarded)
        entire=check(c['judge_context'],guarded_context)
        rollback=not entire['passed']
        if rollback:guarded=c['summary'];guarded_context=c['judge_context']
        schema_error=None
        try:validate_summary(json.dumps(guarded,ensure_ascii=False),c['fulltext'])
        except (ValueError,TypeError,AttributeError,KeyError,IndexError) as e:schema_error=str(e)
        before_nonprose=[(text,path,cat) for text,path,cat in fragments(c['summary']) if cat!='narrative']
        after_nonprose=[(text,path,cat) for text,path,cat in fragments(guarded) if cat!='narrative']
        assert before_nonprose==after_nonprose
        row=dict(uid=c['uid'],document_id=c['document_id'],original_condition=c['original_condition'],source_sha256=c['source_sha256'],
                 raw_summary=raw,raw_judge_context=raw_context,summary=guarded,judge_context=guarded_context,
                 raw_global_critical_values=check(c['judge_context'],raw_context),
                 guarded_global_critical_values=check(c['judge_context'],guarded_context),
                 accepted_window_count=sum(accept) if not rollback else 0,selected_window_count=len(wins),
                 summary_rolled_back=rollback,source_schema_validation_error=schema_error,
                 original_context_sha256=sha(c['judge_context']),guarded_context_sha256=sha(guarded_context))
        summaries.append(row)
        papers.append(dict(document_id=c['uid'],fulltext=c['fulltext'],source_sha256=c['source_sha256']))
        for name,summary,context in [('original',c['summary'],c['judge_context']),('raw_backtranslation',raw,raw_context),('guarded_backtranslation',guarded,guarded_context)]:
            contexts[name][c['uid']]=dict(status='generated',summary=summary,judge_context=context)
    jsonl(output/'summaries.jsonl',summaries)
    overlap=audit_conditions(papers,contexts,output)
    attempted=[r for r in details if r['translation_status']=='translated']
    numerical=[r for r in attempted if r['critical_values']['numeric_content_present']]
    mathematical=[r for r in attempted if r['critical_values']['formula_content_present']]
    metric=dict(arm=arm,critical_values_version=CRITICAL_VERSION,summary_versions=len(summaries),unique_papers=len({r['document_id'] for r in summaries}),
                selected_windows=len(details),translated_windows=len(attempted),accepted_windows=sum(r['accepted'] for r in details),
                actually_inserted_windows=sum(r['accepted_window_count'] for r in summaries),
                changed_summaries=sum(r['guarded_context_sha256']!=r['original_context_sha256'] for r in summaries),
                rolled_back_summaries=sum(r['summary_rolled_back'] for r in summaries),
                numeric_windows=len(numerical),numeric_signature_changed=sum(not r['critical_values']['checks']['numbers'] for r in numerical),
                unit_signature_changed=sum(not r['critical_values']['checks']['quantities'] for r in attempted),
                formula_windows=len(mathematical),formula_signature_changed=sum(not r['critical_values']['checks']['formulas'] or not r['critical_values']['checks']['mathematical_variables'] or not r['critical_values']['checks']['operators'] for r in mathematical),
                raw_summaries_with_suspect_critical_change=sum(not r['raw_global_critical_values']['passed'] for r in summaries),
                guarded_summaries_with_suspect_critical_change=sum(not r['guarded_global_critical_values']['passed'] for r in summaries),
                nli_pairwise_below_cutoff=sum(r['nli_forward'] and r['nli_backward'] and min(r['nli_forward']['entailment'],r['nli_backward']['entailment'])<.9 or False for r in attempted),
                rejection_reasons=dict(Counter(reason for r in details for reason in r['rejection_reasons'])),
                nli_performance=performance,total_quality_seconds=time.monotonic()-tick,copy_overlap=overlap,
                critical_change_flags_are_conservative_not_human_error_labels=True,
                no_question_or_gold_used_for_repair_acceptance=True)
    write(output/'report.json',metric);write(output/'complete.json',dict(complete=True,arm=arm,summary_versions=200,critical_values_version=CRITICAL_VERSION))
    print('QUALITY COMPLETE',arm,metric['changed_summaries'],'changed summaries;',metric['actually_inserted_windows'],'accepted windows',flush=True)


def main():
    start=time.monotonic();nli=NLI();calibration(nli)
    write(ROOT/'outputs/setup/nli.json',dict(model_load_and_control_seconds=time.monotonic()-start))
    pending=list(ARMS)
    while pending:
        for arm in pending[:]:
            if (ROOT/'outputs/translation'/arm/'complete.json').exists():
                evaluate_arm(arm,nli);pending.remove(arm)
        if pending:time.sleep(2)


if __name__=='__main__':main()
