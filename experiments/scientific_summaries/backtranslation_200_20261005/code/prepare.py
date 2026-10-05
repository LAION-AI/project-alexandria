"""Freeze 200 existing no-thinking versions and disjoint training-paper probes."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
from common import ROOT,BASE,PINNED,REFERENCE,TEST_SHA,load,write,jsonl,sha
from passages import select


def main():
    start=time.monotonic()
    testfile=PINNED/'data/testset.json'
    assert hashlib.sha256(testfile.read_bytes()).hexdigest()==TEST_SHA
    papers=load(testfile)['papers'];byid={p['document_id']:p for p in papers}
    assert len(papers)==97
    cohort=[];windows=[]
    choices=[('no_thinking_fp8',papers),('no_thinking_merged',papers),
             ('no_thinking_matched',sorted(papers,key=lambda p:sha(p['document_id']))[:6])]
    for condition,selected in choices:
        for p in selected:
            path=REFERENCE/'outputs/evaluation/generation'/condition/p['document_id']/'result.json'
            generated=load(path);assert generated['status']=='generated'
            assert sha(p['fulltext'])==p['fulltext_sha256']
            uid=condition+'__'+p['document_id']
            item=dict(uid=uid,document_id=p['document_id'],original_condition=condition,
                      source_sha256=p['fulltext_sha256'],fulltext=p['fulltext'],summary=generated['summary'],
                      judge_context=generated['judge_context'],original_file=str(path),
                      original_summary_sha256=sha(json.dumps(generated['summary'],sort_keys=True,ensure_ascii=False)))
            selected_windows=select(item['summary'],item['fulltext'])
            for j,w in enumerate(selected_windows):
                windows.append(dict(w,uid=uid,document_id=p['document_id'],window_id=uid+'__w'+str(j)))
            item['window_ids']=[w['window_id'] for w in windows[-len(selected_windows):]] if selected_windows else []
            cohort.append(item)
    assert len(cohort)==200 and len({x['uid'] for x in cohort})==200
    assert len({x['document_id'] for x in cohort})==97
    jsonl(ROOT/'inputs/cohort.jsonl',cohort)
    jsonl(ROOT/'inputs/windows.jsonl',windows)
    # Tune speed only on teacher training papers outside all 97 evaluation sources.
    training=[]
    with (BASE/'scientific-distillation-865-20261004/release/data/train.jsonl').open() as handle:
        for line in handle:
            value=json.loads(line)
            assert value['document_id'] not in byid
            training.append(value)
    probes=[]
    for item in sorted(training,key=lambda x:sha(x['document_id'])):
        summary=json.loads(item['validated_final_summary_json'])
        for w in select(summary,item['fulltext'])[:4]:
            if 15<=len(w['text'].split())<=120:
                probes.append(dict(text=w['text'],document_id=item['document_id'],source_sha256=item['source_sha256']))
        if len(probes)>=512:break
    probes=probes[:512]
    assert len(probes)==512 and not {p['document_id'] for p in probes}&set(byid)
    jsonl(ROOT/'inputs/probes.jsonl',probes)
    # English original fragments, not questions/gold, are the candidate inputs.
    protocol=dict(summary_versions=200,unique_papers=97,original_conditions={k:len(v) for k,v in choices},
                  testset_sha256=TEST_SHA,selection='All 97 FP8 and all 97 tuned BF16 outputs plus six live-BF16 outputs selected by document-ID SHA256 order',
                  source_copy_allowed_words=5,six_words_borderline=True,window_selection_minimum_run=6,
                  preceding_sentence_included=True,max_window_sentence_chunks=2,max_chunk_whitespace_words=75,
                  no_source_or_summary_truncation=True,reasoning_and_qa_not_in_translation_inputs=True,
                  metadata_and_evidence_quotes_not_backtranslated=True,
                  rounds=1,pivot='de',retry_policy='Only numerical/runtime failures; no paraphrase retry loop in this benchmark',
                  guarded_acceptance=dict(critical_values_preserved=True,bidirectional_nli_entailment_minimum=.9,
                                          candidate_window_source_overlap_maximum_words=5),
                  nli_is_an_experimental_filter_not_a_factuality_guarantee=True,
                  speed_optimization_probes=len(probes),speed_optimization_papers=len({p['document_id'] for p in probes}),
                  performance_tuning_does_not_use_heldout_qa=True,
                  translation_window_count=len(windows),selected_translation_words=sum(len(w['text'].split()) for w in windows),
                  prepare_elapsed_seconds=time.monotonic()-start,
                  code_sha256={p.name:sha(p.read_text()) for p in (ROOT/'code').glob('*.py')})
    write(ROOT/'inputs/protocol.json',protocol)
    print('Prepared',len(cohort),'no-thinking summaries,',len(windows),'sentence windows,',len(probes),'disjoint probes',flush=True)


if __name__=='__main__':main()
