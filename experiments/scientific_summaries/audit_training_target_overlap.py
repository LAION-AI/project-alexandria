"""Audit the exact published raw generation targets used for both LoRA datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from ngram_overlap import SourceIndex,audit_summary,VERSION
from copy_overlap_eval import compact,markdown

BASE=Path('/e/fscratch/reformo/schuhmann1')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True);rows=[];start=time.monotonic()
    with (output/'per_summary.jsonl').open('w') as details:
        for run,teacher in [('scientific-distillation-865-20261004','Qwen27'),('ornith-distillation-736-20261004','Ornith9B')]:
            source=BASE/run/'release/data/train.jsonl'
            with source.open() as handle:
                for lineno,line in enumerate(handle,1):
                    item=json.loads(line);index=SourceIndex(item['fulltext'])
                    assert item['source_sha256']==index.sha256
                    views={'raw generator training target':json.loads(item['raw_generator_summary'])}
                    if teacher=='Qwen27':views['validated corrected final (separate view)']=json.loads(item['validated_final_summary_json'])
                    for stage,summary in views.items():
                        row=dict(condition=teacher+' / '+stage,document_id=item['document_id'],source_sha256=index.sha256,
                            summary_sha256=hashlib.sha256(json.dumps(summary,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                            audit=audit_summary(item['fulltext'],summary,index=index),location=str(source)+':'+str(lineno))
                        details.write(json.dumps(row,ensure_ascii=False)+'\n');rows.append(row)
                    if lineno%200==0:print(teacher,lineno,'papers audited',flush=True)
    labels=sorted({r['condition'] for r in rows})
    report=dict(audit_version=VERSION,complete=True,elapsed_seconds=time.monotonic()-start,
        conditions={label:compact([r for r in rows if r['condition']==label]) for label in labels},
        training_targets_unchanged=True,published_hf_releases_unchanged=True,reasoning_not_summary_prose=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'RESULTS.md').write_text('# Published distillation targets: exact summary-copy audit\n\n'+markdown(report))
    fields=['condition','document_id','source_sha256','narrative_longest','narrative_coverage_6plus','narrative_coverage_7plus','evidence_longest','evidence_fragments_over_five_words','location']
    with (output/'per_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in rows:
            n=r['audit']['categories']['narrative'];e=r['audit']['categories']['evidence']
            writer.writerow(dict(condition=r['condition'],document_id=r['document_id'],source_sha256=r['source_sha256'],
                narrative_longest=n['longest_contiguous_match_words'],narrative_coverage_6plus=n['covered_word_fraction_by_minimum_run']['6'],
                narrative_coverage_7plus=n['covered_word_fraction_by_minimum_run']['7'],evidence_longest=e['longest_contiguous_match_words'],
                evidence_fragments_over_five_words=e['evidence_fragments_over_five_words'],location=r['location']))
    print(json.dumps(dict(complete=True,summary_views=len(rows),elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
