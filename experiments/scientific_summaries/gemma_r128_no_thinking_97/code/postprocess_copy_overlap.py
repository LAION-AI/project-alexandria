"""Add current overlap evidence to completed native evaluations without changing QA."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
from copy_overlap_eval import audit_conditions
from ngram_overlap import VERSION

PINNED=Path('/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a/experiments/scientific_summaries/data/testset.json')


def update(root):
    output=Path(root)/'outputs/evaluation'
    if not (output/'complete.json').exists():return None
    with (output/'.copy_overlap.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _update(root)


def _update(root):
    output=Path(root)/'outputs/evaluation'
    if not (output/'complete.json').exists():return None
    report=json.loads((output/'report.json').read_text())
    existing=output/'copy_overlap.json'
    cached=json.loads(existing.read_text()) if existing.exists() else {}
    if cached.get('audit_version')==VERSION and all('all_fields_strict_five_word_pass' in v for v in cached.get('conditions',{}).values()):
        overlap=cached
    else:
        assert hashlib.sha256(PINNED.read_bytes()).hexdigest()=='a0d5e5f99a0025c6cd8a5140a39a07a220ded6f37f04994500549886ffd83261'
        papers=json.loads(PINNED.read_text())['papers']
        contexts={label:{p['document_id']:json.loads((output/'generation'/label/p['document_id']/'result.json').read_text()) for p in papers} for label in report['models']}
        overlap=audit_conditions(papers,contexts,output)
    report['copy_overlap']=overlap
    report['source_copy_compliance_separate_from_qa']=True
    report['scaling_counts_structurally_finished_outputs_not_paraphrase_compliant_outputs']=True
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    complete=json.loads((output/'complete.json').read_text())
    complete.update(copy_overlap_checked=True,copy_overlap_version=VERSION,
        copy_violations_do_not_remove_papers_from_qa=True)
    (output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    return overlap


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    value=update(parser.parse_args().root)
    print(json.dumps(dict(updated=value is not None,audit_version=VERSION)))
