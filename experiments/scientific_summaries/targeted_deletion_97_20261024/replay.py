"""Verify and unpack saved evidence; check every original QA slot without GPUs."""
import argparse,hashlib,json,statistics,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    root=args.out.resolve();manifest=json.loads((HERE/'manifest.json').read_text())
    for bundle in manifest['bundles']:
        path=HERE/bundle['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==bundle['sha256']
        with tarfile.open(path,'r:gz') as archive:
            for member in archive:
                target=(root/member.name).resolve()
                assert member.isfile() and target.is_relative_to(root),member.name
                data=archive.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==manifest['files'][member.name]
                if target.exists():assert target.read_bytes()==data,target
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    rows=[json.loads(l) for l in (root/'common/cohort97.jsonl').read_text().splitlines()]
    assert len(rows)==97
    for row in rows:assert hashlib.sha256(row['fulltext'].encode()).hexdigest()==row['source_sha256']
    gold={(p['document_id'],q['question_index']):q['answer'] for p in json.loads((root/'common/testset.json').read_text())['papers'] for q in p['questions']}
    checks=[];slots=0
    for run in ['initial','case_guard','modifiers_only','numbers','numbers_guarded']:
        base=root/run;qa=json.loads((base/'outputs/qa_results.json').read_text())
        for condition,score in qa['scores'].items():
            answers=json.loads((base/f'outputs/qa/{condition}.json').read_text())
            assert len(answers)==970 and len({(a['document_id'],a['question_index']) for a in answers})==970
            assert all(a['gold']==gold[a['document_id'],a['question_index']] for a in answers)
            assert sum(a['prediction']==a['gold'] for a in answers)==score['correct']
            contexts={};coverage=[]
            for row in rows:
                value=json.loads((base/f'outputs/repaired/{condition}/{row["uid"]}.json').read_text())
                assert value['source_sha256']==row['source_sha256']
                contexts[row['document_id']]=hashlib.sha256(value['judge_context'].encode()).hexdigest()
                coverage.append(value['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'])
            assert all(a['context_sha256']==contexts[a['document_id']] for a in answers)
            expected=json.loads((base/f'outputs/{condition}_complete.json').read_text())['mean_coverage']
            assert abs(statistics.mean(coverage)-expected)<1e-12
            checks.append(dict(run=run,condition=condition,qa=score['correct'],coverage=expected));slots+=970
            print(run,condition,score['correct'],f'{expected:.6%}')
    result=dict(passed=True,papers=97,raw_qa_slots=slots,conditions=checks,all_source_hashes_checked=True,all_gold_and_context_hashes_checked=True)
    (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS:',slots,'saved QA slots; all97 full-source hashes and every evaluated context verified.')


if __name__=='__main__':main()
