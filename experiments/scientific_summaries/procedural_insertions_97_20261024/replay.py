"""Hash-check saved benchmark evidence and all QA slots without model inference."""
import argparse,hashlib,json,statistics,tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def digest(data):return hashlib.sha256(data).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 root=args.out.resolve();root.mkdir(parents=True,exist_ok=True);manifest=load(HERE/'manifest.json')
 for bundle in manifest['bundles']:
  path=(HERE/bundle['path']).resolve();assert digest(path.read_bytes())==bundle['sha256'],path
  with tarfile.open(path,'r:gz') as tar:
   for member in tar:
    target=(root/member.name).resolve();assert member.isfile() and target.is_relative_to(root),member.name
    data=tar.extractfile(member).read();assert digest(data)==manifest['files'][member.name],member.name
    if target.exists():assert target.read_bytes()==data,target
    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 rows=[json.loads(l) for l in (root/'common/cohort97.jsonl').read_text().splitlines()];assert len(rows)==97
 for row in rows:assert digest(row['fulltext'].encode())==row['source_sha256']
 gold={(p['document_id'],q['question_index']):q['answer'] for p in load(root/'common/testset.json')['papers'] for q in p['questions']};assert len(gold)==970
 slots=0;checks=[]
 for run in manifest['runs']:
  base=root/run;qa=load(base/'outputs/qa_results.json');protocol=load(base/'inputs/protocol.json')
  for name,value in protocol.get('code_sha256',{}).items():assert digest((base/'code'/name).read_bytes())==value,(run,name)
  for condition,score in qa['scores'].items():
   answers=load(base/f'outputs/qa/{condition}.json');assert len(answers)==970
   assert len({(a['document_id'],a['question_index']) for a in answers})==970
   assert all(a['gold']==gold[a['document_id'],a['question_index']] for a in answers)
   assert sum(a['prediction']==a['gold'] for a in answers)==score['correct']
   contexts={};coverage=[]
   for row in rows:
    saved=load(base/f'outputs/repaired/{condition}/{row["uid"]}.json');assert saved['source_sha256']==row['source_sha256']
    contexts[row['document_id']]=digest(saved['judge_context'].encode())
    coverage.append(saved['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'])
   assert all(a['context_sha256']==contexts[a['document_id']] for a in answers)
   expected=load(base/f'outputs/{condition}_complete.json')['mean_coverage'];assert abs(statistics.mean(coverage)-expected)<1e-12
   checks.append(dict(run=run,condition=condition,qa=score['correct'],coverage=expected));slots+=970
  audit=[json.loads(l) for l in (base/'outputs/overlap/copy_overlap_details.jsonl').read_text().splitlines()]
  assert len(audit)==len(qa['scores'])*97
  assert all(a['audit'] and 'expanded_word_diagnostic' in a['audit'] for a in audit)
  print('VERIFIED',run,len(qa['scores'])*970,'QA slots',flush=True)
 assert slots==manifest['raw_qa_slots']
 result=dict(passed=True,papers=97,raw_qa_slots=slots,conditions=checks,source_gold_context_and_code_hashes_checked=True,saved_primary_and_expanded_audits_checked=True,qa_recomputed=False)
 (root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',slots,'saved QA slots')

if __name__=='__main__':main()
