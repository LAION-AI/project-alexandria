"""Verify saved artifacts without repeating a cohort source-copy audit on login."""
import hashlib,json,statistics,sys,re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/e/scratch/reformo/schuhmann1/scientific-post-gemma-insertion-system-v6-97-20261024')
sys.path.insert(0,str(ROOT/'code'))
from common import CONDITIONS,load,read_jsonl,digest
from ngram_overlap import fragments,tokenize
import insertions as editor

def main():
 protocol=load(ROOT/'inputs/protocol.json')
 assert all(hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==value for name,value in protocol['code_sha256'].items())
 rows=read_jsonl(ROOT/'inputs/cohort97.jsonl');assert len(rows)==97 and len({r['document_id'] for r in rows})==97
 assert all(digest(r['fulltext'])==r['source_sha256'] for r in rows)
 gold={(p['document_id'],q['question_index']):q['answer'] for p in load(ROOT/'inputs/testset.json')['papers'] for q in p['questions']};assert len(gold)==970
 audit_rows=read_jsonl(ROOT/'outputs/overlap/copy_overlap_details.jsonl');assert len(audit_rows)==len(CONDITIONS)*97
 assert len({(a['condition'],a['document_id']) for a in audit_rows})==len(audit_rows)
 by_doc={r['document_id']:r for r in rows}
 for a in audit_rows:
  assert a['source_sha256']==by_doc[a['document_id']]['source_sha256']
  assert a['audit']['source_sha256']==a['source_sha256']
  assert 'expanded_word_diagnostic' in a['audit']
 nslots=0;trace_count=0;protected_phrase_checks=0;clinical_frame_checks=0;results=load(ROOT/'outputs/results_and_scaling.json')
 for c in CONDITIONS:
  contexts={r['document_id']:load(ROOT/f'outputs/repaired/{c}/{r["uid"]}.json') for r in rows}
  q=load(ROOT/f'outputs/qa/{c}.json');assert len(q)==970 and len({(x['document_id'],x['question_index']) for x in q})==970
  for a in q:
   assert a['gold']==gold[a['document_id'],a['question_index']]
   assert a['context_sha256']==digest(contexts[a['document_id']]['judge_context'])
  assert sum(a['prediction']==a['gold'] for a in q)==results['qa']['scores'][c]['correct'];nslots+=len(q)
  for row in rows:
   base=load(ROOT/f'outputs/repaired/baseline/{row["uid"]}.json');out=contexts[row['document_id']]
   assert out['source_sha256']==row['source_sha256']
   a=list(fragments(base['summary']));b=list(fragments(out['summary']));assert len(a)==len(b)
   for (x,p,k),(y,z,l) in zip(a,b):
    assert (p,k)==(z,l)
    if k!='narrative':assert x==y
   if c not in CONDITIONS[2:]:continue
   traces=read_jsonl(ROOT/f'traces/{c}/{row["uid"]}.jsonl')
   for (x,p,k),(y,z,l) in zip(a,b):
    if k!='narrative':continue
    if x!=y:assert any(t['path']==p and t['before']==x and t['after']==y for t in traces)
    before_words=tokenize(x)[1];after_words=tokenize(y)[1]
    for phrase in editor.FIXED_TERMS:
     term=tuple(tokenize(phrase)[1]);size=len(term)
     occurrences=lambda words:sum(tuple(words[i:i+size])==term for i in range(len(words)-size+1))
     n=occurrences(before_words)
     if n:assert occurrences(after_words)>=n;protected_phrase_checks+=1
   if 'protects tumor cells from CTL killing' in base['judge_context']:
    # A valid exact definition of CTL may appear as a parenthetical gloss;
    # the governing protects…from frame itself must remain contiguous.
    assert re.search(r'protects tumor cells from CTL(?: \(cytotoxic T lymphocyte\))? killing',out['judge_context']);clinical_frame_checks+=1
   assert 'JigSAP (jigsaw-shaped hydrophobic surfaces)' not in out['judge_context']
   for t in traces:
    assert editor.render(t['before'],t['edits'])==t['after']
    assert editor.compatible(t['edits']) and editor.scientific_guard(t['before'],t['after'])
    assert t['metrics_after']['original_covered']<t['metrics_before']['original_covered']
    assert t['metrics_after']['covered']<t['metrics_before']['covered']
    assert t['metrics_after']['hits']<t['metrics_before']['hits']
    if not out['rolled_back']:assert any(p==z==t['path'] and x==t['before'] and y==t['after'] for (x,p,k),(y,z,l) in zip(a,b))
    trace_count+=1
  if c in CONDITIONS[2:]:
   per=[a for a in audit_rows if a['condition']==c]
   mean=statistics.mean(a['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'] for a in per)
   assert abs(mean-results['conditions'][c]['overlap']['narrative']['mean_covered_fraction_six_plus'])<1e-12
 value=dict(passed=True,papers=97,raw_qa_slots_verified=nslots,full_source_hashes_checked=True,primary_and_punctuation_split_saved_audits_checked=len(audit_rows),unchanged_code_hashes=len(protocol['code_sha256']),verified_edit_plans=trace_count,protected_evidence_metadata=True,protected_phrase_occurrence_checks=protected_phrase_checks,actual_clinical_from_frame_checks=clinical_frame_checks,rejected_custom_material_gloss_absent=True,published_to_github=False,scope='Saved-artifact, hash, prediction and trace verification; full cohort source-copy audits ran within Slurm allocation.')
 (HERE/'verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
