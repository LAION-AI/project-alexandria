"""Source/QA-blind diagnostic of apparent copying reductions from surface edits.

Does not alter any summary, score, or acceptance decision in the frozen run.
"""
import difflib
from collections import Counter
from common import ROOT,CONDITIONS,read_jsonl,write,jsonl
from ngram_overlap import tokenize

def one_edit(a,b):
 if abs(len(a)-len(b))>1:return False
 if len(a)==len(b):
  positions=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
  return len(positions)==1 or (len(positions)==2 and positions[1]==positions[0]+1 and a[positions[0]]==b[positions[1]] and a[positions[1]]==b[positions[0]])
 if len(a)>len(b):a,b=b,a
 i=j=0;skips=0
 while i<len(a) and j<len(b):
  if a[i]==b[j]:i+=1;j+=1
  else:j+=1;skips+=1
  if skips>1:return False
 return True

def main():
 from wordfreq import zipf_frequency
 summary={};examples=[]
 for c in CONDITIONS:
  p=ROOT/f'outputs/final_edits/{c}.jsonl'
  if not p.exists():continue
  rows=read_jsonl(p);flags=[]
  for r in rows:
   before=tokenize(r['before'],expanded=True)[1];after=tokenize(r['after'],expanded=True)[1]
   reasons=[];suspects=[]
   if before==after:reasons.append('identical_punctuation_split_word_sequence')
   for tag,a,b,d,e in difflib.SequenceMatcher(a=before,b=after,autojunk=False).get_opcodes():
    if tag=='replace' and b-a==e-d==1:
     old,new=before[a],after[d]
     if min(len(old),len(new))>=4 and one_edit(old,new) and zipf_frequency(old,'en')>=2 and zipf_frequency(new,'en')==0:
      suspects.append(dict(old=old,new=new,old_zipf=zipf_frequency(old,'en'),new_zipf=0));reasons.append('new_unlisted_word_one_character_from_original')
   if reasons:flags.append(dict(r,surface_flags=sorted(set(reasons)),suspected_spelling_changes=suspects))
  co=Counter(reason for r in flags for reason in r['surface_flags'])
  summary[c]=dict(final_edited_windows=len(rows),flagged_windows=len(flags),counts=dict(co))
  jsonl(ROOT/f'outputs/surface_diagnostics/{c}.jsonl',flags);examples.extend(flags[:4])
 report=dict(complete=True,conditions=summary,examples=examples,
  definition='Orthographic-only = exactly identical punctuation-split normalized token sequence before/after; whitespace/hyphen changes can nevertheless reduce the primary whitespace-word overlap metric. Probable typo = one-character edit or adjacent transposition introducing a word with zero wordfreq frequency from an original word with Zipf >=2.',
  limits='A probable-typo flag is heuristic, not a spellchecker verdict or scientific meaning label. Orthographic-only flags are not meaningful lexical paraphrases. Original summaries, QA, and copying metrics remain unchanged; this is a separate diagnostic, not post-QA filtering.')
 write(ROOT/'outputs/surface_diagnostic.json',report)
 print({c:r['counts'] for c,r in summary.items()},flush=True)
if __name__=='__main__':main()
