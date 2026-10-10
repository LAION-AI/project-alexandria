# SPDX-License-Identifier: CC-BY-4.0
# Copyright 2026 LAION / Project Alexandria contributors.
import time
from common import ROOT,write
from critical_values import check
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

