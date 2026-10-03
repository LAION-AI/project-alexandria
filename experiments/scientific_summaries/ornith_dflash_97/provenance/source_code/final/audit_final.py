import hashlib
import json
import statistics
from common import ROOT,VENDOR,load,write,digest,narrative_context,validate_summary
from project_alexandria.experiments.mcq import historical_answer_prompt,extract_historical_choice

def main():
    bundle=load(VENDOR/'data/testset.json')
    papers={p['document_id']:p for p in bundle['papers']}
    if len(papers)!=97 or sum(len(p['questions']) for p in papers.values())!=970:
        raise ValueError('Benchmark cohort changed')
    qa=load(ROOT/'outputs/qa-results.json')
    if len(qa['documents'])!=97 or {d['document_id'] for d in qa['documents']}!=set(papers):
        raise ValueError('Evaluated cohort changed')
    caches={}
    summaries={}
    for model,prefix in [('ornith_dflash','ornith'),('qwen38_fp8','qwen38')]:
        for condition in ('raw','corrected'):
            cache=load(ROOT/'outputs/cohorts'/model/(condition+'-summaries.json'))
            items={d['document_id']:d for d in cache['documents']}
            if len(cache['documents'])!=97 or set(items)!=set(papers) or cache['failures']:
                raise ValueError('All 97 summary/correction outputs required')
            words=[]
            spans=0
            for identifier,item in items.items():
                paper=papers[identifier]
                if item['fulltext_sha256']!=paper['fulltext_sha256'] or digest(paper['fulltext'])!=paper['fulltext_sha256']:
                    raise ValueError('Source checksum mismatch')
                if narrative_context(item['summary'])!=item['judge_context']:
                    raise ValueError('Student narrative changed')
                words.append(len(item['judge_context'].split()))
                if condition=='corrected':
                    _,text,ledger=validate_summary(json.dumps(item['summary'],ensure_ascii=False),paper['fulltext'])
                    if text!=item['judge_context']:
                        raise ValueError('Validated narrative differs')
                    actual=[(x['path'],x['quote'],x['start'],x['end']) for x in item['evidence_spans']]
                    expected=[(x['path'],x['quote'],x['start'],x['end']) for x in ledger]
                    if actual!=expected:
                        raise ValueError('Source evidence ledger differs')
                    spans+=len(ledger)
            label=prefix+'_'+condition
            caches[label]=items
            summaries[label]=dict(papers=97,mean_narrative_words=statistics.mean(words),
                median_narrative_words=statistics.median(words),min_narrative_words=min(words),
                max_narrative_words=max(words),verified_source_spans=spans if condition=='corrected' else None)
    reference={d['document_id']:d for d in load(VENDOR/'summary_runs/qwen27b/summaries.json')['documents']}
    prompt_checks=0
    prediction_checks=0
    for document in qa['documents']:
        paper=papers[document['document_id']]
        if document['questions']!=paper['questions'] or document['fulltext_sha256']!=paper['fulltext_sha256']:
            raise ValueError('Immutable MCQs changed')
        contexts=dict(no_context='',original=paper['fulltext'],summary=paper['existing_summary'],
            qwen38_reference=reference[paper['document_id']]['judge_context'])
        contexts.update({label:items[paper['document_id']]['judge_context'] for label,items in caches.items()})
        if set(document['conditions'])!=set(contexts):
            raise ValueError('Evaluation conditions differ')
        for label,context in contexts.items():
            value=document['conditions'][label]
            if value.get('generation_failed') or value['context_sha256']!=digest(context) or len(value['rows'])!=10:
                raise ValueError('Stale context or missing evaluation')
            canonical=label if label in ('no_context','original','summary') else 'qwen_summary'
            for question,row in zip(paper['questions'],value['rows']):
                response=row['responses'][canonical]
                prompt=historical_answer_prompt(question['formatted_question'],context)
                if response['prompt_sha256']!=digest(prompt) or response['prompt_tokens']+100>32768:
                    raise ValueError('QA prompt mismatch or truncated context')
                if row['gold']!=question['answer'] or not 1<=len(response['attempts'])<=5:
                    raise ValueError('Gold or retry policy differs')
                expected=extract_historical_choice(response['attempts'][-1])
                if row['predictions'][canonical]!=expected:
                    raise ValueError('Prediction is not derived from historical parser')
                prompt_checks+=1;prediction_checks+=1
    result=dict(passed=True,papers=97,mcqs=970,conditions=8,qa_prompt_hash_checks=prompt_checks,
        prediction_parser_checks=prediction_checks,summaries=summaries,
        heldout_outputs_location=str(ROOT),training_contamination='evaluation artifacts kept outside training workspace',
        testset_sha256=hashlib.sha256((VENDOR/'data/testset.json').read_bytes()).hexdigest())
    write(ROOT/'outputs/final_audit.json',result)
    report=load(ROOT/'outputs/report.json');report['final_audit']=result;write(ROOT/'outputs/report.json',report)
    with (ROOT/'outputs/report.md').open('a') as handle:
        handle.write('\n## Narrative lengths and final audit\n\n')
        handle.write('All 97 raw and corrected outputs are present for both models. All corrected source spans and all 7,760 QA prompt hashes and parsed predictions pass the final audit.\n\n')
        handle.write('| Condition | Mean narrative words | Median narrative words |\n|---|---:|---:|\n')
        for label,value in summaries.items():
            handle.write('| '+label+' | '+str(round(value['mean_narrative_words']))+' | '+str(value['median_narrative_words'])+' |\n')
        handle.write('\nOrnith produces substantially longer summaries under the shared initial prompt. This comparison measures the complete deployed pipelines; output lengths were not equalized. The inference recipe disables thinking. Self-audit scores are not independent human quality labels.\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
