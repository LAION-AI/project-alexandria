"""Finish the last malformed citation ranking without changing scientific narrative."""
import concurrent.futures
import copy
import hashlib
import json
import shutil
from common import *
from run_cohort import review
from orchestrate import start_server,stop,OWNED,status
from rescue import SUMMARY_PROPERTIES
from summary_runtime import REPAIR_SYSTEM
from score import score_all

class CitationClient(Client):
    def post(self,route,payload,timeout=1800):
        if route=='/v1/chat/completions' and payload['messages'][0]['content']==REPAIR_SYSTEM:
            schema=dict(type='object',properties=dict(top_influential_citations=SUMMARY_PROPERTIES['top_influential_citations']),
                required=['top_influential_citations'],additionalProperties=False)
            payload['response_format']=dict(type='json_schema',json_schema=dict(name='ranked_citations',strict=True,schema=schema))
        return super().post(route,payload,timeout)

def main():
    identifier='arxiv-500010'
    paper=next(p for p in load(ROOT/'inputs/eval_sources_only.json') if p['document_id']==identifier)
    directory=ROOT/'outputs/cohorts/ornith_dflash'
    path=directory/'documents'/identifier/'document.json'
    doc=load(path)
    if doc.get('corrected'):
        raise RuntimeError('The remaining paper was already completed; do not overwrite it')
    draft=copy.deepcopy(doc['structural_repair_failure']['draft_summary'])
    if set(invalid_fields(draft,paper['fulltext']))!={'top_influential_citations'}:
        raise ValueError('Recovery must affect only the remaining citation metadata')
    shutil.copyfile(path,path.with_name('before-guided-citation-recovery.json'))
    shutil.copyfile(ROOT/'outputs/report.md',ROOT/'outputs/report.before-last-citation.md')
    shutil.copyfile(ROOT/'outputs/finalized.json',ROOT/'outputs/finalized.before-last-citation.json')
    status('last_citation_schema_recovery',document_id=identifier)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        generator_future=executor.submit(start_server,0,'dflash8','summary-model')
        judge_future=executor.submit(start_server,1,'judge','qwen25')
        generator=generator_future.result()
        client=CitationClient(generator['endpoint'],'summary-model',path.parent/'calls')
        seed=int(digest(identifier+'|guided-citation-recovery')[:8],16)%2147483000
        user='BEGIN_PAPER\n'+paper['fulltext']+'\nEND_PAPER\n\nRepair only top_influential_citations. '
        user+='The previous output is a plain reference string, not the required ranked list. '
        user+='Return the field as a list of at most five dictionaries, each with one literal source citation key, '
        user+='a source-supported justification value and a quotes list of literal proof fragments of at most five words. '
        user+='If an influence ranking cannot be supported by the supplied source, use the empty string. '
        user+='Never output a bare reference string. No other fields.\nCURRENT_VALUE\n'+json.dumps(draft['top_influential_citations'],ensure_ascii=False)
        for attempt in range(3):
            record=client.generate(REPAIR_SYSTEM,user,4096,seed+attempt,phase='guided_citation_format_recovery')
            doc['attempts'].append(record)
            try:
                patch,_=parse_json(record['response'])
                if record['finish_reason']=='length' or set(patch)!={'top_influential_citations'}:
                    raise ValueError('Incomplete or incorrect citation patch')
                draft['top_influential_citations']=patch['top_influential_citations']
                summary,context,spans=validate_summary(json.dumps(draft,ensure_ascii=False),paper['fulltext'])
                break
            except (ValueError,TypeError,KeyError) as error:
                record['validation_error']=str(error)
                user+='\nVALIDATION_ERROR\n'+str(error)+'\nUse only actually verifiable citations and proof fragments.'
        else:
            raise RuntimeError('Guided citation recovery did not pass exact validation')
        # The final patch affects bibliography metadata, excluded by the historical renderer.
        if context!=narrative_context(doc['structural_repair_failure']['draft_summary']):
            raise ValueError('Citation repair changed scientific narrative')
        doc['corrected']=dict(summary=summary,judge_context=context,evidence_spans=spans,
            changed=summary!=doc['raw']['summary'],citation_schema_recovery=True)
        doc['corrected_quality_assessment']=review(paper,summary,client,doc['attempts'],'guided_citation_corrected_review')
        doc.setdefault('source_recovery_normalizations',[]).append(dict(
            operation='model_generated_schema_constrained_citation_patch',
            original_value=doc['structural_repair_failure']['draft_summary']['top_influential_citations'],
            request_trace=record.get('trace_file'),scientific_narrative_unchanged=True))
        doc.pop('error',None)
        write(path,doc)
        stop(generator)
        judge=judge_future.result()
    cache_path=directory/'corrected-summaries.json'
    cache=load(cache_path)
    cache['documents'].append(dict(document_id=identifier,fulltext_sha256=paper['fulltext_sha256'],**doc['corrected']))
    cache['failures']=[f for f in cache['failures'] if f['document_id']!=identifier]
    sources=load(ROOT/'inputs/eval_sources_only.json')
    by_id={d['document_id']:d for d in cache['documents']}
    if len(by_id)!=97 or set(by_id)!={p['document_id'] for p in sources}:
        raise ValueError('Final corrected cohort must contain all 97 unique papers')
    cache['documents']=[by_id[p['document_id']] for p in sources]
    cache['config']['last_citation_recovery']='source-only constrained ranked-citation JSON; no narrative edits'
    write(cache_path,cache)
    completion=load(directory/'complete.json');completion['corrected']=97;write(directory/'complete.json',completion)
    report=score_all(judge['endpoint'])
    report['supplemental_throughput_benchmark']=load(ROOT/'outputs/throughput/dflash4.json')
    report['qa_speculative_configuration']='dflash8'
    write(ROOT/'outputs/report.json',report)
    prior=(ROOT/'outputs/report.before-last-citation.md').read_text()
    if '## Measured output throughput per GH200 GPU' in prior:
        with (ROOT/'outputs/report.md').open('a') as handle:
            handle.write('\n## Measured output throughput per GH200 GPU'+prior.split('## Measured output throughput per GH200 GPU',1)[1])
    stop(judge)
    write(ROOT/'outputs/finalized.json',dict(papers=97,questions=970,
        completion={model:load(ROOT/'outputs/cohorts'/model/'complete.json') for model in ('ornith_dflash','qwen38_fp8')}))
    status('complete',papers=97,questions=970,all_summaries_and_corrections_complete=True)

if __name__=='__main__':
    try:
        main()
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
