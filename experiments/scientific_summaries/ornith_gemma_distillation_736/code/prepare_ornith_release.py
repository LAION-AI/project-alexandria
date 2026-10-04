"""Release frozen native-AR Ornith outputs and render their exact generator SFT targets."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import tarfile

from common import ROOT,SOURCE,load,write,digest,jsonl
sys.path.insert(0,str(SOURCE/'code'))
from holdout_guard import Guard
from render_training import render
from run import json_object
from transformers import AutoTokenizer

ORIGIN=Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004')
SECRET=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
TEACHER=dict(model='ornith-ai/Ornith-1.5-9B',revision='489cb97981b8654bcfcf30ce1f94ed1b62e07b53',precision='BF16',speculative_decoding=False)


def release_only():
    cohort=load(ROOT/'inputs/frozen_cohort.json');assert cohort['generated_papers']==736
    sources={p['document_id']:p for p in load(SOURCE/'inputs/papers.json')}
    prompts=load(ROOT/'inputs/reference_prompts.json')
    selected=[sources[p['document_id']] for p in cohort['papers']]
    # The frozen subset uses identical source bytes from the already audited 865 cohort.
    original_cohort=load(ORIGIN/'inputs/frozen_cohort.json')
    original_by_id={p['document_id']:p for p in original_cohort['papers']}
    assert all(original_by_id[p['document_id']]['source_sha256']==p['source_sha256'] for p in cohort['papers'])
    audit=load(ORIGIN/'release/evaluation_overlap_audit.json')
    audit=dict(audit,subset_papers=736,subset_original_papers=865,
               subset_audit_method='Exact frozen source-hash subset of the previously audited zero-overlap collection',
               original_audit_sha256=digest(ORIGIN/'release/evaluation_overlap_audit.json'))
    write(ROOT/'release/evaluation_overlap_audit.json',audit)
    counts=[];rows=[];conversations=[];native=[];budgets=collections.Counter();schema=collections.Counter()
    for item in cohort['papers']:
        source=sources[item['document_id']];path=Path(item['generation_result_path'])
        assert digest(path)==item['generation_result_sha256']
        result=load(path);assert result['status']=='generated'
        assert hashlib.sha256(source['fulltext'].encode()).hexdigest()==item['source_sha256']
        call_path=path.parent/('attempt'+str(result['attempts']-1)+'.json');call=load(call_path)
        choice=call['response']['choices'][0];assistant=choice['message']
        reasoning=assistant.get('reasoning_content',assistant.get('reasoning'))
        assert choice['finish_reason']=='stop' and reasoning and reasoning.strip()
        assert reasoning==result['reasoning_content'] and isinstance(assistant['content'],str)
        assert json_object(assistant['content'])==result['summary']
        messages=call['request']['messages']
        assert len(messages)==2 and messages[0]['role']=='system' and messages[1]['role']=='user'
        assert source['fulltext'] in messages[1]['content']
        expected_user=prompts['summary-user'].replace('{paper_text}',source['fulltext'])
        assert messages[1]['content'].startswith(expected_user)
        suffix=messages[1]['content'][len(expected_user):]
        assert not suffix or suffix.startswith('\n\nFormat error from the preceding attempt: ')
        budget=call['request'].get('thinking_token_budget');budgets['uncapped' if budget is None else str(budget)]+=1
        schema[str(result['strict_source_schema_validation_passed'])]+=1
        conversation=dict(document_id=item['document_id'],domain=item['domain'],split='train',
            source_sha256=item['source_sha256'],teacher=TEACHER,call_id='generation-'+str(result['attempts']-1),
            task='source_only_summary_generation_with_actual_reasoning',reasoning_in_target=True,
            semantic_correction=False,strict_source_schema_validation_passed=result['strict_source_schema_validation_passed'],
            messages=messages+[dict(role='assistant',content=assistant['content'],reasoning_content=reasoning)])
        conversations.append(conversation)
        rows.append(dict(document_id=item['document_id'],domain=item['domain'],split='train',
            source_sha256=item['source_sha256'],fulltext=source['fulltext'],source_metadata_json=json.dumps(source,ensure_ascii=False),
            teacher_model=TEACHER['model'],teacher_revision=TEACHER['revision'],teacher_precision='BF16',
            speculative_decoding=False,raw_generator_summary=assistant['content'],raw_generator_reasoning=reasoning,
            summary_narrative=result['judge_context'],thinking_token_budget=budget,
            strict_source_schema_validation_passed=result['strict_source_schema_validation_passed'],
            source_validation_error=result['source_validation_error'],
            actual_request_json=json.dumps(call['request'],ensure_ascii=False),
            actual_response_json=json.dumps(call['response'],ensure_ascii=False),
            generation_result_json=json.dumps(result,ensure_ascii=False)))
    jsonl(ROOT/'release/data/train.jsonl',rows)
    jsonl(ROOT/'release/conversations/generation_reasoning.jsonl',conversations)
    release=ROOT/'release';provenance=release/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['frozen_cohort.json','reference_prompts.json','selection_manifest.json']:
        shutil.copy2(ROOT/'inputs'/name,provenance/name)
    (provenance/'summary_system_prompt.txt').write_text(conversations[0]['messages'][0]['content'])
    (provenance/'summary_user_prompt.txt').write_text(load(ROOT/'inputs/reference_prompts.json')['summary-user'])
    shutil.copytree(ORIGIN/'outputs/server_commands',provenance/'original_server_commands',dirs_exist_ok=True)
    shutil.copytree(ROOT/'code',provenance/'experiment_code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['run.py','reference_validator.py','render_training.py','holdout_guard.py']:
        shutil.copy2(SOURCE/'code'/name,provenance/name)
    archives={item['document_id']:[('final_generation',Path(item['generation_result_path']).parent)] for item in cohort['papers']}
    for name in ['ornith_initial_attempts','ornith_resume_attempts','ornith_retry_attempts']:
        base=ORIGIN/'outputs'/name
        if not base.exists():continue
        for folder in base.rglob('*'):
            if folder.is_dir() and folder.name in archives:
                archives[folder.name].append(('earlier_attempts/'+name+'/'+str(folder.relative_to(base)),folder))
    def archive(item):
        target=release/'traces'/(item['document_id']+'.jsonl');target.parent.mkdir(exist_ok=True)
        members=[]
        for prefix,folder in archives[item['document_id']]:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and not path.name.endswith('.tmp'):
                    payload=path.read_bytes()
                    assert not SECRET.search(payload), 'Potential credential in raw trace'
                    members.append(dict(path=prefix+'/'+str(path.relative_to(folder)),
                        sha256=hashlib.sha256(payload).hexdigest(),content=payload.decode('utf-8')))
        jsonl(target,members)
        return dict(document_id=item['document_id'],file=str(target.relative_to(release)),
            format='uncompressed JSONL archive: original UTF-8 content and each member SHA256',
            bytes=target.stat().st_size,sha256=digest(target))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:trace_index=list(pool.map(archive,cohort['papers']))
    jsonl(release/'trace_archives.jsonl',trace_index)
    manifest=dict(status='complete_frozen_collection',generated_papers=736,original_target_papers=865,
        stopped_by_user=True,teacher=TEACHER,domains=cohort['domains'],successful_papers_by_thinking_budget=dict(budgets),
        strict_source_schema_validation=dict(schema),quality_assessments_generated=False,semantic_correction=False,
        targets='actual raw source-only generator answer plus its matching emitted reasoning',
        external_evaluation=audit,dataset_attribution=load(ROOT/'inputs/selection_manifest.json')['attribution'],
        dataset_license='CC-BY-4.0; underlying source-paper rights remain with their authors',
        native_training_preparation='Prepared separately on a compute node using this frozen conversation view; published targets are unchanged',trace_archives=736,trace_archive_format='jsonl',
        mean_narrative_words=statistics.mean(len(r['summary_narrative'].split()) for r in rows))
    write(release/'manifest.json',manifest)
    card='''---
language:
- en
license: cc-by-4.0
task_categories:
- summarization
- text-generation
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# Ornith 1.5 9B: 736 scientific paper generator summaries and actual reasoning

Frozen on 2026-10-04 at the user request. These are 736 successful raw generator
outputs from the same source cohort as the 865-paper Qwen distillation collection.
The remaining 129 papers were stopped; there is no claim that all 865 were completed.
Ten scientific domains; all records use the train split. No evaluation MCQs or
gold answers are included. The external 97-paper/970-MCQ evaluation is excluded
by IDs, DOI, normalized titles, source hashes and substantial shared text; the
saved overlap audit reports zero matches.

## Teacher and exact generation recipe

Teacher: [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B),
revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, BF16 native autoregressive
decoding **without DFlash or other speculative decoding**. vLLM 0.30.0; one
model replica per GH200 GPU, tensor parallelism 1, eight replicas over two nodes,
16 concurrent source papers per GPU, max_num_seqs=64, max_num_batched_tokens=8192,
GPU memory utilization 0.90, prefix caching and chunked prefill enabled.
Full available untruncated source text; 65,536 context tokens, up to 24,576 total
completion tokens including reasoning. Source-only Qwen scientific summary
prompts, copied verbatim to `provenance/summary_system_prompt.txt` and
`provenance/summary_user_prompt.txt`. Actual per-call prompts are authoritative.
Thinking enabled, reasoning_effort medium, preserve_thinking true; temperature
1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0.
JSON-object response format. At most three format attempts under the collected
recipe; format-error messages and actual deterministic request seeds are saved.

**Mixed thinking budgets:** 111 retained outputs came from the original uncapped
recipe; 625 later outputs used thinking_token_budget=16,384 inside the same total
completion budget. Each record and raw request contains its actual recipe. No
outputs from the subsequently submitted but cancelled rest-run are included.

## Quality status and target provenance

Every included output has all 19 top-level fields, a readable narrative, a stopped
completion and nonempty actual emitted reasoning. **None of the 736 outputs passes
the stricter source-evidence schema validator.** Common errors are evidence quotes
longer than five words and invalid nested evidence-field structure. This is not a
human quality rating or proof that each narrative statement is false. These are
unreviewed, uncorrected outputs: no quality assessments or correction traces were
generated for this Ornith collection, and no Qwen quality labels are transferred.
The validation flag and original error are retained for every paper.

Training targets are exactly the successful generator's raw answer and its matching
actual emitted reasoning, with the actual source-only request messages. JSON is not
rewritten, reasoning is not fabricated, and no corrected final answers are substituted.
The Gemma 4 12B IT native chat template preserves the teacher reasoning in its thought
channel, with assistant-only loss, no truncation, one record per source paper.
The planned generator LoRAs use ranks 64/128, alpha 128/256, dropout 0.05, one epoch,
AdamW 2e-5, weight decay 0.01, 5% warmup then cosine, effective batch 8 on 8 GH200 GPUs.
Reviewer and corrector adapters are not trained.

## Files and reproducibility

- `data/train.jsonl`: full source, original summary/reasoning, narrative, actual request
  and response, model identity, thinking budget and source-validation result.
- `conversations/generation_reasoning.jsonl`: genuine generator SFT conversations.
- `traces/*.jsonl`: original file paths, UTF-8 content and member hashes for successful generation plus all preserved earlier returned attempts
  for each included source paper. Interrupted in-flight requests may have no returned
  response or usage record and are not reconstructed.
- `provenance/`: verbatim prompts, frozen hashes/identities, source selection and code.
- `manifest.json`, `evaluation_overlap_audit.json`, `trace_archives.jsonl`, `SHA256SUMS`:
  collection counts, source subset holdout audit and file checksums. Native training lengths are recorded separately in the training workflow.

No aggregate throughput is claimed for this mixed, resumed collection. Failed/startup
allocations and interrupted calls prevent inferring end-to-end throughput from retained
token counts alone. The separate historical 97-paper Ornith QA benchmark is not a
quality score for these 736 training records.

Sources: LAION Scientific-Summaries at revision
`232547b7b938a6588ce5d9d7faa96384860b7a16`, CC-BY-4.0; underlying paper rights remain
with their authors. The teacher model card lists MIT. Redistribution attribution and
source metadata are preserved. All available dataset source texts are included in full;
publisher PDF completeness is not guaranteed.
'''
    (release/'README.md').write_text(card)
    paths=sorted(p for p in release.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    for path in paths:
        if path.suffix!='.gz':assert not SECRET.search(path.read_bytes()), 'Potential credential in release'
    (release/'SHA256SUMS').write_text(''.join(digest(p)+'  '+str(p.relative_to(release))+'\n' for p in paths))
    write(ROOT/'outputs/release_ready.json',dict(papers=736,trace_archives=736,manifest_sha256=digest(release/'manifest.json'),checksums_sha256=digest(release/'SHA256SUMS'),bytes=sum(p.stat().st_size for p in paths)))
    print(json.dumps({'phase':'release_ready','papers':736,'bytes':sum(p.stat().st_size for p in paths)}),flush=True)


def native_only():
    cohort=load(ROOT/'inputs/frozen_cohort.json')
    sources={p['document_id']:p for p in load(SOURCE/'inputs/papers.json')}
    # Repeat the complete holdout guard on the compute node without modifying the published release.
    audit=Guard().require_disjoint([sources[p['document_id']] for p in cohort['papers']])
    write(ROOT/'training/evaluation_overlap_audit.json',audit)
    ready=load(ROOT/'outputs/release_ready.json')
    assert digest(ROOT/'release/manifest.json')==ready['manifest_sha256']
    rows=[json.loads(line) for line in (ROOT/'release/conversations/generation_reasoning.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['document_id'] for r in rows})==736
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/gemma-4-12b-it',local_files_only=True)
    counts=[];native=[]
    for row in rows:
        text,prefix,transformation=render(tokenizer,row['messages'])
        encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids']
        assert len(ids)<=65536, 'Full untruncated native example exceeds the training context'
        labels=[token if start>=len(prefix) and end>start else -100 for token,(start,end) in zip(ids,encoded['offset_mapping'])]
        assert any(v!=-100 for v in labels)
        native.append(dict(document_id=row['document_id'],domain=row['domain'],input_ids=ids,labels=labels,
            reasoning_in_target=True,prompt_characters=len(prefix),template_transformation=transformation,
            teacher_call_id=row['call_id']))
        counts.append(dict(document_id=row['document_id'],tokens=len(ids),target_tokens=sum(v!=-100 for v in labels)))
    path=ROOT/'training/gemma12_generation_reasoning.jsonl';jsonl(path,native)
    manifest=dict(examples=736,papers=736,path=str(path),sha256=digest(path),max_length=65536,
        truncation=False,assistant_only=True,reasoning_supervised=True,all_papers_used_once_per_epoch=True,
        min_tokens=min(r['tokens'] for r in counts),max_tokens=max(r['tokens'] for r in counts),
        mean_tokens=statistics.mean(r['tokens'] for r in counts),total_tokens=sum(r['tokens'] for r in counts),
        target_tokens=sum(r['target_tokens'] for r in counts),examples_by_length=counts,
        tokenizer_model='google/gemma-4-12B-it',tokenizer_revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
        tokenizer_sha256=digest(ROOT/'models/gemma-4-12b-it/tokenizer.json'),teacher=TEACHER,
        frozen_conversations_sha256=digest(ROOT/'release/conversations/generation_reasoning.jsonl'))
    write(ROOT/'training/manifest.json',manifest)
    write(ROOT/'outputs/core_ready.json',dict(papers=736,manifest_sha256=digest(ROOT/'release/manifest.json'),training_manifest_sha256=digest(ROOT/'training/manifest.json')))
    print(json.dumps({'phase':'core_ready','papers':736,'training_tokens':manifest['total_tokens'],'supervised_tokens':manifest['target_tokens'],'max_native_length':manifest['max_tokens']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release-only',action='store_true')
    args=parser.parse_args()
    if args.release_only:release_only()
    else:
        while not (ROOT/'outputs/release_ready.json').exists():__import__('time').sleep(2)
        native_only()
