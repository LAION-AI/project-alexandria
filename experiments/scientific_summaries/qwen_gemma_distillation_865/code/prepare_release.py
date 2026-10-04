"""Freeze the accepted cohort, preserve genuine traces, and prepare generator-only SFT."""
import argparse
import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
import tarfile

from common import ROOT, SOURCE, TRAINING_ROOT, load, write, digest, jsonl

sys.path.insert(0, str(SOURCE / 'code'))
from run import valid_summary, review_valid, statements, summary_system, json_object
from holdout_guard import Guard
from render_training import render

SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def native_tokens(rows):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/gemma-4-12b-it', local_files_only=True)
    output, counts = [], []
    for row in rows:
        text, prefix, transformation = render(tokenizer, row['messages'])
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids']
        if len(ids) > 65536:
            raise ValueError('Untruncated native example exceeds 65536: ' + row['document_id'])
        labels = [token if start >= len(prefix) and end > start else -100
                  for token, (start, end) in zip(ids, encoded['offset_mapping'])]
        if not any(value != -100 for value in labels):
            raise ValueError('Missing assistant target')
        output.append(dict(document_id=row['document_id'], domain=row['domain'],
                           input_ids=ids, labels=labels, reasoning_in_target=True,
                           prompt_characters=len(prefix), template_transformation=transformation,
                           teacher_call_id=row['call_id']))
        counts.append(dict(document_id=row['document_id'], tokens=len(ids),
                           target_tokens=sum(value != -100 for value in labels)))
    path = ROOT / 'training/gemma12_generation_reasoning.jsonl'
    jsonl(path, output)
    result = dict(examples=len(output), papers=len(output), path=str(path), sha256=digest(path),
                  max_length=65536, truncation=False, assistant_only=True, reasoning_supervised=True,
                  all_papers_used_once_per_epoch=True, min_tokens=min(r['tokens'] for r in counts),
                  max_tokens=max(r['tokens'] for r in counts), mean_tokens=statistics.mean(r['tokens'] for r in counts),
                  total_tokens=sum(r['tokens'] for r in counts), target_tokens=sum(r['target_tokens'] for r in counts),
                  examples_by_length=counts,
                  tokenizer_model='google/gemma-4-12B-it',
                  tokenizer_revision=load(TRAINING_ROOT / 'inputs/model_gemma-4-12b-it.json')['revision'],
                  tokenizer_sha256=digest(ROOT / 'models/gemma-4-12b-it/tokenizer.json'))
    write(ROOT / 'training/manifest.json', result)
    return result


def archive_paper(item):
    folder = SOURCE / 'outputs/documents' / item['document_id']
    target = ROOT / 'release/traces' / (item['document_id'] + '.tar.gz')
    if target.exists():
        expected={str(p.relative_to(folder)):p for p in folder.rglob('*')
                  if p.is_file() and p.name!='.lock' and not p.name.endswith('.tmp')}
        try:
            with tarfile.open(target,'r:gz') as handle:
                actual=handle.getmembers()
                assert len(actual)==len(expected) and {p.name for p in actual}==set(expected)
                for member in actual:
                    payload=handle.extractfile(member).read()
                    assert hashlib.sha256(payload).hexdigest()==digest(expected[member.name])
                    assert not SECRET.search(payload)
            return dict(document_id=item['document_id'],file=str(target.relative_to(ROOT/'release')),
                        bytes=target.stat().st_size,sha256=digest(target))
        except (AssertionError,OSError,tarfile.TarError):
            pass
    temporary = target.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as handle:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.name == '.lock' or path.name.endswith('.tmp'):
                continue
            if SECRET.search(path.read_bytes()):
                raise ValueError('Potential credential in trace file: ' + str(path.relative_to(folder)))
            handle.add(path, arcname=str(path.relative_to(folder)), recursive=False)
    temporary.replace(target)
    return dict(document_id=item['document_id'], file=str(target.relative_to(ROOT / 'release')),
                bytes=target.stat().st_size, sha256=digest(target))


def core():
    cohort = load(ROOT / 'inputs/frozen_cohort.json')
    papers = {p['document_id']: p for p in load(SOURCE / 'inputs/papers.json')}
    selected = [papers[p['document_id']] for p in cohort['papers']]
    guard = Guard()
    audit = guard.require_disjoint(selected)
    assert len({p['source_sha256'] for p in selected}) == cohort['accepted_papers']
    write(ROOT / 'release/evaluation_overlap_audit.json', audit)
    prompts = load(ROOT / 'inputs/reference_prompts.json')
    config = load(ROOT / 'inputs/run_config.json')
    system = summary_system(prompts)
    records, generation, final_reasoning, final_summary = [], [], [], []
    scores = collections.Counter()
    conditioned = 0
    for item in cohort['papers']:
        path = Path(item['final_path'])
        if digest(path) != item['final_sha256']:
            raise ValueError('Accepted artifact changed after user stop')
        final = load(path)
        paper = papers[item['document_id']]
        assert final['source_sha256'] == paper['source_sha256']
        assert final['protocol_sha256'] == config['protocol_sha256']
        valid_summary(final['summary'], paper['fulltext'], [])
        passed, _ = review_valid(final['quality_assessment'], statements(final['summary']), paper['fulltext'])
        if not passed:
            raise ValueError('Saved final no longer meets its original quality gate')
        journal = load(SOURCE / final['journal_file'])
        original = None
        for name in journal['calls']:
            if not Path(name).name.startswith('generation'):
                continue
            call = load(SOURCE / name)
            if (call['status'] == 'completed' and call.get('finish_reason') == 'stop'
                    and call.get('reasoning_content', '').strip()
                    and any(c['check'] == 'complete_summary_schema_root' and c['passed']
                            for c in call.get('quality_checks', []))):
                if json_object(call['content']) == journal['initial_draft']:
                    original = call
                    break
        if original is None:
            raise ValueError('No actual complete generator trace for ' + item['document_id'])
        request_messages = original['request']['messages']
        expected_user = prompts['summary-user'].replace('{paper_text}', paper['fulltext'])
        assert len(request_messages) == 2 and request_messages[0] == {'role': 'system', 'content': system}
        assert request_messages[1]['role'] == 'user' and request_messages[1]['content'].startswith(expected_user)
        remainder = request_messages[1]['content'][len(expected_user):]
        if remainder and not remainder.startswith('\n\nFormat error from the preceding attempt: '):
            raise ValueError('Unexpected non-generation input')
        target = load(path.parent / 'calls' / (final['full_summary_reasoning_call_id'] + '.json'))
        candidate, _, _ = valid_summary(json_object(target['content']), paper['fulltext'], [])
        assert candidate == final['summary'] and target['reasoning_content'].strip()
        meta = dict(document_id=paper['document_id'], domain=paper['domain'], split='train',
                    source_sha256=paper['source_sha256'], teacher=config['teacher'])
        generation.append(dict(**meta, task='source_only_summary_generation_with_reasoning',
                               call_id=original['call_id'], reasoning_in_target=True,
                               target_is_uncorrected_generator_output=True,
                               final_acceptance_applies_to_repaired_summary=True,
                               messages=request_messages + [dict(role='assistant', content=original['content'],
                                                                 reasoning_content=original['reasoning_content'])]))
        final_reasoning.append(dict(**meta, task='final_summary_with_original_conditioned_reasoning',
                                    call_id=target['call_id'], reasoning_in_target=True,
                                    conditioned_on_prior_draft_and_feedback=final['reasoning_target_conditioned_on_prior_draft_and_feedback'],
                                    used_for_requested_generator_training=False,
                                    messages=target['request']['messages'] + [dict(role='assistant', content=target['content'],
                                                                                 reasoning_content=target['reasoning_content'])]))
        final_summary.append(dict(**meta, task='validated_final_summary_without_reasoning',
                                  reasoning_in_target=False, used_for_requested_generator_training=False,
                                  messages=[dict(role='system', content=system), dict(role='user', content=expected_user),
                                            dict(role='assistant', content=json.dumps(final['summary'], ensure_ascii=False))]))
        records.append(dict(**meta, source=paper, raw_generation=original,
                            final=final, final_reasoning_call=target,
                            correction_events=journal['events'], assessment_rounds=journal['reviews']))
        scores.update(final['quality_assessment']['scores'])
        conditioned += bool(final['reasoning_target_conditioned_on_prior_draft_and_feedback'])
    release = ROOT / 'release'
    jsonl(release / 'papers_and_traces.jsonl', records)
    jsonl(release / 'conversations/generation_reasoning.jsonl', generation)
    jsonl(release / 'conversations/final_conditioned_reasoning.jsonl', final_reasoning)
    jsonl(release / 'conversations/validated_final_summary.jsonl', final_summary)
    flat = [dict(document_id=r['document_id'], domain=r['domain'], source_sha256=r['source_sha256'],
                 fulltext=r['source']['fulltext'], source_metadata_json=json.dumps(r['source'], ensure_ascii=False),
                 teacher_model=r['teacher']['model'], teacher_revision=r['teacher']['revision'],
                 raw_generator_summary=r['raw_generation']['content'],
                 raw_generator_reasoning=r['raw_generation']['reasoning_content'],
                 validated_final_summary_json=json.dumps(r['final']['summary'], ensure_ascii=False),
                 validated_final_narrative=r['final']['narrative'],
                 final_summary_reasoning=r['final_reasoning_call']['reasoning_content'],
                 final_reasoning_conditioned_on_draft=r['final']['reasoning_target_conditioned_on_prior_draft_and_feedback'],
                 final_quality_assessment_json=json.dumps(r['final']['quality_assessment'], ensure_ascii=False))
            for r in records]
    (release / 'data').mkdir(exist_ok=True)
    jsonl(release / 'data/train.jsonl', flat)
    (release / 'provenance').mkdir(exist_ok=True)
    for file in (ROOT / 'inputs').glob('*.json'):
        if file.name != 'evaluation_exclusion_index.json':
            shutil.copy2(file, release / 'provenance' / file.name)
    (release / 'provenance/summary_system_prompt.txt').write_text(system)
    (release / 'provenance/summary_user_prompt.txt').write_text(prompts['summary-user'])
    (release / 'provenance/quality_review_system_prompt.txt').write_text(config['review_system'])
    for folder, target in [(SOURCE / 'code', release / 'provenance/teacher_code'),
                           (ROOT / 'code', release / 'provenance/experiment_code')]:
        shutil.copytree(folder, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    native = native_tokens(generation)
    manifest = dict(status='complete_frozen_cohort', accepted_papers=len(records), domains=cohort['domains'],
                    teacher=config['teacher'], protocol_sha256=config['protocol_sha256'],
                    stopped_by_user=True, final_reasoning_conditioned_papers=conditioned,
                    final_reasoning_source_only_papers=len(records) - conditioned,
                    generator_training_examples=len(generation), generator_target='actual original completed generator output plus matching emitted reasoning',
                    corrected_final_targets_used_for_generator_training=False,
                    reviewer_or_corrector_training=False, student_training_manifest_sha256=digest(ROOT / 'training/manifest.json'),
                    quality_scores_mean={k:v/len(records) for k,v in scores.items()},
                    native_training_summary={k:v for k,v in native.items() if k != 'examples_by_length'},
                    external_evaluation=audit, dataset_attribution=load(ROOT / 'inputs/selection_manifest.json')['attribution'],
                    dataset_license='CC-BY-4.0; underlying source-paper rights remain with their authors',
                    actual_teacher_reasoning_only=True)
    write(release / 'manifest.json', manifest)
    write(ROOT / 'outputs/core_ready.json', dict(papers=len(records), manifest_sha256=digest(release / 'manifest.json'),
                                               training_manifest_sha256=digest(ROOT / 'training/manifest.json')))
    print(json.dumps({'phase':'core_ready','papers':len(records),'training':manifest['native_training_summary']}), flush=True)


def archive_and_finish():
    cohort=load(ROOT / 'inputs/frozen_cohort.json')
    release=ROOT / 'release'
    manifest=load(release / 'manifest.json')
    config=load(ROOT / 'inputs/run_config.json')
    (release / 'traces').mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        archives = list(pool.map(archive_paper, cohort['papers']))
    jsonl(release / 'trace_archives.jsonl', archives)
    shutil.copytree(ROOT / 'code', release / 'provenance/experiment_code', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    from dataset_card import render_card
    (release / 'README.md').write_text(render_card(manifest, config, archives))
    paths = sorted(p for p in release.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (release / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + str(p.relative_to(release)) + '\n' for p in paths))
    write(ROOT / 'outputs/release_ready.json', dict(papers=len(cohort['papers']), trace_archives=len(archives),
                                                  manifest_sha256=digest(release / 'manifest.json'),
                                                  checksums_sha256=digest(release / 'SHA256SUMS'),
                                                  bytes=sum(p.stat().st_size for p in paths)))
    print(json.dumps({'phase':'release_ready','papers':len(cohort['papers']),'trace_archives':len(archives)}), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-only',action='store_true')
    parser.add_argument('--archive-only',action='store_true')
    args=parser.parse_args()
    assert not (args.core_only and args.archive_only)
    if not args.archive_only:core()
    if not args.core_only:archive_and_finish()
