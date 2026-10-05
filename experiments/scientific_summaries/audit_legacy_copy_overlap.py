"""Audit earlier repository runs, including recovered outputs and failed drafts.

Use a compute CPU step. The immutable full paper sources are the same pinned 97.
Anchor-selection responses and other control messages are not summary prose.
"""
import argparse
from collections import defaultdict
import concurrent.futures
import csv
import datetime
import hashlib
import json
from pathlib import Path
import time
from audit_all_copy_overlap import PINNED, parsed_summary
from copy_overlap_eval import compact, markdown
from ngram_overlap import SourceIndex, audit_summary, VERSION


def task(item):
    docid, source, outputs = item
    index = SourceIndex(source)
    found = {}
    for condition, location, expected, summary, representation in outputs:
        if expected:
            assert expected == index.sha256, location
        fingerprint = hashlib.sha256(json.dumps(summary, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        key = condition, fingerprint
        if key in found:
            found[key]['locations'].append(location)
            continue
        found[key] = dict(condition=condition, document_id=docid, source_sha256=index.sha256,
                          summary_sha256=fingerprint, locations=[location], representation=representation,
                          audit=audit_summary(source, summary, index=index))
    return list(found.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--root-snapshot-only', action='store_true',
                        help='Audit only the preserved four early Qwen failed drafts')
    args = parser.parse_args()
    start = time.monotonic()
    root = Path(__file__).resolve().parent / 'summary_runs'
    sources = {p['document_id']: p['fulltext'] for p in json.loads(PINNED.read_text())['papers']}
    pending = defaultdict(list)
    files = sorted(set(root.rglob('summaries.json')) | set(root.glob('**/documents/*.json')) |
                   set(root.glob('**/failure_journals/*.json')))
    early_snapshot = root.parent / 'qwen_summaries.json'
    files = [early_snapshot] if args.root_snapshot_only else files + [early_snapshot]
    skipped = defaultdict(int)
    empty_records = 0
    for path in files:
        value = json.loads(path.read_text())
        run = path.parent if path.name == 'summaries.json' else path.parent.parent
        group = 'Legacy Qwen27 early failed draft snapshot' if path == early_snapshot else 'Legacy ' + str(run.relative_to(root))
        records = [(f'documents[{i}]', r) for i, r in enumerate(value.get('documents', []))] + \
                  [(f'failures[{i}]', r) for i, r in enumerate(value.get('failures', []))] \
                  if path.name in ['summaries.json', 'qwen_summaries.json'] else [('', value)]
        for pointer, record in records:
            docid = record['document_id']
            assert docid in sources, 'Missing full source: ' + docid
            expected = record.get('fulltext_sha256', record.get('source_sha256'))
            assert not expected or hashlib.sha256(sources[docid].encode()).hexdigest() == expected
            location = str(path) + ('#' + pointer if pointer else '')
            count = 0
            for key, phase in [('summary', 'final'), ('draft_summary', 'failed draft')]:
                if record.get(key):
                    pending[docid].append((group + ' / ' + phase, location + '/' + key, expected,
                                           record[key], 'schema_object'))
                    count += 1
            for i, attempt in enumerate(record.get('attempts', [])):
                phase = attempt.get('phase', '')
                response = attempt.get('response')
                if not phase and path == early_snapshot:
                    phase = 'full_summary_repair' if attempt.get('repair_protocol') else 'generation'
                if phase not in ['generation', 'field_repair', 'full_summary_repair'] or response is None:
                    skipped[phase or 'unknown'] += 1
                    continue
                if isinstance(response, dict) and response.get('choices'):
                    response = response['choices'][0].get('message', {}).get('content')
                summary = parsed_summary(response)
                representation = 'schema_object'
                if phase == 'field_repair' and isinstance(summary, str):
                    try:
                        text = summary.strip()
                        if text.startswith('```'):
                            text = '\n'.join(text.splitlines()[1:-1])
                        payload = json.loads(text)
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        summary = payload
                    elif isinstance(payload, list) and len(attempt.get('requested_fields', [])) == 1:
                        summary = {attempt['requested_fields'][0]: payload}
                    representation = 'partial_field_repair' if isinstance(summary, dict) else 'unparsed_returned_text'
                elif isinstance(summary, str):
                    representation = 'unparsed_returned_text'
                if summary is not None:
                    condition = group + ' / returned ' + phase
                    if representation == 'unparsed_returned_text':
                        condition += ' / unparsed text diagnostic'
                    pending[docid].append((condition, location + f'/attempts[{i}]/response', expected,
                                           summary, representation))
                    count += 1
            empty_records += count == 0
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for found in pool.map(task, [(k, sources[k], v) for k, v in sorted(pending.items())]):
            rows.extend(found)
    groups = sorted({r['condition'] for r in rows})
    report = dict(audit_version=VERSION, complete=True,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - start, paper_sources=len(pending), input_files=len(files),
                  unique_summary_views=len(rows),
                  unique_source_summary_pairs=len({(r['source_sha256'], r['summary_sha256']) for r in rows}),
                  duplicate_locations_within_condition=sum(len(r['locations']) - 1 for r in rows),
                  excluded_control_or_metadata_only_attempts=dict(skipped), empty_records=empty_records,
                  source_roots=[str(root)], source_testset_sha256=hashlib.sha256(PINNED.read_bytes()).hexdigest(),
                  conditions={g: compact([r for r in rows if r['condition'] == g]) for g in groups},
                  qa_or_training_records_filtered=False,
                  scope='Legacy repository final/recovered summaries, failed drafts and returned generator/field-repair text; source-anchor selections excluded as control messages')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'per_summary.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    flats = []
    for row in rows:
        flat = {k: row[k] for k in ['condition', 'document_id', 'source_sha256', 'summary_sha256', 'representation']}
        for category in ['narrative', 'evidence', 'metadata']:
            value = row['audit']['categories'][category]
            flat.update({category + '_longest_match': value['longest_contiguous_match_words'],
                         category + '_classification': value['classification'],
                         category + '_coverage_6plus': value['covered_word_fraction_by_minimum_run']['6']})
        flats.append(flat)
    with (out / 'per_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flats[0]) if flats else ['condition', 'document_id'])
        writer.writeheader()
        writer.writerows(flats)
    (out / 'RESULTS.md').write_text('# Earlier scientific-summary runs: source-copy audit\n\n' +
                                  f'{len(pending)} paper sources; {len(rows)} saved/returned summary views.\n\n' +
                                  markdown(report) + '\n' + report['scope'] + '\n')
    print(json.dumps({k: report[k] for k in ['complete', 'paper_sources', 'input_files', 'unique_summary_views',
                                         'unique_source_summary_pairs', 'elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
