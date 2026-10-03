"""Export/check the exact 97-paper, 970-MCQ test inputs; no network or inference."""
import argparse
import hashlib
import json
from collections import Counter

from evaluate import ROOT, SEED, load, questions_for
from project_alexandria.io import write_json_atomic


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    papers = load(ROOT / 'data/papers.json')
    selection = load(ROOT / 'summary_comparison_manifest.json')
    ids = set(selection['documents'])
    if len(ids) != 97 or len(selection['documents']) != 97:
        raise ValueError('Expected unique frozen 97-paper cohort')
    frozen = {d['document_id']: d for d in load(ROOT / 'summary_runs/qwen35_9b/qa_evaluated.json')}
    exported = []
    for index, paper in enumerate(papers):
        if paper['document_id'] not in ids:
            continue
        questions = questions_for(paper, index)
        if questions != frozen[paper['document_id']]['questions']:
            raise ValueError('QA differs from completed Qwen9B evaluation')
        for field, expected in [('fulltext', 'fulltext_sha256'), ('existing_summary', 'summary_sha256')]:
            if hashlib.sha256(paper[field].encode()).hexdigest() != paper[expected]:
                raise ValueError('Source/summary content hash mismatch')
        exported.append(dict(document_id=paper['document_id'], original_paper_index=index,
            source={k: paper.get(k) for k in ('paper_id', 'viewer_config', 'viewer_row_index',
                'source_title', 'source_authors', 'source_year', 'source_doi', 'openalex_id')},
            fulltext=paper['fulltext'], fulltext_sha256=paper['fulltext_sha256'],
            existing_summary=paper['existing_summary'], summary_sha256=paper['summary_sha256'],
            qa_author='gpt-6-luna', questions=questions))
    if len(exported) != 97 or sum(len(p['questions']) for p in exported) != 970:
        raise ValueError('Incomplete test set')
    bundle = dict(schema_version='alexandria_scientific_summaries_test_v1',
        dataset='laion/Scientific-Summaries', dataset_revision_observed=load(ROOT / 'data/manifest.json')['revision_observed'],
        paper_count=97, question_count=970, selection_seed=SEED,
        option_order='exact evaluated order; do not reshuffle',
        excluded_document_ids=selection['excluded_ids'], papers=exported)
    sources = ['data/papers.json', 'data/manifest.json', 'summary_runs/qwen35_9b/qa_evaluated.json']
    sources += ['data/qa/' + p['document_id'] + '.json' for p in exported]
    manifest = dict(schema_version=bundle['schema_version'], paper_count=97, question_count=970,
        subsets=dict(Counter(p['source']['viewer_config'] for p in exported)),
        ordered_document_ids=[p['document_id'] for p in exported],
        excluded_document_ids=bundle['excluded_document_ids'],
        source_file_sha256={p: sha(ROOT / p) for p in sources},
        note='Exact inputs/options/gold keys used for Qwen9B evaluation; source-only generation, '
             'no gold/rationale/evidence disclosure to the student; file hashes checked separately.')
    return bundle, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Verify existing export without writing')
    args = parser.parse_args()
    bundle, manifest = build()
    targets = [(ROOT / 'data/testset.json', bundle), (ROOT / 'data/testset_manifest.json', manifest)]
    if args.check:
        for path, value in targets:
            if load(path) != value:
                raise ValueError('Export is stale or changed: ' + path.name)
    else:
        for path, value in targets:
            write_json_atomic(str(path), value)
    print('VERIFIED' if args.check else 'EXPORTED', '97 papers, 970 MCQs; exact completed evaluation inputs')


if __name__ == '__main__':
    main()
