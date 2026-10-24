"""Recalculate saved QA, copy aggregates and scaling without model inference."""
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(os.environ.get('ALEXANDRIA_HYBRID_ROOT', Path(__file__).resolve().parent))


def read(path):
    return json.loads(path.read_text())


def main():
    rows = [json.loads(line) for line in (ROOT / 'inputs/cohort97.jsonl').read_text().splitlines()]
    assert len(rows) == 97 and len({r['document_id'] for r in rows}) == 97
    scores = read(ROOT / 'outputs/qa_results.json')['scores']
    testset = read(ROOT / 'inputs/testset.json')
    gold = {(p['document_id'], q['question_index']): q['answer'] for p in testset['papers'] for q in p['questions']}
    for condition, score in scores.items():
        answers = read(ROOT / f'outputs/qa/{condition}.json')
        assert len(answers) == 970 and len({(s['document_id'], s['question_index']) for s in answers}) == 970
        assert sum(s['prediction'] == s['gold'] for s in answers) == score['correct']
        assert all(s['gold'] == gold[s['document_id'], s['question_index']] for s in answers)
        contexts = {}
        coverage = []
        for row in rows:
            record = read(ROOT / f'outputs/repaired/{condition}/{row["uid"]}.json')
            assert record['source_sha256'] == row['source_sha256']
            contexts[row['document_id']] = hashlib.sha256(record['judge_context'].encode()).hexdigest()
            coverage.append(record['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'])
        assert all(contexts[s['document_id']] == s['context_sha256'] for s in answers)
        saved = read(ROOT / f'outputs/{condition}_complete.json')
        assert abs(statistics.mean(coverage) - saved['mean_coverage']) < 1e-12
        print(condition, f'{score["correct"]}/970', f'coverage={statistics.mean(coverage):.6%}')
    meanings = read(ROOT / 'outputs/semantic_results.json')
    assert meanings['complete']
    for condition, value in meanings['conditions'].items():
        assert sum(value['counts'].values()) == value['total']
    print('PASS: all 5,820 QA slots, context hashes, saved copy aggregates, and complete meaning counts.')


if __name__ == '__main__':
    main()
