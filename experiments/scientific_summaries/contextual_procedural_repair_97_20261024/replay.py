"""Hash-verify and materialize saved experiments; verify QA without model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import statistics

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for bundle in manifest['bundles']:
        path = HERE / bundle['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle['sha256']
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                target = (root / member.name).resolve()
                assert member.isfile() and target.is_relative_to(root)
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == manifest['files'][member.name]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert target.read_bytes() == data, target
                else:
                    target.write_bytes(data)
    gold = {(p['document_id'], q['question_index']): q['answer']
            for p in json.loads((root / 'common/testset.json').read_text())['papers'] for q in p['questions']}
    rows = [json.loads(line) for line in (root / 'common/cohort97.jsonl').read_text().splitlines()]
    for case in ['phrase', 'structural']:
        base = root / case
        result = json.loads((base / 'outputs/results_and_scaling.json').read_text())
        for condition, entry in result['conditions'].items():
            answers = json.loads((base / f'outputs/qa/{condition}.json').read_text())
            assert len(answers) == 970 and len({(a['document_id'], a['question_index']) for a in answers}) == 970
            assert all(a['gold'] == gold[a['document_id'], a['question_index']] for a in answers)
            assert sum(a['prediction'] == a['gold'] for a in answers) == entry['qa']['correct']
            contexts = {}
            coverage = []
            for row in rows:
                value = json.loads((base / f'outputs/repaired/{condition}/{row["uid"]}.json').read_text())
                contexts[row['document_id']] = hashlib.sha256(value['judge_context'].encode()).hexdigest()
                coverage.append(value['audit']['categories']['narrative']['covered_word_fraction_by_minimum_run']['6'])
            assert all(contexts[a['document_id']] == a['context_sha256'] for a in answers)
            assert abs(statistics.mean(coverage) - entry['overlap']['mean_coverage']) < 1e-12
            print(case, condition, entry['qa']['correct'], f'{statistics.mean(coverage):.6%}')
    print('PASS: hash-verified evidence and all 7,760 QA slots across both saved experiments.')


if __name__ == '__main__':
    main()
