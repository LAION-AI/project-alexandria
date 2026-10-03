"""Verify released measurements against frozen questions and complete QA/trace artifacts."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def main():
    entries = (HERE / 'SHA256SUMS').read_text().splitlines()
    for line in entries:
        expected, name = line.split('  ', 1)
        path = HERE / name
        require(path.resolve().is_relative_to(HERE), 'Path escapes package')
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected, 'Checksum mismatch: ' + name)
        if name.endswith('.tar.gz'):
            with tarfile.open(path, 'r:gz') as tar:
                for member in tar:
                    require(member.isfile() and not Path(member.name).is_absolute() and
                            '..' not in Path(member.name).parts, 'Unsafe archive member')
                    require(not SECRET.search(tar.extractfile(member).read()), 'Potential credential in archive')
        elif name.endswith('.gz'):
            require(not SECRET.search(gzip.decompress(content)), 'Potential credential in gzip')
        else:
            require(not SECRET.search(content), 'Potential credential in ' + name)
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
              p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    require(actual == {line.split('  ', 1)[1] for line in entries}, 'Manifest/file set differs')
    manifest = load(HERE / 'provenance/experiment_manifest.json')
    frozen_path = HERE.parent / 'data/testset.json'
    require(hashlib.sha256(frozen_path.read_bytes()).hexdigest() == manifest['testset_sha256'], 'Test set changed')
    papers = {p['document_id']: p for p in load(frozen_path)['papers']}
    qa = json.loads(gzip.decompress((HERE / 'artifacts/qa-results.json.gz').read_bytes()))
    report = load(HERE / 'artifacts/report.json')
    require(len(qa['documents']) == len(papers) == 97, 'Paper count mismatch')
    require({d['document_id'] for d in qa['documents']} == set(papers), 'Paper identities mismatch')
    totals = {condition: dict(correct=0, total=0, invalid=0) for condition in report['models']}
    traces = {}
    for model, prefix in [('ornith35_dflash', 'ornith35')]:
        archives = list((HERE / 'artifacts/paper_traces').glob('*.tar.gz'))
        require(len(archives) == 97, 'Missing paper traces')
        for archive in archives:
            with tarfile.open(archive, 'r:gz') as tar:
                document = json.load(tar.extractfile('document.json'))
            require(document['fulltext_sha256'] == papers[archive.name[:-7]]['fulltext_sha256'], 'Trace source mismatch')
            for phase in ['raw', 'corrected']:
                context = document[phase]['judge_context']
                traces[(document['document_id'], prefix + '_' + phase)] = hashlib.sha256(context.encode()).hexdigest()
    for doc in qa['documents']:
        paper = papers[doc['document_id']]
        require(doc['questions'] == paper['questions'], 'Question/gold order changed')
        require(doc['fulltext_sha256'] == paper['fulltext_sha256'], 'QA source mismatch')
        require(set(doc['conditions']) == set(totals), 'Missing conditions')
        for label, condition in doc['conditions'].items():
            require(len(condition['rows']) == 10, 'Missing answers')
            if (doc['document_id'], label) in traces:
                require(condition['context_sha256'] == traces[(doc['document_id'], label)], 'QA/trace context mismatch')
            for row, question in zip(condition['rows'], paper['questions']):
                require(row['gold'] == question['answer'] and row['question_index'] == question['question_index'],
                        'Answer alignment changed')
                prediction = next(iter(row['predictions'].values()))
                totals[label]['correct'] += prediction == row['gold']
                totals[label]['total'] += 1
                totals[label]['invalid'] += prediction is None
    for label, values in totals.items():
        require(all(values[k] == report['models'][label][k] for k in values), 'Reported score mismatch: ' + label)
    with (HERE / 'paper_scores.csv').open() as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 970, 'Missing paper score rows')
    for label, values in totals.items():
        require(sum(int(r['correct']) for r in rows if r['condition'] == label) == values['correct'], 'CSV score mismatch')
    with (HERE / 'throughput.csv').open() as f:
        require(len(list(csv.DictReader(f))) == 96, 'Missing throughput observations')
    with (HERE / 'self_audit_scores.csv').open() as f:
        require(len(list(csv.DictReader(f))) == 194, 'Missing audit rows')
    require(report['final_audit']['passed'], 'Recorded original audit failed')
    print(json.dumps(dict(passed=True, checksummed_files=len(entries), papers=97, conditions=10,
                          scored_answers=sum(v['total'] for v in totals.values()), paper_trace_archives=97,
                          source_context_checks=len(traces), throughput_observations=96, self_audit_rows=194)))


if __name__ == '__main__':
    main()
