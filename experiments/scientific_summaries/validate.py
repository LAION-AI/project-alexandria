"""Audit corpus, evidence, checkpoint consistency, and report links without inference."""
import argparse
import hashlib
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from evaluate import ROOT, CONDITIONS, historical_answer_prompt, load, questions_for


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.questions = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
        if tag == 'details' and attrs.get('class') == 'question':
            self.questions += 1


def audit(partial=False):
    papers = load(ROOT / 'data' / 'papers.json')
    manifest = load(ROOT / 'data' / 'manifest.json')
    expected = {p['document_id']: p for p in manifest['documents']}
    assert len(papers) == 100 and len(expected) == 100
    assert Counter(p['viewer_config'] for p in papers) == {'arxiv': 50, 'bethgelab': 50}
    assert len({p['fulltext_sha256'] for p in papers}) == 100
    normalized_titles = {' '.join(p['source_title'].lower().split()) for p in papers}
    assert len(normalized_titles) == 100
    results = load(ROOT / 'results.json')
    conditions = results.get('protocol', {}).get('conditions', CONDITIONS)
    documents = {d['document_id']: d for d in results['documents']}
    assert len(documents) == len(results['documents']), 'Duplicate completed document'
    gold = Counter()
    ready = 0
    for index, paper in enumerate(papers):
        identifier = paper['document_id']
        for field, folder in [('fulltext', 'papers'), ('existing_summary', 'summaries')]:
            text = paper[field]
            digest = hashlib.sha256(text.encode()).hexdigest()
            hash_key = 'fulltext_sha256' if field == 'fulltext' else 'summary_sha256'
            assert digest == paper[hash_key] == expected[identifier][hash_key]
            # read_text() normalizes source CRLFs and would falsely reject byte-identical exports.
            assert (ROOT / 'data' / folder / (identifier + '.txt')).read_bytes().decode('utf-8') == text
        questions = questions_for(paper, index)
        if questions is None:
            assert partial, 'Missing QA: ' + identifier
            continue
        ready += 1
        for q in questions:
            assert paper['fulltext'][q['evidence_start']:q['evidence_end']] == q['evidence_quote']
            gold[q['answer']] += 1
        if identifier not in documents:
            assert partial, 'Missing judge results: ' + identifier
            continue
        document = documents[identifier]
        assert document['questions'] == questions, 'Changed QA after checkpoint: ' + identifier
        assert len(document['rows']) == 10
        for q, row in zip(questions, document['rows']):
            assert row['question_index'] == q['question_index'] and row['gold'] == q['answer']
            assert set(row['predictions']) == set(conditions)
            for condition in conditions:
                response = row['responses'][condition]
                contexts = {'no_context': '', 'original': paper['fulltext'], 'summary': paper['existing_summary']}
                if condition == 'knowledge_units':
                    contexts[condition] = load(ROOT / 'data/kus' / (identifier + '.json'))['judge_context']
                if condition == 'qwen_summary':
                    generated = load(ROOT / 'data/qwen_summaries' / (identifier + '.json'))
                    assert generated['fulltext_sha256'] == paper['fulltext_sha256']
                    for span in generated['evidence_spans']:
                        assert paper['fulltext'][span['start']:span['end']] == span['quote']
                        assert 0 < len(span['quote'].split()) <= 5
                    contexts[condition] = generated['judge_context']
                context = contexts[condition]
                prompt = historical_answer_prompt(q['formatted_question'], context)
                assert response['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
                assert response['prompt_tokens'] + 100 <= response.get('context_limit', 16384)
                assert 1 <= len(response['attempts']) <= 5
                assert row['predictions'][condition] in (None, 'A', 'B', 'C', 'D')
    if not partial:
        assert len(documents) == ready == 100
        assert gold == {letter: 250 for letter in 'ABCD'}
        evaluated = load(ROOT / 'data' / 'qa_evaluated.json')
        assert len(evaluated) == 100
        assert {q['document_id']: q['questions'] for q in evaluated} == {
            k: d['questions'] for k, d in documents.items()}
        for condition in conditions:
            s = results['summary'][condition]
            rows = [r for d in documents.values() for r in d['rows']]
            correct = sum(r['predictions'][condition] == r['gold'] for r in rows)
            invalid = sum(r['predictions'][condition] is None for r in rows)
            assert s['total'] == 1000 and s['correct'] == correct and s['invalid'] == invalid
            assert s['accuracy'] == correct / 1000
        report = (ROOT / 'report.html').read_text(encoding='utf-8')
        links = Links()
        links.feed(report)
        assert links.questions == 1000
        for link in links.links:
            if link.startswith(('https://', 'http://', '#')):
                continue
            assert (ROOT / link).is_file(), 'Broken report link: ' + link
        for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    print(json.dumps({'papers': len(papers), 'qa_ready': ready, 'judged': len(documents),
                      'gold_distribution': dict(gold), 'partial': partial}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    audit(parser.parse_args().partial)
