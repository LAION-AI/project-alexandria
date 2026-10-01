"""Historical fixed-Qwen evaluation of existing Scientific-Summaries, with checkpoints."""
import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'src'))
from project_alexandria.backends import OpenAICompatibleBackend
from project_alexandria.experiments.mcq import historical_answer_prompt, extract_historical_choice
from project_alexandria.experiments.reproduce import JUDGE_SYSTEM_PROMPT
from project_alexandria.io import write_json_atomic

CONDITIONS = ('no_context', 'original', 'summary')
SEED = 250219413

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def evidence_span(text, quote):
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError('evidence quote is empty or not text')
    start = text.find(quote)
    if start >= 0:
        return start, start + len(quote), quote
    # Whitespace differences from PDF line wrapping are permitted; retain the exact source span.
    import re
    pattern = r'\s+'.join(re.escape(x) for x in quote.split())
    match = re.search(pattern, text) if pattern else None
    if match:
        return match.start(), match.end(), match.group()
    # Align typographical PDF whitespace/ligatures, then save the actual source substring.
    # No approximate semantic or fuzzy matching is accepted here.
    import unicodedata
    normalized = []
    positions = []
    for index, character in enumerate(text):
        for char in unicodedata.normalize('NFKC', character):
            if not char.isspace():
                normalized.append(char)
                positions.append(index)
    target = ''.join(c for c in unicodedata.normalize('NFKC', quote) if not c.isspace())
    hit = ''.join(normalized).find(target) if target else -1
    if hit >= 0:
        start, end = positions[hit], positions[hit + len(target) - 1] + 1
        return start, end, text[start:end]
    raise ValueError('evidence quote is not present in fulltext: ' + quote[:100])

def questions_for(paper, index):
    path = ROOT / 'data' / 'qa' / (paper['document_id'] + '.json')
    if not path.exists():
        return None
    authored = load(path)
    if authored['document_id'] != paper['document_id'] or len(authored['questions']) != 10:
        raise ValueError('wrong document ID or question count: ' + path.name)
    questions = []
    for q_index, original in enumerate(authored['questions']):
        q = dict(original)
        if (not isinstance(q['options'], dict) or set(q['options']) != set('ABCD')
                or not isinstance(q['answer'], str) or q['answer'] not in set('ABCD')):
            raise ValueError('invalid MCQ schema: ' + path.name)
        if (not all(isinstance(v, str) and v.strip() for v in q['options'].values())
                or not isinstance(q['question'], str) or not q['question'].strip()
                or not isinstance(q['rationale'], str) or not q['rationale'].strip()):
            raise ValueError('MCQ fields must contain nonempty text: ' + path.name)
        if len(set(q['options'].values())) != 4 or not q['question'] or not q['rationale']:
            raise ValueError('duplicate/missing MCQ content: ' + path.name)
        start, end, quote = evidence_span(paper['fulltext'], q['evidence_quote'])
        q.update(evidence_quote=quote, evidence_start=start, evidence_end=end)
        # Balanced answer positions assigned independently of author preference.
        target = 'ABCD'[(index * 10 + q_index) % 4]
        wrong = [value for letter, value in q['options'].items() if letter != q['answer']]
        random.Random(SEED + index * 10 + q_index).shuffle(wrong)
        correct = q['options'][q['answer']]
        q['options'] = {letter: correct if letter == target else wrong.pop() for letter in 'ABCD'}
        q['answer'] = target
        q['formatted_question'] = q['question'] + '\n' + '\n'.join(
            f'{letter}) {value}' for letter, value in q['options'].items())
        q['question_index'] = q_index
        questions.append(q)
    if len({q['question'].strip().lower() for q in questions}) != 10:
        raise ValueError('duplicate questions within document: ' + path.name)
    return questions

def count_tokens(base_url, prompt):
    payload = json.dumps({'model': 'qwen25', 'messages': [
        {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt}], 'add_generation_prompt': True}).encode()
    request = urllib.request.Request(base_url.replace('/v1', '') + '/tokenize',
                                     data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)['count']

def run_document(paper, questions, backend, previous=None, context_limit=16384):
    slots = []
    for question in questions:
        for condition in CONDITIONS:
            context = {'no_context': '', 'original': paper['fulltext'],
                       'summary': paper['existing_summary'],
                       'knowledge_units': paper.get('knowledge_units', ''),
                       'qwen_summary': paper.get('qwen_summary', '')}[condition]
            prompt = historical_answer_prompt(question['formatted_question'], context)
            if previous and condition in previous['rows'][question['question_index']]['predictions']:
                record = previous['rows'][question['question_index']]['responses'].get(condition, {})
                if record.get('prompt_sha256') == hashlib.sha256(prompt.encode()).hexdigest():
                    continue
            slots.append((question['question_index'], condition, prompt))
    if not slots:
        return dict(previous, questions=questions)
    # Preflight all prompts. Explicitly reject context overflow instead of truncating any paper.
    counts = [count_tokens(backend.base_url, x[2]) for x in slots]
    if max(counts) + 100 > context_limit:
        raise ValueError(f'context exceeds judge limit for {paper["document_id"]}: {max(counts)}')
    responses = backend.generate_batch(JUDGE_SYSTEM_PROMPT, [x[2] for x in slots], max_tokens=100)
    records = []
    for slot, response, tokens in zip(slots, responses, counts):
        records.append({'question_index': slot[0], 'condition': slot[1],
                        'prompt_sha256': hashlib.sha256(slot[2].encode()).hexdigest(),
                        'prompt_tokens': tokens, 'attempts': [response],
                        'context_limit': context_limit,
                        'prediction': extract_historical_choice(response)})
    for _ in range(4):
        failed = [i for i, r in enumerate(records) if r['prediction'] is None]
        if not failed:
            break
        responses = backend.generate_batch(JUDGE_SYSTEM_PROMPT,
            [slots[i][2] for i in failed], max_tokens=100)
        for i, response in zip(failed, responses):
            records[i]['attempts'].append(response)
            records[i]['prediction'] = extract_historical_choice(response)
    rows = []
    for q in questions:
        q_records = [r for r in records if r['question_index'] == q['question_index']]
        old = previous['rows'][q['question_index']] if previous else {}
        rows.append({'document_id': paper['document_id'], 'source': paper['viewer_config'],
                     'question_index': q['question_index'], 'gold': q['answer'],
                     'predictions': dict(old.get('predictions', {}), **{r['condition']: r['prediction'] for r in q_records}),
                     'responses': dict(old.get('responses', {}), **{r['condition']: r for r in q_records})})
    return {'document_id': paper['document_id'], 'questions': questions, 'rows': rows,
            'max_prompt_tokens': max(max(counts), previous.get('max_prompt_tokens', 0) if previous else 0),
            'author_model': 'gpt-6-luna'}

def question_signature(questions):
    return [(q['question'], q['options'], q['answer']) for q in questions]

def statistics(documents):
    rows = [row for d in documents for row in d['rows']]
    summary = {}
    for condition in CONDITIONS:
        correct = sum(row['predictions'][condition] == row['gold'] for row in rows)
        invalid = sum(row['predictions'][condition] is None for row in rows)
        summary[condition] = {'correct': correct, 'total': len(rows), 'invalid': invalid,
                              'accuracy': correct / len(rows) if rows else None}
    if not documents:
        return summary
    rng = random.Random(SEED)
    per_doc = [{c: sum(r['predictions'][c] == r['gold'] for r in d['rows'])
                for c in CONDITIONS} for d in documents]
    samples = {c: [] for c in CONDITIONS}
    samples['summary_minus_original'] = []
    pairs = [('summary', 'original')]
    if 'knowledge_units' in CONDITIONS:
        pairs += [('knowledge_units', 'original'), ('knowledge_units', 'summary')]
        samples['knowledge_units_minus_original'] = []
        samples['knowledge_units_minus_summary'] = []
    if 'qwen_summary' in CONDITIONS:
        for other in ('original', 'summary', 'knowledge_units'):
            if other in CONDITIONS:
                pairs.append(('qwen_summary', other))
                samples['qwen_summary_minus_' + other] = []
    for _ in range(10000):
        chosen = [rng.randrange(len(per_doc)) for _ in per_doc]
        acc = {c: sum(per_doc[i][c] for i in chosen) / (10 * len(chosen)) for c in CONDITIONS}
        for c in CONDITIONS:
            samples[c].append(acc[c])
        for left, right in pairs:
            samples[left + '_minus_' + right].append(acc[left] - acc[right])
    for c, values in samples.items():
        values.sort()
        if c not in summary:
            left, right = c.split('_minus_')
            summary[c] = {'accuracy': summary[left]['accuracy'] - summary[right]['accuracy']}
        summary[c]['ci95'] = [values[249], values[9749]]
    return summary

def main():
    global CONDITIONS
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--with-kus', action='store_true')
    parser.add_argument('--with-qwen-summaries', action='store_true')
    parser.add_argument('--context-limit', type=int, default=16384)
    parser.add_argument('--skip-id', action='append', default=[])
    parser.add_argument('--summary-cache', type=Path, default=ROOT / 'qwen_summaries.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results.json')
    parser.add_argument('--baseline-checkpoint', type=Path)
    args = parser.parse_args()
    if args.with_qwen_summaries and not args.with_kus:
        parser.error('The Qwen-summary comparison requires --with-kus to retain the preceding comparison.')
    existing_path = args.output
    if existing_path.exists():
        existing_protocol = load(existing_path).get('protocol', {})
        if 'knowledge_units' in existing_protocol.get('conditions', []) and not args.with_kus:
            raise SystemExit('This checkpoint includes KUs; use --with-kus to preserve all conditions.')
        if 'qwen_summary' in existing_protocol.get('conditions', []) and not args.with_qwen_summaries:
            raise SystemExit('This checkpoint includes generated summaries; preserve --with-qwen-summaries.')
    papers = load(ROOT / 'data' / 'papers.json')
    for index, paper in enumerate(papers):
        paper['_paper_index'] = index
    papers = [p for p in papers if p['document_id'] not in args.skip_id]
    if args.with_kus:
        CONDITIONS = ('no_context', 'original', 'summary', 'knowledge_units')
        cache = load(ROOT / 'kus.json')
        cached = {d['document_id']: d for d in cache['documents']}
        for paper in papers:
            if paper['document_id'] not in cached:
                raise ValueError('Missing KUs: ' + paper['document_id'])
            entry = cached[paper['document_id']]
            if entry['fulltext_sha256'] != paper['fulltext_sha256']:
                raise ValueError('KU source mismatch: ' + paper['document_id'])
            paper['knowledge_units'] = entry['judge_context']
    if args.with_qwen_summaries:
        CONDITIONS += ('qwen_summary',)
        summary_cache = load(args.summary_cache)
        generated = {d['document_id']: d for d in summary_cache['documents']}
        for paper in papers:
            if paper['document_id'] not in generated:
                raise ValueError('Missing Qwen summary: ' + paper['document_id'])
            entry = generated[paper['document_id']]
            if entry['fulltext_sha256'] != paper['fulltext_sha256']:
                raise ValueError('Summary source mismatch: ' + paper['document_id'])
            paper['qwen_summary'] = entry['judge_context']
    result_path = args.output
    result_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path = result_path if result_path.exists() else args.baseline_checkpoint
    results = load(seed_path) if seed_path else {
        'judge_model': 'Qwen/Qwen2.5-7B-Instruct', 'judge_alias': 'qwen25',
        'protocol': {'temperature': 0.5, 'top_p': 0.95, 'max_tokens': 100,
            'frequency_penalty': 1.05, 'presence_penalty': 1.05, 'attempts': 5,
            'parser': 'historical_semicolon_v1', 'legacy_ascii_sanitizer': True,
            'precision': 'BF16', 'runtime': 'vLLM 0.27.1', 'tensor_parallel_size': 1,
            'max_model_len': 16384, 'concurrency': 4, 'summary_condition': 'all_existing_fields',
            'qa_author': 'gpt-6-luna', 'questions_per_paper': 10,
            'bootstrap_seed': SEED, 'bootstrap_resamples': 10000},
        'documents': [], 'elapsed_judge_seconds': 0.0}
    results['protocol']['max_model_len'] = max(args.context_limit, results['protocol'].get('max_model_len', 0))
    results['protocol']['conditions'] = list(CONDITIONS)
    results['protocol']['excluded_ids'] = args.skip_id
    if args.with_kus:
        results['extractor'] = {k: cache[k] for k in ('model', 'revision', 'config', 'decoding', 'runtime', 'elapsed_seconds')}
    if args.with_qwen_summaries:
        results['qwen_summary_generator'] = dict(summary_cache['config'], elapsed_seconds=summary_cache['elapsed_seconds'])
    backend = OpenAICompatibleBackend('qwen25', base_url=args.base_url, api_key='',
        max_tokens=100, temperature=0.5, concurrency=4, thinking=False,
        frequency_penalty=1.05, presence_penalty=1.05)
    validation_errors = {}
    # Always perform one reconciliation pass, including on a completed checkpoint.
    # Author corrections must not silently leave stale predictions in a final report.
    while True:
        completed = {d['document_id']: d for d in results['documents']}
        progress = False
        for index, paper in enumerate(papers):
            try:
                questions = questions_for(paper, paper['_paper_index'])
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                if validation_errors.get(paper['document_id']) != str(error):
                    print('QA_VALIDATION_ERROR', paper['document_id'], str(error), flush=True)
                    validation_errors[paper['document_id']] = str(error)
                continue
            if questions is None:
                continue
            validation_errors.pop(paper['document_id'], None)
            previous = completed.get(paper['document_id'])
            if previous:
                if question_signature(previous['questions']) == question_signature(questions):
                    previous['questions'] = questions
                    if all(set(CONDITIONS) <= set(r['predictions']) for r in previous['rows']):
                        contexts = {'no_context': '', 'original': paper['fulltext'],
                                    'summary': paper['existing_summary'], 'knowledge_units': paper.get('knowledge_units', ''),
                                    'qwen_summary': paper.get('qwen_summary', '')}
                        if all(r['responses'][c]['prompt_sha256'] == hashlib.sha256(
                                historical_answer_prompt(q['formatted_question'], contexts[c]).encode()).hexdigest()
                                for q, r in zip(questions, previous['rows']) for c in CONDITIONS):
                            continue
                else:
                    print('QA_CHANGED_RERUN', paper['document_id'], flush=True)
                    previous = None
                results['documents'] = [d for d in results['documents']
                                        if d['document_id'] != paper['document_id']]
            started = time.time()
            document = run_document(paper, questions, backend, previous, args.context_limit)
            document['elapsed_judge_seconds'] = time.time() - started
            results['documents'].append(document)
            results['elapsed_judge_seconds'] += document['elapsed_judge_seconds']
            write_json_atomic(str(result_path), results)
            print('JUDGED', len(results['documents']), '/', len(papers), paper['document_id'], flush=True)
            progress = True
        if not args.watch or len(results['documents']) == len(papers):
            break
        if not progress:
            time.sleep(10)
    results['summary'] = statistics(results['documents'])
    results['subsets'] = {source: statistics([d for d in results['documents']
                                            if d['rows'][0]['source'] == source])
                          for source in ('arxiv', 'bethgelab')}
    write_json_atomic(str(result_path), results)
    qas = [{'document_id': d['document_id'], 'author_model': d['author_model'],
            'questions': d['questions']} for d in results['documents']]
    qa_path = ROOT / 'data' / 'qa_evaluated.json' if result_path == ROOT / 'results.json' else result_path.parent / 'qa_evaluated.json'
    write_json_atomic(str(qa_path), qas)
    print(json.dumps(results['summary'], indent=2), flush=True)

if __name__ == '__main__':
    main()
