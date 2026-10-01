"""Audited, offline English dashboard for the three requested summary models."""
import csv
import hashlib
import html
import json
import random
import statistics
from datetime import datetime, timezone

from evaluate import ROOT, load, questions_for, historical_answer_prompt, SEED
from project_alexandria.io import write_json_atomic
from run_summary_comparison import MODELS
from summarize import validate_summary

LABELS = {'no_context': 'No context', 'original': 'Dataset paper text',
          'summary': 'Existing Gemini summary', 'knowledge_units': 'Qwen27B Knowledge Units'}


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return f'{100 * value:.2f}%'


def audit_run(model, results, cache, papers, baseline):
    """Check exact source evidence and all student prompts; never merely trust the metrics."""
    selected = load(ROOT / 'summary_comparison_manifest.json')['documents']
    by_id = {d['document_id']: d for d in results['documents']}
    generated = {d['document_id']: d for d in cache['documents']}
    old = {d['document_id']: d for d in baseline['documents']}
    kus = {d['document_id']: d for d in load(ROOT / 'kus.json')['documents']}
    if len(by_id) != len(results['documents']) or set(by_id) != set(selected):
        raise ValueError('An evaluated model must cover exactly the fixed 97-paper cohort')
    if set(generated) != set(selected):
        raise ValueError('A summary model silently omitted or added documents')
    if cache['config']['model'] != model['model'] or cache['config']['revision'] != model['revision']:
        raise ValueError('Summary model identity mismatch')
    for index, paper in enumerate(papers):
        identifier = paper['document_id']
        if identifier not in by_id:
            continue
        document, summary = by_id[identifier], generated[identifier]
        if summary['fulltext_sha256'] != paper['fulltext_sha256']:
            raise ValueError('Summary source hash mismatch')
        _, narrative, spans = validate_summary(json.dumps(summary['summary'], ensure_ascii=False), paper['fulltext'])
        if narrative != summary['judge_context']:
            raise ValueError('Student-facing summary differs from the validated narrative')
        for span in summary['evidence_spans']:
            if paper['fulltext'][span['start']:span['end']] != span['quote'] or not 0 < len(span['quote'].split()) <= 5:
                raise ValueError('Invalid retained evidence span')
        if len(summary['evidence_spans']) != len(spans):
            raise ValueError('Evidence ledger is incomplete')
        questions = questions_for(paper, index)
        if document['questions'] != questions or len(document['rows']) != 10:
            raise ValueError('Questions changed or row count invalid')
        contexts = {'no_context': '', 'original': paper['fulltext'], 'summary': paper['existing_summary'],
                    'knowledge_units': kus[identifier]['judge_context'], 'qwen_summary': narrative}
        for question, row, previous in zip(questions, document['rows'], old[identifier]['rows']):
            if row['gold'] != question['answer'] or set(row['predictions']) != set(contexts):
                raise ValueError('Student output conditions or gold key mismatch')
            for condition, context in contexts.items():
                record = row['responses'][condition]
                expected = hashlib.sha256(historical_answer_prompt(question['formatted_question'], context).encode()).hexdigest()
                if record['prompt_sha256'] != expected:
                    raise ValueError('Student prompt hash mismatch')
                if record['prompt_tokens'] + 100 > record.get('context_limit', 16384):
                    raise ValueError('Student context overflow')
                if row['predictions'][condition] not in (None, 'A', 'B', 'C', 'D'):
                    raise ValueError('Invalid student choice')
                if condition != 'qwen_summary' and (record != previous['responses'][condition]
                        or row['predictions'][condition] != previous['predictions'][condition]):
                    raise ValueError('The fixed controls were unexpectedly regenerated')
    return by_id, generated


def paired(left, right):
    if set(left) != set(right):
        raise ValueError('Paired model comparison requires identical paper IDs')
    identifiers = sorted(left)
    differences = [left[i] - right[i] for i in identifiers]
    rng = random.Random(SEED)
    values = sorted(sum(differences[rng.randrange(len(differences))] for _ in differences)
                    / (10 * len(differences)) for _ in range(10000))
    return {'difference': sum(differences) / (10 * len(differences)), 'ci95': [values[249], values[9749]]}


def main():
    manifest = load(ROOT / 'summary_comparison_manifest.json')
    papers = load(ROOT / 'data/papers.json')
    by_paper = {p['document_id']: p for p in papers}
    baseline = load(ROOT / 'pre_qwen_summary_results.json')
    evaluated = {}
    progress = {}
    for model in MODELS:
        directory = ROOT / 'summary_runs' / model['name']
        cache_path, result_path = directory / 'summaries.json', directory / 'results.json'
        cache = load(cache_path) if cache_path.exists() else None
        progress[model['name']] = {'validated_summaries': len(cache['documents']) if cache else 0,
            'failed_generation_records': len(cache['failures']) if cache else 0, 'evaluated': False}
        if result_path.exists():
            results = load(result_path)
            if 'summary' not in results or len(results['documents']) != 97:
                continue
            by_id, generated = audit_run(model, results, cache, papers, baseline)
            if evaluated and cache['config']['system_prompt_sha256'] != next(iter(evaluated.values()))['config']['system_prompt_sha256']:
                raise ValueError('Summary models used different user system prompts')
            evaluated[model['name']] = dict(model=model, results=results, by_id=by_id,
                generated=generated, config=cache['config'], elapsed_seconds=cache['elapsed_seconds'])
            progress[model['name']]['evaluated'] = True
    comparisons = {}
    names = list(evaluated)
    for index, name in enumerate(names):
        left = {i: sum(row['predictions']['qwen_summary'] == row['gold'] for row in d['rows'])
                for i, d in evaluated[name]['by_id'].items()}
        for other in names[:index]:
            right = {i: sum(row['predictions']['qwen_summary'] == row['gold'] for row in d['rows'])
                     for i, d in evaluated[other]['by_id'].items()}
            comparisons[name + '_minus_' + other] = paired(left, right)
    output = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'papers': 97, 'questions': 970,
        'complete': len(evaluated) == len(MODELS), 'progress': progress,
        'baseline': baseline['summary'], 'models': {name: dict(config=entry['config'],
        elapsed_seconds=entry['elapsed_seconds'], summary=entry['results']['summary'],
        subsets=entry['results']['subsets']) for name, entry in evaluated.items()},
        'paired_model_differences': comparisons}
    write_json_atomic(str(ROOT / 'summary_comparison.json'), output)
    cards = []
    table_rows = []
    for condition, label in LABELS.items():
        item = baseline['summary'][condition]
        table_rows.append(f'<tr><td>{esc(label)}</td><td>{pct(item["accuracy"])}</td>'
            f'<td>{item["correct"]}/970</td><td>{pct(item["ci95"][0])}–{pct(item["ci95"][1])}</td><td>Fixed control</td></tr>')
    method_rows = []
    for model in MODELS:
        name = model['name']
        if name in evaluated:
            entry = evaluated[name]; item = entry['results']['summary']['qwen_summary']
            hours = entry['elapsed_seconds'] / 3600
            words = statistics.median(len(d['judge_context'].split()) for d in entry['generated'].values())
            repaired = sum(len(d['attempts']) > 1 for d in entry['generated'].values())
            result = pct(item['accuracy'])
            table_rows.append(f'<tr><td>{esc(model["model"])}</td><td>{result}</td><td>{item["correct"]}/970</td>'
                f'<td>{pct(item["ci95"][0])}–{pct(item["ci95"][1])}</td><td>Summary</td></tr>')
            info = f'{hours:.2f} generation wall-hours / {hours * model["allocated_gpus"]:.2f} allocated GPU-hours; median {words:,.0f} narrative words; {repaired}/97 papers repaired.'
            method_rows.append(f'<tr><td>{esc(name)}</td><td>{esc(model["runtime"])}</td><td>{esc(info)}</td></tr>')
        else:
            result = 'Pending'
        cards.append(f'<section><h3>{esc(name)}</h3><strong>{result}</strong><p>{progress[name]["validated_summaries"]}/97 validated summaries</p></section>')
    pair_rows = ''.join(f'<tr><td>{esc(name)}</td><td>{100*item["difference"]:+.2f} pp</td>'
        f'<td>{100*item["ci95"][0]:+.2f} to {100*item["ci95"][1]:+.2f} pp</td></tr>' for name, item in comparisons.items())
    questions = []
    base_by_id = {d['document_id']: d for d in baseline['documents']}
    for identifier in manifest['documents']:
        paper = by_paper[identifier]; document = base_by_id[identifier]
        links = f'<a href="data/papers/{identifier}.txt">Dataset text</a> · <a href="data/summaries/{identifier}.txt">Existing summary</a> · <a href="data/kus/{identifier}.json">KUs</a>'
        for name in evaluated:
            links += f' · <a href="summary_runs/{name}/documents/{identifier}.json">{esc(name)} summary, evidence and repair journal</a>'
        details = []
        for index, (question, row) in enumerate(zip(document['questions'], document['rows'])):
            predictions = [f'{LABELS[c]}: {row["predictions"][c] or "invalid"}' for c in LABELS]
            predictions += [f'{name}: {entry["by_id"][identifier]["rows"][index]["predictions"]["qwen_summary"] or "invalid"}' for name, entry in evaluated.items()]
            options = ''.join(f'<li>{letter}: {esc(value)}</li>' for letter, value in question['options'].items())
            details.append(f'<details><summary>Q{index+1}: {esc(question["question"])}</summary><ul>{options}</ul>'
                f'<p>Gold: {question["answer"]} · {esc(" · ".join(predictions))}</p><p>{esc(question["rationale"])}</p>'
                f'<blockquote>{esc(question["evidence_quote"])}</blockquote></details>')
        questions.append(f'<article><h3>{esc(paper["source_title"])}</h3><p>{links}</p>' + ''.join(details) + '</article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alexandria — Three-model scientific summary comparison</title><style>
body{font:16px/1.6 system-ui;color:#183342;background:#f5f8fa;margin:0}main{max-width:1180px;margin:auto;padding:32px 24px}h1{font-size:36px;line-height:1.2}a{color:#087f8c}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}section,article,.panel{background:white;border:1px solid #d4e1e7;border-radius:12px;padding:20px;margin:16px 0}strong{font-size:30px}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;border-bottom:1px solid #d4e1e7;padding:12px}details{border-top:1px solid #d4e1e7;padding:12px 0}summary{cursor:pointer}blockquote{white-space:pre-wrap;background:#eef6f8;border-left:3px solid #087f8c;padding:12px}.scroll{overflow:auto}code{word-break:break-all}small{color:#536e7b}</style><main>
<small>Project Alexandria · Evidence-grounded summaries · Updated __DATE__</small><h1>Scientific knowledge retained by three summary models</h1>
<p>__STATE__ The fixed cohort contains 97 dataset papers and 970 Luna-authored, four-choice MCQs. Three original cohort papers without QA remain explicitly excluded; no paper was selected or excluded by student performance.</p>
<div class="cards">__CARDS__</div><h2>QA accuracy with the identical fixed student</h2><div class="scroll"><table><tr><th>Representation</th><th>Accuracy</th><th>Correct</th><th>95% document-cluster CI</th><th>Condition</th></tr>__ROWS__</table></div>
<h2>Paired model differences</h2><table><tr><th>Comparison</th><th>Difference</th><th>Paired 95% CI</th></tr>__PAIRS__</table>
<h2>Generation, repairs, and comparability</h2><div class="panel"><p>Every model receives the same <a href="summary-systemprompt+.txt">unchanged user system prompt</a> and saved dataset text, never MCQs, gold answers, existing summaries or KUs. Settings: temperature 0.2, top-p 0.95, thinking disabled, 16,000 output tokens, a 32,768-token context cap. Inputs and repairs are tokenized before inference; source text is never truncated. Context availability is the dataset text, not independently verified publisher PDFs. The two initial diagnostic papers had no bibliography, so absent citation fields must be empty rather than fabricated.</p>
<p>JSON-object constrained decoding prevents code fences; a conservative parser also recognizes a single enclosing fence and records its removal. Invalid fields are repaired by the same model using a separate, saved evidence-repair prompt and the original text. The primary stage has up to six calls; repair responses can modify only requested fields and have a 4,096-token budget. A second bounded stage (up to six calls) proposes real source fragments through lexical retrieval and asks the same model to choose supporting candidate indices or explicitly drop unsupported entries. Retrieval is not fuzzy validation: every selected quote is an actual source span, with candidates, offsets, selections and removals logged. Calls process at most eight quote tasks without truncating the source. Schema errors still get field-only patches. A failed cohort pass is resumed once; unresolved failures stop the queue, never silently dropping papers. Changing field order is logged. Every retained quote must match an actual source span and have at most five words. PDF whitespace/ligature alignment is logged, but mathematical substitutions are not accepted. Quote presence does not prove semantic entailment, and unsupported-entry removals can shorten the representation.</p>
<p>The fixed student remains <code>Qwen/Qwen2.5-7B-Instruct</code>, BF16, vLLM 0.27.1, temperature 0.5, top-p 0.95, maximum 100 output tokens, frequency/presence penalties 1.05, historical ASCII sanitizer and semicolon parser. Five total format attempts; final invalid answers count as wrong. Frozen original/no-context/existing-summary/KU controls are reused only when question and input hashes match. The student receives substantive generated narrative, not evidence quotes or citation rankings. All model results use identical source IDs, questions and answer permutations.</p>
<p>27B uses mixed-INT4 weights on two RTX 3090s with vLLM and eight concurrent requests. Both requested 9B GGUFs use Q8_0 weights with llama.cpp on one RTX 3090 and four concurrent slots; KV cache is Q8_0. GGUF revisions and SHA-256 hashes are pinned in the manifest. Runtime, quantization, and batching differ; this is an end-to-end representation comparison, not an isolated model-weight ablation. Word targets in the prompt are not guaranteed; observed narrative lengths and repair rates are disclosed. Timing includes generation and repair attempts, excludes server startup, and allocated GPU-hours are not measured GH200/Jupiter timings.</p>
<p>Uncertainty uses 10,000 paired document-cluster bootstrap samples, seed 250219413, keeping each paper’s ten questions together. Differences within those intervals are not established improvements. This 97-paper convenience cohort and Luna-authored questions differ from the previous Physics/Medical benchmark.</p></div>
<table><tr><th>Model</th><th>Runtime</th><th>Measured generation</th></tr>__METHODS__</table>
<h2>Reproducibility</h2><p><a href="summary_comparison_manifest.json">Pinned cohort and model manifest</a> · <a href="summary_comparison.json">Aggregate results and progress</a> · <a href="summary_comparison.csv">CSV</a> · <a href="summary_comparison.SHA256SUMS">Checksums</a> · <a href="summary_runtime_fingerprints.json">Runtime and GGUF fingerprints</a> · <a href="gguf_smoke_results.json">CPU-only GGUF transport checks</a> · <a href="pre_qwen_summary_results.json">Frozen four-condition controls</a> · <a href="data/papers.json">Corpus and existing summaries</a> · <a href="https://arxiv.org/html/2502.19413v2">Alexandria paper</a></p><p>Resume from the repository root: <code>python experiments/scientific_summaries/run_summary_comparison.py</code>. Each model has separate summaries, source-only prompt snapshots, raw responses, token usage, student attempts and audited results under <code>summary_runs/</code>. Dataset: <a href="https://huggingface.co/datasets/laion/Scientific-Summaries">laion/Scientific-Summaries</a>, declared CC-BY-4.0; source metadata remain saved. Model cards: <a href="https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF">Ornith</a>, <a href="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF">Qwen3.5</a>.</p>
<h2>All questions, source evidence, and per-model predictions</h2>__QUESTIONS__</main></html>'''
    replacements = {'__DATE__': esc(output['updated_utc']), '__STATE__': 'Complete.' if output['complete'] else 'Work in progress; pending scores are not reported as zero.',
        '__CARDS__': ''.join(cards), '__ROWS__': ''.join(table_rows), '__PAIRS__': pair_rows,
        '__METHODS__': ''.join(method_rows), '__QUESTIONS__': ''.join(questions)}
    for key, value in replacements.items():
        page = page.replace(key, value)
    (ROOT / 'summary_comparison.html').write_text(page, encoding='utf-8')
    with (ROOT / 'summary_comparison.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model', 'condition', 'questions', 'correct', 'invalid', 'accuracy', 'ci95_low', 'ci95_high'])
        for name, summary in [('baseline', baseline['summary'])] + [(n, e['results']['summary']) for n, e in evaluated.items()]:
            for condition, item in summary.items():
                if 'total' in item:
                    writer.writerow([name, condition, item['total'], item['correct'], item['invalid'], item['accuracy'], *item['ci95']])
    paths = ['summary_comparison.html', 'summary_comparison.json', 'summary_comparison.csv',
        'summary_comparison_manifest.json', 'summary-systemprompt+.txt', 'data/papers.json',
        'data/manifest.json', 'pre_qwen_summary_results.json', 'judge_model_fingerprints.json',
        'summary_runtime.py', 'summarize.py', 'evaluate.py', 'run_summary_comparison.py',
        'summary_comparison_report.py', 'summary_runtime_fingerprints.json', 'gguf_smoke_results.json',
        'capture_summary_runtime.py', 'publish_summary_results.py', 'README.md']
    for name in evaluated:
        paths += ['summary_runs/' + name + '/' + file for file in
            ('summaries.json', 'results.json', 'qa_evaluated.json', 'summary_prompt_snapshot.json')]
    (ROOT / 'summary_comparison.SHA256SUMS').write_text(''.join(
        hashlib.sha256((ROOT / path).read_bytes()).hexdigest() + '  ' + path + '\n' for path in paths))
    print(ROOT / 'summary_comparison.html', 'evaluated models', list(evaluated), flush=True)


if __name__ == '__main__':
    main()
