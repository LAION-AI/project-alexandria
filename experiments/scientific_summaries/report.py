"""Render an English, offline HTML report and reproducible aggregate statistics."""
import argparse
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = {'no_context': 'No context', 'original': 'Full paper', 'summary': 'Existing summary'}

def esc(value):
    return html.escape(str(value), quote=True)

def percent(value):
    return f'{100 * value:.2f}%'

def main():
    global LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true', help='Create a clearly labelled incomplete snapshot.')
    partial = parser.parse_args().partial
    results = json.loads((ROOT / 'results.json').read_text(encoding='utf-8'))
    papers = json.loads((ROOT / 'data' / 'papers.json').read_text(encoding='utf-8'))
    paper_by_id = {p['document_id']: p for p in papers}
    evaluated = results['documents']
    with_kus = 'knowledge_units' in results.get('protocol', {}).get('conditions', [])
    with_qwen_summaries = 'qwen_summary' in results.get('protocol', {}).get('conditions', [])
    if with_kus:
        LABELS = dict(LABELS, knowledge_units='Qwen3.8-27B KUs')
    if with_qwen_summaries:
        LABELS = dict(LABELS, qwen_summary='Qwen3.8-27B summaries')
    if with_kus or with_qwen_summaries:
        evaluated = [d for d in evaluated if all(set(LABELS) <= set(r['predictions']) for r in d['rows'])]
    if not evaluated:
        raise SystemExit('No complete evaluation cohort is ready for a report yet.')
    if not partial and len(evaluated) != len(papers):
        raise SystemExit('Evaluation is incomplete; use --partial for a labelled snapshot.')
    suffix = '.partial' if partial else ''
    if partial:
        from evaluate import statistics as evaluation_statistics
        import evaluate
        evaluate.CONDITIONS = tuple(LABELS)
        results['documents'] = evaluated
        results['summary'] = evaluation_statistics(evaluated)
        results['subsets'] = {source: evaluation_statistics([
            d for d in evaluated if d['rows'][0]['source'] == source])
            for source in ('arxiv', 'bethgelab')}
        (ROOT / 'results.partial.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        qas = [{'document_id': d['document_id'], 'author_model': d['author_model'],
                'questions': d['questions']} for d in evaluated]
        (ROOT / 'data/qa_evaluated.partial.json').write_text(json.dumps(qas, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [r for d in evaluated for r in d['rows']]
    cards = []
    bars = []
    for condition, label in LABELS.items():
        item = results['summary'][condition]
        cards.append(f'<section class="card"><span>{label}</span><strong>{percent(item["accuracy"])}</strong>'
                     f'<small>{item["correct"]}/{item["total"]} correct · {item["invalid"]} invalid</small>'
                     f'<small>95% CI {percent(item["ci95"][0])}–{percent(item["ci95"][1])}</small></section>')
        bars.append(f'<div class="barrow"><span>{label}</span><div class="track"><div class="bar {condition}" '
                    f'style="width:{100*item["accuracy"]:.2f}%">{percent(item["accuracy"])}</div></div></div>')
    diff = results['summary']['summary_minus_original']
    gain = results['summary']['summary']['accuracy'] - results['summary']['no_context']['accuracy']
    subset_rows = []
    for source, summary in results['subsets'].items():
        if summary['original']['total']:
            cells = ''.join(f'<td>{percent(summary[c]["accuracy"])}</td>' for c in LABELS)
            subset_rows.append(f'<tr><td>{source}</td><td>{summary["original"]["total"]//10}</td>{cells}</tr>')
    original_lengths = [len(paper_by_id[d['document_id']]['fulltext'].split()) for d in evaluated]
    summary_lengths = [len(paper_by_id[d['document_id']]['existing_summary'].split()) for d in evaluated]
    question_blocks = []
    paper_rows = []
    for document in evaluated:
        p = paper_by_id[document['document_id']]
        scores = {c: sum(r['predictions'][c] == r['gold'] for r in document['rows']) for c in LABELS}
        details = []
        for q, row in zip(document['questions'], document['rows']):
            options = ''.join(f'<li class="{"gold" if k == q["answer"] else ""}">{k}: {esc(v)}</li>'
                              for k, v in q['options'].items())
            predictions = ' · '.join(f'{label}: {row["predictions"][c] or "invalid"}' for c, label in LABELS.items())
            details.append(f'<details class="question"><summary>Q{q["question_index"]+1}. {esc(q["question"])}</summary>'
                           f'<ul>{options}</ul><p><b>Gold: {q["answer"]}</b> · {predictions}</p>'
                           f'<p>{esc(q["rationale"])}</p><blockquote>{esc(q["evidence_quote"])}</blockquote>'
                           f'<small>Exact fulltext evidence: characters {q["evidence_start"]}–{q["evidence_end"]}; '
                           f'category {esc(q.get("category",""))}.</small></details>')
        identifier = p['document_id']
        cells = ''.join(f'<td>{scores[c]}/10</td>' for c in LABELS)
        paper_rows.append(f'<tr><td><a href="#{identifier}">{esc(p["source_title"])}</a></td>'
                          f'<td>{p["viewer_config"]}</td>{cells}</tr>')
        question_blocks.append(f'<article id="{identifier}" class="paper" data-search="{esc(p["source_title"].lower())}" '
                               f'data-source="{p["viewer_config"]}"><h3>{esc(p["source_title"])}</h3>'
                               f'<p class="muted">{identifier} · {esc(p.get("field_subfield",""))} · '
                               f'<a href="data/papers/{identifier}.txt">Saved full paper</a> · '
                               f'<a href="data/summaries/{identifier}.txt">Saved existing summary</a>'
                               + (f' · <a href="data/kus/{identifier}.json">Saved Qwen3.8-27B KUs</a>' if with_kus else '')
                               + (f' · <a href="data/qwen_summaries/{identifier}.json">Saved Qwen summary + evidence</a>' if with_qwen_summaries else '') + '</p>'
                               + ''.join(details) + '</article>')
    gold_counts = {letter: sum(r['gold'] == letter for r in rows) for letter in 'ABCD'}
    hard_subset = [r for r in rows if r['predictions']['original'] == r['gold']
                   and r['predictions']['no_context'] != r['gold']]
    retained = sum(r['predictions']['summary'] == r['gold'] for r in hard_subset)
    html_text = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Scientific Summaries — 100-paper evaluation</title>
<style>
:root{--ink:#102c3a;--muted:#536873;--teal:#087e8b;--line:#d7e3e8;--bg:#f4f8fa}*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;background:var(--bg);color:var(--ink);margin:0}main{max-width:1120px;margin:auto;padding:44px 24px}h1{font-size:clamp(30px,5vw,48px);line-height:1.15;margin:10px 0 20px}h2{margin-top:42px}a{color:var(--teal)}.eyebrow{letter-spacing:.14em;text-transform:uppercase;font-size:12px;color:var(--teal)}.muted,small{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}.card,.paper,.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:22px}.card span,.card strong,.card small{display:block}.card strong{font-size:38px;margin:8px 0}.barrow{display:flex;align-items:center;gap:18px;margin:14px 0}.barrow>span{min-width:155px}.track{background:#e7eef2;border-radius:6px;flex:1}.bar{background:var(--teal);color:white;padding:8px 12px;border-radius:6px;min-width:70px}.bar.no_context{background:#7c8fa0}.bar.original{background:#214b69}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line)}.scroll{overflow-x:auto;max-height:470px}.paper{margin:16px 0}.question{padding:13px 0;border-top:1px solid var(--line)}summary{cursor:pointer;font-weight:550}.question ul{list-style:none;padding-left:0}.question li{padding:5px 10px}.gold{background:#e2f4e9;border-radius:4px}blockquote{border-left:3px solid var(--teal);margin:12px 0;padding:8px 16px;background:#f0f7f9;white-space:pre-wrap}input,select{font:inherit;padding:11px;border:1px solid var(--line);border-radius:6px;margin-right:8px}.hidden{display:none}code{font-size:13px}footer{margin-top:40px;color:var(--muted)}@media(max-width:650px){.cards{grid-template-columns:1fr}.barrow>span{min-width:100px}main{padding:24px 14px}}@media print{input,select{display:none}.paper{break-inside:avoid}}
</style></head><body><main>
<div class="eyebrow">Project Alexandria · Existing summaries benchmark · 1 October 2026</div>
<h1>How much paper knowledge survives in an existing summary?</h1>
<p>__COUNT__ saved papers from <a href="https://huggingface.co/datasets/laion/Scientific-Summaries">laion/Scientific-Summaries</a>, ten Luna-authored multiple-choice questions per paper, and the same fixed Qwen2.5-7B answerer used in the previous Alexandria runs. __REPRESENTATION_NOTE__</p>
<div class="cards">__CARDS__</div><div class="panel">__BARS__</div>
<p>Summary minus full-paper accuracy: <b>__DIFF__ percentage points</b> (paired 95% CI __DIFFCI__). Summary improves on no context by __GAIN__ percentage points. On the diagnostic subset answered correctly with full text but incorrectly without context, summaries retain __RETAINED__/__HARD__ answers (__RETENTION__). This conditional measure uses the same judge responses and is not an independent difficulty certification.</p>
__KU_COMPARISON__
__QWEN_SUMMARY_COMPARISON__
__EXCLUDED_NOTE__
<h2>Source subsets</h2><table><thead><tr><th>Source</th><th>Papers</th>__CONDITION_HEADERS__</tr></thead><tbody>__SUBSETS__</tbody></table>
<h2>Method and limits</h2><div class="panel">
<p>The sample contains 50 arXiv and 50 Bethgelab entries selected reproducibly from five row-offset pools per source. Candidates required a nonempty body text, results and summary fields, 8,000–30,000 text characters, and at least 1,500 words. The primary pools also required a conclusion, discussion or references marker. This is a length-filtered convenience sample, not a random estimate of the complete dataset. PubMed and web-paper examples examined during selection lacked saved fulltext fields.</p>
<p><code>gpt-6-luna</code> agents authored ten questions per paper from saved fulltext, without access to existing summaries. Questions were intended to require paper-specific results, methods, limitations or mechanisms, with four plausible options and one gold answer. Each gold answer carries an exact supporting fulltext span and rationale. Option positions were rearranged deterministically, giving balanced gold letters: __GOLD__. Exact evidence presence is validated automatically; it does not establish that every question is unambiguous. No questions were excluded using judge scores.</p>
<p>The summary condition concatenates all existing substantive summary fields, including executive summary, methodology, results, limitations, claims and takeaways. Every selected summary is attributed by dataset metadata to <code>priv-gemini-2.0-flash-lite</code>. Median lengths: __TEXTWORDS__ full-paper words and __SUMWORDS__ summary words. The saved texts are the dataset’s available fulltexts; their fidelity to publisher PDFs was not independently verified. Dataset source and summary hashes are frozen in the manifest.</p>
<p>Judge: <code>Qwen/Qwen2.5-7B-Instruct</code>, BF16, vLLM 0.27.1; temperature 0.5, top-p 0.95, 100 output tokens, frequency and presence penalties 1.05. Each invalid format is retried up to four times; final invalids count as incorrect. The historical prompt, ASCII sanitizer and case-sensitive semicolon parser are reused. ASCII sanitization can remove scientific symbols and is retained for compatibility. Evidence and gold rationales are never passed to the judge.</p>
<p>The initial control pass used one RTX 3090 alongside an existing process, tensor parallelism 1, four request slots and a 16,384-token limit. __JUDGE_PHASE_NOTE__ Every rendered prompt was tokenized before evaluation; the largest was __MAXTOKENS__ tokens, and no context was truncated. These serving parameters differ from the earlier two-GPU deployment; model weights and scoring settings are unchanged. Judge processing took __SECONDS__ seconds. Confidence intervals use 10,000 paired document-cluster bootstrap resamples, seed 250219413, keeping each paper’s ten questions together.</p>
<p>QA author, dataset, and corpus differ from the earlier Physics/Medical benchmark; absolute scores should be compared with that limitation. Dataset card license: CC-BY-4.0; source-paper identifiers and source metadata are preserved in <a href="data/papers.json">the saved corpus</a>.</p>
</div>
<h2>Reproducibility files</h2><p><a href="data/manifest.json">Selection manifest</a> · <a href="data/papers.json">Papers + existing summaries</a> · <a href="data/qa_evaluated.json">Evaluated MCQs + evidence</a> · <a href="results.json">Predictions + attempts</a> · <a href="summary.csv">Summary CSV</a> · <a href="SHA256SUMS">SHA-256 checksums</a> · <a href="judge_model_fingerprints.json">Judge-weight fingerprints</a></p>
<p>The frozen corpus and evaluated QA files are the benchmark inputs. Running the live selection again is not an exact dataset-revision replay: the viewer endpoint can change. <a href="qa_authoring_prompt.txt">Saved authoring protocol</a> describes how the new questions were made. Stochastic generation and batching mean a rerun need not produce bitwise-identical judge responses.</p>
<pre><code># From the repository root, install the package and start the fixed judge:
python -m pip install -e .
# Set ALEXANDRIA_JUDGE_PYTHON, ALEXANDRIA_JUDGE_WEIGHTS, ALEXANDRIA_JUDGE_GPU
# to your local vLLM environment, BF16 Qwen2.5-7B directory, and free GPU.
bash experiments/scientific_summaries/serve_judge.sh
# In a second terminal:
python experiments/scientific_summaries/evaluate.py
python experiments/scientific_summaries/report.py
python experiments/scientific_summaries/validate.py
# Existing checkpoints are resumed; no completed unchanged question is rejudged.
# To obtain an independent stochastic rerun, use a copy without results.json.</code></pre>
<h2>Per-paper results</h2><div class="scroll"><table><thead><tr><th>Paper</th><th>Source</th>__CONDITION_HEADERS__</tr></thead><tbody>__PAPERROWS__</tbody></table></div>
<h2>Question and evidence explorer</h2><p>Open any question to inspect its options, gold rationale, exact supporting passage and predictions for every evaluated condition.</p>
<input id="search" type="search" placeholder="Search paper titles" aria-label="Search paper titles"><select id="subset" aria-label="Filter source"><option value="">All sources</option><option>arxiv</option><option>bethgelab</option></select><span id="visible"></span>
__QUESTIONS__
<footer>Prepared for Project Alexandria. All charts and interactions run locally; no external scripts or network requests are needed.</footer>
</main><script>function filter(){let q=document.getElementById('search').value.toLowerCase(),s=document.getElementById('subset').value,n=0;document.querySelectorAll('.paper').forEach(p=>{let yes=p.dataset.search.includes(q)&&(!s||p.dataset.source===s);p.classList.toggle('hidden',!yes);if(yes)n++});document.getElementById('visible').textContent=n+' papers shown'}document.getElementById('search').addEventListener('input',filter);document.getElementById('subset').addEventListener('change',filter);filter();</script></body></html>'''
    replacements = {
        '__REPRESENTATION_NOTE__': ('The dataset’s existing structured summaries are compared with newly extracted Qwen3.8-27B Knowledge Units on exactly the same papers and questions.' if with_kus else 'This experiment uses the dataset’s existing structured summaries. No Knowledge Units were generated in this baseline snapshot.'),
        '__CONDITION_HEADERS__': ''.join('<th>' + label + '</th>' for label in LABELS.values()),
        '__JUDGE_PHASE_NOTE__': ('After the GPU processes were stopped for extraction, the KU comparison resumed with the same BF16 one-GPU answerer and a 32,768-token limit. Existing unchanged control predictions were preserved; only missing conditions were generated.' if with_kus else ''),
        '__KU_COMPARISON__': '',
        '__QWEN_SUMMARY_COMPARISON__': '',
        '__EXCLUDED_NOTE__': ('<p class="panel">The following entries are pending a cohort decision after automated QA-authoring safety blocks: '
            + esc(', '.join(results['protocol'].get('excluded_ids', []))) + '. Their absent questions are not silently counted as evaluated papers. These exclusions were not based on student performance.</p>'
            if results.get('protocol', {}).get('excluded_ids') else ''),
        '__COUNT__': str(len(evaluated)), '__CARDS__': ''.join(cards), '__BARS__': ''.join(bars),
        '__DIFF__': f'{100*diff["accuracy"]:+.2f}',
        '__DIFFCI__': f'{100*diff["ci95"][0]:+.2f} to {100*diff["ci95"][1]:+.2f} pp',
        '__GAIN__': f'{100*gain:.2f}', '__RETAINED__': str(retained), '__HARD__': str(len(hard_subset)),
        '__RETENTION__': percent(retained / len(hard_subset)) if hard_subset else 'n/a',
        '__SUBSETS__': ''.join(subset_rows), '__GOLD__': esc(gold_counts),
        '__TEXTWORDS__': f'{statistics.median(original_lengths):,.0f}',
        '__SUMWORDS__': f'{statistics.median(summary_lengths):,.0f}',
        '__MAXTOKENS__': str(max(d['max_prompt_tokens'] for d in evaluated)),
        '__SECONDS__': f'{results["elapsed_judge_seconds"]:,.1f}',
        '__PAPERROWS__': ''.join(paper_rows), '__QUESTIONS__': ''.join(question_blocks)}
    if with_kus:
        comparisons = []
        for target, label in [('original', 'full paper'), ('summary', 'existing summary')]:
            item = results['summary']['knowledge_units_minus_' + target]
            comparisons.append(f'KU minus {label}: <b>{100*item["accuracy"]:+.2f} pp</b> '
                               f'(paired 95% CI {100*item["ci95"][0]:+.2f} to {100*item["ci95"][1]:+.2f} pp).')
        extractor = results['extractor']
        replacements['__KU_COMPARISON__'] = '<p>' + ' '.join(comparisons) + '</p>' + (
            '<h2>Parallel KU extraction</h2><div class="panel"><p>Extractor: '
            '<a href="https://huggingface.co/Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound">'
            'Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound</a>, pinned revision '
            + esc(extractor['revision']) + '. Independently extracted 500-word target chunks receive up to '
            '1,000 words on each side and the first 350 source words as document-opening context. '
            'No summaries, questions, gold answers or earlier chunk outputs enter extraction prompts. '
            'The repository’s few-shot templates and naming conventions are reused. Target prompts are '
            'flattened across batches of eight documents and continuously batched with ten concurrent '
            'requests. One document-level canonicalization pass reconciles entity names and relations. '
            'Extraction temperature is 0.2; target output cap 2,500 tokens, canonicalization cap 1,800. '
            'Malformed generations are repaired; failed batches are isolated and every completed batch '
            'is checkpointed. The judge receives factual KU summaries/entities/relations only, without '
            'source fingerprints or provenance.</p><p>Extraction used two RTX 3090s, vLLM 0.27.1, '
            '32,768-token context, BF16 compute with mixed-INT4 weights, thinking disabled. Measured '
            f'extraction time: {extractor["elapsed_seconds"]/3600:.3f} wall-hours '
            f'({2*extractor["elapsed_seconds"]/3600:.3f} allocated GPU-hours), excluding server startup. '
            '<a href="kus.json">Complete extraction cache</a>.</p></div>')
        html_text = html_text.replace('repeat(3,1fr)', 'repeat(4,1fr)')
        html_text = html_text.replace('How much paper knowledge survives in an existing summary?',
                                      'Paper knowledge: existing summaries vs. Knowledge Units')
        html_text = html_text.replace('bash experiments/scientific_summaries/serve_judge.sh',
            'ALEXANDRIA_JUDGE_GPU=0 ALEXANDRIA_JUDGE_MAX_LEN=32768 ALEXANDRIA_JUDGE_GPU_UTIL=0.90 bash experiments/scientific_summaries/serve_judge.sh')
        skip_flags = ''.join(' --skip-id ' + identifier for identifier in results['protocol'].get('excluded_ids', []))
        html_text = html_text.replace('python experiments/scientific_summaries/evaluate.py',
            'python experiments/scientific_summaries/evaluate.py --with-kus --context-limit 32768' + skip_flags)
    if with_qwen_summaries:
        config = results['qwen_summary_generator']
        differences = []
        for target, label in [('original', 'full paper'), ('summary', 'existing summary'), ('knowledge_units', 'Qwen KUs')]:
            item = results['summary']['qwen_summary_minus_' + target]
            differences.append(f'Qwen summary minus {label}: <b>{100*item["accuracy"]:+.2f} pp</b> '
                               f'(paired 95% CI {100*item["ci95"][0]:+.2f} to {100*item["ci95"][1]:+.2f} pp).')
        replacements['__QWEN_SUMMARY_COMPARISON__'] = (
            '<h2>Whole-paper Qwen summaries — user-supplied Schema-v4 prompt</h2><p>' + ' '.join(differences) +
            '</p><div class="panel"><p>The same pinned Qwen3.8-27B extractor receives each complete paper '
            'independently with the <a href="summary-systemprompt+.txt">user-supplied system prompt</a>. '
            'Only the surrounding Python literal assignment is removed; the prompt itself is unchanged. '
            'No QA questions, answers, existing summaries or KUs enter generation. Temperature '
            f'{config["temperature"]}, top-p {config["top_p"]}, thinking disabled, output budget '
            f'{config["max_tokens"]:,} tokens, {config["concurrency"]} concurrent document requests. '
            'Full inputs and repair messages are preflighted; no source text is truncated. Schema violations '
            f'are repaired up to {config["attempts"]} attempts and raw responses remain saved. '
            'Every retained evidence fragment is a nonempty source substring of at most five words. '
            'Whitespace/Unicode-ligature alignment is deterministic and logged; no fuzzy matching is used. '
            'This presence check does not prove semantic entailment of the narrative claim.</p><p>'
            'The student receives all substantive narrative fields, including claims and takeaways; '
            'short grounding quotes, metadata and citation rankings are retained as provenance but are '
            'not added to the student context. This parallels the source-free factual KU condition. '
            'The exact student-facing text is saved alongside the structured summary. '
            'Existing control/KU responses are reused only when their question and prompt hashes match. '
            'Summary generation took '
            f'{config["elapsed_seconds"]/3600:.3f} wall-hours '
            f'({2*config["elapsed_seconds"]/3600:.3f} allocated GPU-hours, excluding model startup). '
            '<a href="qwen_summaries.json">Complete generated-summary cache</a>.</p><p>'
            'Effective system-prompt SHA-256: <code>' + esc(config['system_prompt_sha256']) + '</code>.</p></div>')
        html_text = html_text.replace('--with-kus --context-limit 32768',
                                      '--with-kus --with-qwen-summaries --context-limit 32768')
        replacements['__REPRESENTATION_NOTE__'] = ('Existing dataset summaries, parallel Qwen3.8-27B KUs '
            'and new whole-paper Qwen3.8-27B summaries are compared using identical papers and questions.')
    html_text = html_text.replace('repeat(3,1fr)', 'repeat(auto-fit,minmax(190px,1fr))')
    html_text = html_text.replace('repeat(4,1fr)', 'repeat(auto-fit,minmax(190px,1fr))')
    html_text = html_text.replace('.bar.original{background:#214b69}',
        '.bar.original{background:#214b69}.bar.knowledge_units{background:#bd5d20}.bar.qwen_summary{background:#6757a2}')
    for old, new in replacements.items():
        html_text = html_text.replace(old, new)
    if partial:
        html_text = html_text.replace('<h1>', '<p class="panel"><b>INCOMPLETE SNAPSHOT: '
            + str(len(evaluated)) + '/100 papers evaluated. These are preliminary, not final results.</b></p><h1>', 1)
        for name in ('results.json', 'summary.csv', 'SHA256SUMS'):
            replacement = name.replace('.', '.partial.', 1) if '.' in name else name + '.partial'
            html_text = html_text.replace('href="' + name + '"', 'href="' + replacement + '"')
        html_text = html_text.replace('href="data/qa_evaluated.json"', 'href="data/qa_evaluated.partial.json"')
    (ROOT / ('report' + suffix + '.html')).write_text(html_text, encoding='utf-8')
    with (ROOT / ('summary' + suffix + '.csv')).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['subset', 'condition', 'questions', 'correct', 'invalid', 'accuracy', 'ci95_low', 'ci95_high'])
        for subset, summary in [('all', results['summary'])] + list(results['subsets'].items()):
            for condition in LABELS:
                s = summary[condition]
                if s['total']:
                    writer.writerow([subset, condition, s['total'], s['correct'], s['invalid'], s['accuracy'], *s['ci95']])
    paths = ['data/manifest.json', 'data/papers.json', 'data/qa_evaluated' + suffix + '.json',
             'results' + suffix + '.json', 'summary' + suffix + '.csv', 'report' + suffix + '.html',
             'prepare.py', 'evaluate.py', 'report.py',
             'validate.py', 'serve_judge.sh', 'qa_authoring_prompt.txt', 'capture_model.py',
             'judge_model_fingerprints.json']
    paths += ['extract.py', 'serve_extractor.sh', 'finish.py', 'summarize.py']
    if with_kus:
        paths.append('kus.json')
    if with_qwen_summaries:
        paths += ['qwen_summaries.json', 'summary-systemprompt+.txt']
    (ROOT / ('SHA256SUMS' + suffix)).write_text(''.join(
        hashlib.sha256((ROOT / p).read_bytes()).hexdigest() + '  ' + p + '\n' for p in paths), encoding='utf-8')
    print(ROOT / ('report' + suffix + '.html'))

if __name__ == '__main__':
    main()
