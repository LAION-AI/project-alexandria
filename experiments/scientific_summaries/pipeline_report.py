"""Build a standalone methods report from frozen evidence; never start inference.

Standard-library only. Prompts are read as literals/AST, not imported or executed.
This writes only its three dedicated documentation outputs, never run checkpoints.
"""
import argparse
import ast
import hashlib
import html
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
URL = 'https://github.com/LAION-AI/project-alexandria'
AS_OF = '2026-10-02'
SOURCES = {
    'jupiter': 'https://apps.fz-juelich.de/jsc/hps/jupiter/configuration.html',
    'qwen27_recipe': 'https://recipes.vllm.ai/Qwen/Qwen3.8-27B',
    'qwen9_recipe': 'https://recipes.vllm.ai/Qwen/Qwen3.5-9B',
    'qwen9_benchmark': 'https://docs.gpustack.ai/2.1/performance-lab/qwen3.5-9b/h100/',
    'qwen27_benchmark': 'https://www.reddit.com/r/openrouter/comments/1woqt5j/qwen3827b_on_1_h200_202_toks_at_low_concurrency/',
    'prefill_benchmark': 'https://sgl-project.github.io/SpecForge/recipes/qwen3.8-27b-dflash2-disaggregated.html',
    'kv_cache': 'https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/',
    'qwen27_fp8': 'https://huggingface.co/Qwen/Qwen3.8-27B-FP8',
    'qwen9_fp8': 'https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic',
}
RATES = {
    'Qwen3.8-27B FP8': {'fast': (16000, 1500, .90), 'central': (10000, 900, .85),
                       'slow': (6000, 700, .80)},
    'Qwen3.5-9B FP8': {'fast': (40000, 4000, .90), 'central': (25000, 2500, .85),
                      'slow': (15000, 1500, .80)},
}


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode('utf-8')).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def esc(value):
    return html.escape(str(value), quote=True)


def literal(path, name):
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Literal not found: ' + name)


def function(path, name):
    text = path.read_text(encoding='utf-8')
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return text, node
    raise ValueError('Function not found: ' + name)


def assignment(node, name, occurrence=0):
    found = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    found.sort(key=lambda n: n.lineno)
    return found[occurrence].value


def request_template(node):
    """Render only known literal concatenations, preserving literal whitespace."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return request_template(node.left) + request_template(node.right)
    if isinstance(node, ast.Name) and node.id == 'source':
        return '{paper_text}'
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'json' and node.func.attr == 'dumps':
            arg = node.args[0]
            names = {'fields': 'fields_to_repair_json', 'errors': 'errors_json', 'public_tasks': 'tasks_json'}
            key = names.get(arg.id) if isinstance(arg, ast.Name) else 'fields_to_repair_json'
            if key is not None:
                return '{' + key + '}'
        if node.func.attr == 'join':
            return '{all_detected_quote_errors_joined_with_newlines}'
    raise ValueError('Unrecognized prompt expression: ' + ast.dump(node))


def prompt_inventory():
    runtime = ROOT / 'summary_runtime.py'
    source, generation = function(runtime, 'generate_document')
    _, anchored = function(runtime, 'repair_saved_document')
    mcq = REPO / 'src/project_alexandria/experiments/mcq.py'
    mcq_source, historical = function(mcq, 'historical_answer_prompt')
    expression = assignment(historical, 'prompt')
    judge_template = ast.literal_eval(expression.func.value)
    snapshot = load(ROOT / 'summary_runs/qwen27b/summary_prompt_snapshot.json')
    effective = literal(ROOT / 'summary-systemprompt+.txt', 'SYSTEM_PROMPT_SUMMARY')
    if effective != snapshot['effective_system_prompt'] or sha(effective) != snapshot['effective_prompt_sha256']:
        raise ValueError('Effective summary prompt differs from the completed run')
    if sha((ROOT / 'summary-systemprompt+.txt').read_bytes()) != snapshot['file_sha256']:
        raise ValueError('Original prompt file bytes changed')
    repairs = literal(runtime, 'REPAIR_SYSTEM')
    anchors = literal(runtime, 'ANCHOR_SYSTEM')
    if repairs != snapshot['repair_system_prompt'] or anchors != snapshot['anchor_selection_system_prompt']:
        raise ValueError('Repair prompts differ from the completed run')
    suffix = next(ast.literal_eval(n.value) for n in ast.walk(generation)
                  if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == 'user')
    items = [
        ('summary-system', 'A1. Initial summary system prompt — exact effective text', effective,
         'summary-systemprompt+.txt', 'The literal Python assignment wrapper is not sent. UI-copy artifacts in the original prompt are intentionally retained.'),
        ('summary-user', 'A2. Initial summary user message', request_template(assignment(generation, 'user')),
         'summary_runtime.py', 'Replace {paper_text} with the unchanged frozen dataset text.'),
        ('summary-retry', 'A3. Initial-generation format retry suffix', suffix,
         'summary_runtime.py', 'Appended to the current initial user message if no complete draft exists; repeated failures can append it again.'),
        ('repair-system', 'A4. Field-repair system prompt', repairs,
         'summary_runtime.py', 'Same generator model; distinct source-only repair instructions.'),
        ('repair-user', 'A5. Primary field-repair user message', request_template(assignment(generation, 'request')),
         'summary_runtime.py', 'Insert Python json.dumps output for the requested fields/errors (ensure_ascii=False). The final error list is newline-joined grounding_errors output.'),
        ('anchor-system', 'A6. Source-anchor selection system prompt', anchors,
         'summary_runtime.py', 'The model chooses a provided candidate index or "drop"; it cannot supply a new quote.'),
        ('anchor-user', 'A7. Source-anchor selection user message', request_template(assignment(anchored, 'user', 0)),
         'summary_runtime.py', 'tasks_json contains at most eight tasks, each with id, narrative, invalid_quote, and candidates with explicit index, quote, start, end. Internal path is not sent.'),
        ('secondary-repair-user', 'A8. Secondary-stage field-repair user message', request_template(assignment(anchored, 'user', 1)),
         'summary_runtime.py', 'Used for remaining non-quote/schema/citation errors; system prompt A4 is reused. errors_json uses json.dumps default ASCII escaping in this branch.'),
        ('qa-authoring', 'A9. QA-authoring protocol', (ROOT / 'qa_authoring_prompt.txt').read_text(encoding='utf-8'),
         'qa_authoring_prompt.txt', 'Saved task protocol for gpt-6-luna authors. Per-document assignments and subsequent review dialogue are not a fully archived agent transcript. Reuse frozen QA for exact comparison.'),
        ('judge-system', 'A10. Fixed-student system prompt', literal(REPO / 'src/project_alexandria/experiments/reproduce.py', 'JUDGE_SYSTEM_PROMPT'),
         '../../src/project_alexandria/experiments/reproduce.py', 'Historical wording is retained exactly.'),
        ('judge-user', 'A11. Fixed-student user template — before historical sanitization', judge_template,
         '../../src/project_alexandria/experiments/mcq.py', 'Replace {context} and {question}, then apply historical_sanitize. Formatting retries send the same prompt, not a corrective prompt.'),
        ('judge-sanitizer', 'A12. Exact historical sanitizer and choice parser',
         '\n\n'.join(ast.get_source_segment(mcq_source, function(mcq, n)[1])
                       for n in ('historical_sanitize', 'extract_historical_choice')),
         '../../src/project_alexandria/experiments/mcq.py', 'Executable definitions, not model prompts. Included because punctuation removal changes the effective student input.'),
    ]
    for title, node in [('Primary initial message builder', assignment(generation, 'user')),
                        ('Primary repair message builder', assignment(generation, 'request')),
                        ('Anchor message builder', assignment(anchored, 'user', 0)),
                        ('Secondary field message builder', assignment(anchored, 'user', 1))]:
        items.append(('builder-' + str(len(items)), title, ast.get_source_segment(source, node),
                      'summary_runtime.py', 'Exact expression from the implementation; use it to reproduce JSON escaping and whitespace.'))
    v3 = ROOT / 'summary_repair_v3.py'
    if v3.exists():
        items.extend([
            ('v3-field-user', 'A13. V3 single-field repair user template', literal(v3, 'FIELD_USER_TEMPLATE'),
             'summary_repair_v3.py', 'Restarted 9B policy only. Fill source, field, value, error, and example. REPAIR_SYSTEM (A4) is unchanged.'),
            ('v3-anchor-user', 'A14. V3 source-anchor user template', literal(v3, 'ANCHOR_USER_TEMPLATE'),
             'summary_repair_v3.py', 'Restarted 9B policy only. The tasks also include candidate source_context. ANCHOR_SYSTEM (A6) is unchanged.'),
            ('v3-field-examples', 'A15. V3 exact field-shape example builder',
             ast.get_source_segment(v3.read_text(encoding='utf-8'), function(v3, 'field_example')[1]),
             'summary_repair_v3.py', 'Produces the syntax-only example inserted in A13; concrete per-field examples are saved in V3 prompt snapshots.'),
        ])
    return items


def measured_workload(cache, papers):
    documents = cache['documents']
    if not documents or len({d['document_id'] for d in documents}) != len(documents):
        raise ValueError('Empty or duplicated completed cohort')
    by_id = {p['document_id']: p for p in papers}
    for d in documents:
        if sha(by_id[d['document_id']]['fulltext']) != d['fulltext_sha256']:
            raise ValueError('Source text changed')
    initial = [next(a for a in d['attempts'] if a['phase'] == 'generation') for d in documents]
    seen, attempts = set(), []
    for d in documents + cache.get('failures', []):
        for a in d.get('attempts', []):
            # Saved failure attempts may be copied into the eventual successful journal.
            # Elapsed duration distinguishes separately executed otherwise identical calls.
            key = (d['document_id'], a.get('phase'), a.get('seed'), a.get('user_prompt_sha256'),
                   a.get('system_prompt_sha256'), sha(a.get('response', '')), a.get('elapsed_seconds'))
            if key not in seen:
                seen.add(key)
                attempts.append(a)
    if any(not all(isinstance(a.get('usage', {}).get(k), int)
                   for k in ('prompt_tokens', 'completion_tokens')) for a in initial + attempts):
        raise ValueError('Missing measured token counts; cannot silently estimate them')
    count = len(documents)
    mean_tokens = lambda records, key: sum(a['usage'][key] for a in records) / count
    return {'documents': count, 'distinct_saved_calls': len(attempts),
            'initial_input_tokens': mean_tokens(initial, 'prompt_tokens'),
            'initial_output_tokens': mean_tokens(initial, 'completion_tokens'),
            'validated_input_tokens': mean_tokens(attempts, 'prompt_tokens'),
            'validated_output_tokens': mean_tokens(attempts, 'completion_tokens'),
            'successful_journal_calls_per_paper': sum(len(d['attempts']) for d in documents) / count,
            'source_words_mean': statistics.mean(len(by_id[d['document_id']]['fulltext'].split()) for d in documents),
            'narrative_words_mean': statistics.mean(len(d['judge_context'].split()) for d in documents),
            'checkpointed_active_wall_hours': cache['elapsed_seconds'] / 3600,
            'checkpointed_gpu_hours_proxy': cache['elapsed_seconds'] / 3600 * cache['config']['allocated_gpus'],
            'saved_failure_records': len(cache.get('failures', []))}


def gpu_hours(count, input_tokens, output_tokens, rates):
    prefill, decode, utilization = rates
    if min(count, input_tokens, output_tokens, prefill, decode, utilization) <= 0 or utilization > 1:
        raise ValueError('Invalid scaling assumptions')
    return count / 3600 * (input_tokens / prefill + output_tokens / decode) / utilization


def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + str(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def link(path, label=None):
    return '<a href="' + esc(path) + '">' + esc(label or path) + '</a>'


def build():
    cache = load(ROOT / 'summary_runs/qwen27b/summaries.json')
    results = load(ROOT / 'summary_runs/qwen27b/results.json')
    papers = load(ROOT / 'data/papers.json')
    manifest = load(ROOT / 'summary_comparison_manifest.json')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(REPO), text=True).strip()
    inventory = prompt_inventory()
    work = measured_workload(cache, papers)
    if set(manifest['documents']) != {d['document_id'] for d in cache['documents']}:
        raise ValueError('Completed run does not match the frozen comparison cohort')
    cost_rows, wall_rows, estimates, rate_rows = [], [], [], []
    for model, rates in RATES.items():
        rate_rows.append([esc(model), f"{rates['central'][0]:,}", f"{rates['central'][1]:,}", '85%'])
        for condition, prefix in [('One generation per paper', 'initial'), ('Observed validation + repair workload', 'validated')]:
            costs = {k: gpu_hours(60000000, work[prefix + '_input_tokens'], work[prefix + '_output_tokens'], v)
                     for k, v in rates.items()}
            estimates.append(dict(model=model, workload=condition, papers=60000000, gpu_hours=costs))
            cost_rows.append([esc(model), esc(condition), f"<strong>{costs['central']:,.0f}</strong>",
                              f"{costs['fast']:,.0f}–{costs['slow']:,.0f}"])
            wall_rows.append([esc(model), esc(condition), f"{costs['central'] / 1024:.1f} h",
                              f"{costs['central'] / 4096:.1f} h"])
    labels = [('no_context', 'No context'), ('original', 'Frozen dataset paper text'),
              ('summary', 'Existing dataset summary'), ('knowledge_units', 'Existing Qwen27B KUs'),
              ('qwen_summary', 'New grounded Qwen3.8-27B summary')]
    score_rows = []
    for key, label in labels:
        s = results['summary'][key]
        score_rows.append([esc(label), f"{s['correct']} / {s['total']}", f"{s['accuracy'] * 100:.2f}%",
                           f"{s['ci95'][0] * 100:.2f}–{s['ci95'][1] * 100:.2f}%", str(s['invalid'])])
    model_rows = []
    for m in manifest['models']:
        model_rows.append([link('https://huggingface.co/' + m['model'], m['model']), esc(m['runtime']),
                           str(m['allocated_gpus']), str(m['concurrency']), '<code>' + esc(m['revision']) + '</code>'])
    appendix = ''
    for identifier, title, text, path, note in inventory:
        appendix += ('<details id="' + identifier + '" open><summary>' + esc(title) + '</summary><p>' + esc(note)
                     + ' Source: ' + link(path) + '</p><p class="hash">SHA-256 of displayed text: <code>' + sha(text)
                     + '</code></p><pre><code>' + esc(text) + '</code></pre></details>\n')
    inputs = ['summary_runtime.py', 'summarize.py', 'evaluate.py', 'run_summary_comparison.py',
              'serve_extractor.sh', 'serve_judge.sh', 'qa_authoring_prompt.txt', 'summary-systemprompt+.txt',
              'summary_comparison_manifest.json', 'summary_runtime_fingerprints.json', 'judge_model_fingerprints.json',
              'data/papers.json', 'data/manifest.json', 'kus.json', 'pre_qwen_summary_results.json',
              'summary_runs/qwen27b/summaries.json', 'summary_runs/qwen27b/results.json',
              'summary_runs/qwen27b/summary_prompt_snapshot.json',
              '../../src/project_alexandria/experiments/mcq.py', '../../src/project_alexandria/experiments/reproduce.py',
              'pipeline_report.py', 'pipeline_report.template.html']
    if (ROOT / 'summary_repair_v3.py').exists():
        inputs.append('summary_repair_v3.py')
    if (ROOT / 'finish_qwen9b.py').exists():
        inputs.append('finish_qwen9b.py')
    if (ROOT / 'run_ornith9b.py').exists():
        inputs.append('run_ornith9b.py')
    fingerprints = {path: sha((ROOT / path).read_bytes()) for path in inputs}
    # Individual QA files are part of the reproducibility record, not just aggregate results.
    for identifier in manifest['documents']:
        path = 'data/qa/' + identifier + '.json'
        fingerprints[path] = sha((ROOT / path).read_bytes())
    modified = [path for path in inputs if subprocess.run(
        ['git', 'diff', '--quiet', 'HEAD', '--', str((ROOT / path).resolve())], cwd=str(REPO)).returncode == 1]
    report_data = dict(as_of=AS_OF, code_reference_commit=commit, reference_note='Input SHA-256 hashes are authoritative; current working-tree differences are disclosed.',
                       modified_tracked_inputs=modified, measured_workload=work, throughput_assumptions=RATES,
                       estimates=estimates, sources=SOURCES, input_sha256=fingerprints,
                       prompts=[dict(id=i, title=t, text=p, sha256=sha(p), source=s, note=n)
                                for i, t, p, s, n in inventory])
    provenance = table(['Reproduction input', 'SHA-256 of exact file bytes'],
                       [[link(path), '<code>' + digest + '</code>'] for path, digest in fingerprints.items()
                        if not path.startswith('data/qa/')])
    values = {
        'AS_OF': AS_OF, 'COMMIT': esc(commit), 'CODE_URL': URL + '/tree/' + commit,
        'MODELS': table(['Generator (model repository)', 'Runtime', 'GPUs', 'Concurrent requests', 'Pinned revision'], model_rows),
        'COSTS': table(['Projected FP8 model', 'Workload', 'Central GPU-hours', 'Planning range'], cost_rows),
        'RATES': table(['Projected model', 'Prefill tokens/s/GPU', 'Decode tokens/s/GPU', 'Effective utilization'], rate_rows),
        'WALL': table(['Projected model', 'Workload', '1,024 GH200s', '4,096 GH200s'], wall_rows),
        'SCORES': table(['Student context', 'Correct / total', 'Accuracy', 'Document-bootstrap 95% CI', 'Invalid answers'], score_rows),
        'WORKLOAD': table(['Measured quantity (97 completed papers)', 'Value'], [
            ['Mean source words', f"{work['source_words_mean']:,.0f}"],
            ['Mean final student-facing narrative words', f"{work['narrative_words_mean']:,.0f}"],
            ['First-generation input / output tokens per paper', f"{work['initial_input_tokens']:,.1f} / {work['initial_output_tokens']:,.1f}"],
            ['All distinct saved-call input / output tokens per completed paper', f"{work['validated_input_tokens']:,.1f} / {work['validated_output_tokens']:,.1f}"],
            ['Calls per successful document journal', f"{work['successful_journal_calls_per_paper']:.2f}"],
            ['Distinct saved calls, including historical failures', str(work['distinct_saved_calls'])],
            ['Checkpointed active wall time / two-GPU hours proxy', f"{work['checkpointed_active_wall_hours']:.2f} h / {work['checkpointed_gpu_hours_proxy']:.2f} GPU-h"]]),
        'APPENDIX': appendix, 'PROVENANCE': provenance,
        'MODIFIED': esc(', '.join(modified) or 'None among tracked inputs'),
        'EXCLUDED': esc(', '.join(manifest['excluded_ids'])),
        'FIELDS': ', '.join('<code>' + esc(k) + '</code>' for k in literal(ROOT / 'summarize.py', 'KEYS')),
        'FINGERPRINTS': esc(json.dumps(load(ROOT / 'summary_runtime_fingerprints.json'), indent=2)),
        'JUDGE_FINGERPRINTS': esc(json.dumps(load(ROOT / 'judge_model_fingerprints.json'), indent=2)),
    }
    for name, url in SOURCES.items():
        values['SOURCE_' + name.upper()] = esc(url)
    text = (ROOT / 'pipeline_report.template.html').read_text(encoding='utf-8')
    text = re.sub(r'@@([A-Z0-9_]+)@@', lambda m: values[m.group(1)], text)
    if re.search(r'@@[A-Z0-9_]+@@', text):
        raise ValueError('Unresolved report placeholders')
    # A focused credential scan: no environment variables, remote credentials, or logs are read.
    if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})', text):
        raise ValueError('Credential-shaped material detected; report not written')
    return text, report_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'summary_pipeline.html')
    args = parser.parse_args()
    report, data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding='utf-8')
    data_path = args.output.with_suffix('.json')
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    args.output.with_suffix('.SHA256SUMS').write_text(''.join(
        sha(p.read_bytes()) + '  ' + p.name + '\n' for p in (args.output, data_path)), encoding='utf-8')
    print('Built', args.output.name, 'and its JSON provenance/checksums; no inference started.')


if __name__ == '__main__':
    main()
