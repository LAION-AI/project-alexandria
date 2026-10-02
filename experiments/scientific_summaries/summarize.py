"""Whole-paper Qwen summaries using the user's literal Schema-v4 system prompt."""
import argparse
import ast
import copy
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluate import ROOT, evidence_span, load
from extract import MODEL, REVISION
from project_alexandria.backends import OpenAICompatibleBackend
from project_alexandria.io import write_json_atomic
from summary_runtime import SummaryClient, generate_document, repair_saved_document, REPAIR_SYSTEM, ANCHOR_SYSTEM

KEYS = ('title', 'authors', 'field_subfield', 'type_of_paper', 'executive_summary',
        'research_context', 'research_question_and_hypothesis', 'methodological_details',
        'procedures_and_architectures', 'key_results', 'interpretation_and_theoretical_implications',
        'contradictions_and_limitations', 'claims', 'data_and_code_availability',
        'robustness_and_ablation_notes', 'ethical_considerations', 'key_figures_tables',
        'top_influential_citations', 'three_takeaways')
GROUNDED = KEYS[5:12] + KEYS[13:17] + ('three_takeaways',)


def system_prompt(path):
    raw = path.read_text(encoding='utf-8')
    if raw.lstrip().startswith('SYSTEM_PROMPT_SUMMARY'):
        tree = ast.parse(raw)
        if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
                or len(tree.body[0].targets) != 1
                or not isinstance(tree.body[0].targets[0], ast.Name)
                or tree.body[0].targets[0].id != 'SYSTEM_PROMPT_SUMMARY'):
            raise ValueError('Prompt wrapper must be one literal string assignment; code is never executed.')
        value = ast.literal_eval(tree.body[0].value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('System prompt must be nonempty text')
        return value
    if not raw.strip():
        raise ValueError('Empty system prompt')
    return raw


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key[:80])
        result[key] = value
    return result


def validate_summary(response, source):
    summary = json.loads(response, object_pairs_hook=unique_object)
    if not isinstance(summary, dict) or tuple(summary) != KEYS:
        raise ValueError('Root keys/order must exactly match the 19-field Schema-v4 template')
    summary = copy.deepcopy(summary)
    spans = []

    def quote(value, path):
        if not isinstance(value, str) or not value.strip() or len(value.split()) > 5:
            raise ValueError('Evidence must be a nonempty <=5-word quote: ' + path)
        start, end, exact = evidence_span(source, value)
        if len(exact.split()) > 5:
            raise ValueError('Aligned source quote exceeds five words: ' + path)
        spans.append({'path': path, 'quote': exact, 'start': start, 'end': end,
                      'typographic_alignment': exact != value, 'original_quote': value})
        return exact

    def sequence(value, path):
        if value == '':
            return ''
        if not isinstance(value, list) or not value:
            raise ValueError('Grounded field must be a nonempty list or empty string: ' + path)
        resolved = []
        for index, item in enumerate(value):
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError('Grounded entry must have one narrative key: ' + path)
            narrative, evidence = next(iter(item.items()))
            if not isinstance(narrative, str) or not narrative.strip():
                raise ValueError('Empty narrative statement: ' + path)
            resolved.append({narrative: quote(evidence, f'{path}[{index}]')})
        return resolved

    for field in KEYS[:5]:
        if not isinstance(summary[field], str):
            raise ValueError('Metadata and executive_summary must be strings: ' + field)
    for field in GROUNDED:
        summary[field] = sequence(summary[field], field)
    claims = summary['claims']
    if claims != '':
        if not isinstance(claims, list) or not claims:
            raise ValueError('claims must be a nonempty list or empty string')
        for index, claim in enumerate(claims):
            if (not isinstance(claim, dict) or tuple(claim) != (
                    'description', 'supporting_evidence', 'contradicting_evidence', 'implications')
                    or not isinstance(claim['description'], str)):
                raise ValueError('Invalid claim object')
            for field in ('supporting_evidence', 'contradicting_evidence', 'implications'):
                claim[field] = sequence(claim[field], f'claims[{index}].{field}')
    citations = summary['top_influential_citations']
    if citations != '':
        if not isinstance(citations, list) or not 1 <= len(citations) <= 5:
            raise ValueError('Citation list must contain up to five source-supported entries')
        for index, citation in enumerate(citations):
            if not isinstance(citation, dict) or len(citation) != 2 or 'quotes' not in citation:
                raise ValueError('Each citation needs one citation/justification pair and quotes')
            key = next(k for k in citation if k != 'quotes')
            if not isinstance(citation[key], str) or not citation[key].strip():
                raise ValueError('Citation justification is missing')
            # Preserve the exact source citation spelling as well as the <=5-word proof fragments.
            _, _, exact_key = evidence_span(source, key)
            if exact_key != key:
                value = citation.pop(key)
                citation = {exact_key: value, 'quotes': citation['quotes']}
                citations[index] = citation
            if not isinstance(citation['quotes'], list) or not citation['quotes']:
                raise ValueError('Citation needs at least one nonempty proof fragment')
            citation['quotes'] = [quote(v, f'top_influential_citations[{index}].quotes[{j}]')
                                  for j, v in enumerate(citation['quotes'])]
    narrative = narrative_context(summary)
    if not summary['executive_summary'].strip() or not narrative.strip():
        raise ValueError('A usable executive summary is required')
    return summary, narrative, spans


def narrative_context(summary):
    """Match the original substantive-field condition, leaving proof quotes as provenance."""
    def sentences(sequence):
        return ' '.join(next(iter(item)) for item in sequence) if isinstance(sequence, list) else ''
    sections = []
    for field in KEYS[4:]:
        if field == 'top_influential_citations':
            continue  # Bibliographic metadata/ranking is not the paper's distilled knowledge.
        value = summary[field]
        if field == 'claims':
            value = '\n'.join(claim['description'] + ' ' + ' '.join(
                sentences(claim[sub]) for sub in ('supporting_evidence', 'contradicting_evidence', 'implications'))
                for claim in value) if isinstance(value, list) else ''
        elif field != 'executive_summary':
            value = sentences(value)
        if value:
            sections.append(field.replace('_', ' ').title() + '\n' + value)
    return '\n\n'.join(sections)


def grounding_errors(response, source):
    """List all quote/citation failures together, without inventing replacements."""
    try:
        summary = json.loads(response, object_pairs_hook=unique_object)
    except ValueError as error:
        return [str(error)]
    errors = []

    def check(value, path, citation=False):
        try:
            if not isinstance(value, str) or not value.strip():
                raise ValueError('nonempty source text required')
            if not citation and len(value.split()) > 5:
                raise ValueError('quote exceeds five words')
            _, _, exact = evidence_span(source, value)
            if not citation and len(exact.split()) > 5:
                raise ValueError('aligned quote exceeds five words')
        except ValueError as error:
            errors.append(path + ': ' + str(error))

    def sequence(value, path):
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict) and len(item) == 1:
                    check(next(iter(item.values())), f'{path}[{index}]')

    if not isinstance(summary, dict):
        return ['Root must be a JSON object']
    for field in GROUNDED:
        sequence(summary.get(field), field)
    claims = summary.get('claims')
    if isinstance(claims, list):
        for index, claim in enumerate(claims):
            if isinstance(claim, dict):
                for field in ('supporting_evidence', 'contradicting_evidence', 'implications'):
                    sequence(claim.get(field), f'claims[{index}].{field}')
    citations = summary.get('top_influential_citations')
    if isinstance(citations, list):
        for index, citation in enumerate(citations):
            if isinstance(citation, dict):
                for key in citation:
                    if key != 'quotes':
                        check(key, f'top_influential_citations[{index}].citation', citation=True)
                if isinstance(citation.get('quotes'), list):
                    for j, value in enumerate(citation['quotes']):
                        check(value, f'top_influential_citations[{index}].quotes[{j}]')
    return errors


def generate_one(paper, backend, prompt, max_tokens, attempts, previous_failure=None):
    """Legacy failed full-regeneration protocol, retained for diagnosis; main does not use it."""
    user = 'Summarize only the following complete paper using the required Schema-v4 JSON.\n\nBEGIN_PAPER\n' + paper['fulltext'] + '\nEND_PAPER'
    payload = json.dumps({'model': 'qwen38', 'messages': [
        {'role': 'system', 'content': prompt}, {'role': 'user', 'content': user}],
        'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
    request = urllib.request.Request(backend.base_url.rsplit('/v1', 1)[0] + '/tokenize',
        data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        tokens = json.load(response)['count']
    if tokens + max_tokens > 32768:
        raise ValueError('Whole-paper input plus output budget exceeds 32,768 tokens; no truncation allowed.')
    records = []
    error = ''
    prior_response = ''
    if previous_failure and previous_failure.get('attempts'):
        prior_response = previous_failure['attempts'][-1]['response']
        error = previous_failure.get('validation_error', 'Prior output failed validation')
    started = time.monotonic()
    for attempt in range(attempts):
        # The system prompt never changes; repair instructions are explicit user-message additions.
        request_prompt = user
        output_budget = max_tokens
        if error:
            issues = grounding_errors(prior_response, paper['fulltext'])
            request_prompt += '\n\nRepair this prior JSON, preserving its supported narrative and exact schema:\n'
            request_prompt += prior_response + '\n\nVALIDATION ERRORS:\n' + error
            request_prompt += '\n' + '\n'.join(issues)
            request_prompt += ('\nReturn the complete corrected JSON only. Fix ALL listed problems. '
                'Copy quote values literally from BEGIN_PAPER, including its spelling, punctuation, '
                'LaTeX and whitespace; do not paraphrase them or substitute Unicode/math notation. '
                'Use only nonempty <=5-word source quotes that support the narrative statement. '
                'Citation keys must also be copied from the source, not reconstructed. '
                'If a statement or citation cannot be supported, remove that entry rather than invent evidence. '
                'An unreported field must be an empty string. Do not add speculation to meet word targets.')
        # Recheck the repair message, which is longer than the first user message.
        if error:
            repair_payload = json.dumps({'model': 'qwen38', 'messages': [
                {'role': 'system', 'content': prompt}, {'role': 'user', 'content': request_prompt}],
                'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}).encode()
            with urllib.request.urlopen(urllib.request.Request(
                    backend.base_url.rsplit('/v1', 1)[0] + '/tokenize', data=repair_payload,
                    headers={'Content-Type': 'application/json'}), timeout=30) as response:
                repair_tokens = json.load(response)['count']
            output_budget = min(max_tokens, 32768 - repair_tokens - 128)
            if output_budget < 4096:
                raise ValueError('Repair message leaves insufficient output budget; no source truncation allowed.')
        raw = backend.generate(prompt, request_prompt, max_tokens=output_budget)
        record = {'attempt': attempt + 1, 'response': raw,
                  'max_tokens': output_budget, 'repair_protocol': 'all_grounding_errors_with_prior_json_v1',
                  'user_prompt_sha256': hashlib.sha256(request_prompt.encode()).hexdigest()}
        if error:
            record['repair_errors'] = [error] + issues
            record['prior_response_sha256'] = hashlib.sha256(prior_response.encode()).hexdigest()
        records.append(record)
        try:
            summary, narrative, spans = validate_summary(raw, paper['fulltext'])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            prior_response = raw
            record['validation_error'] = error
            continue
        return {'document_id': paper['document_id'], 'fulltext_sha256': paper['fulltext_sha256'],
                'summary': summary, 'judge_context': narrative, 'evidence_spans': spans,
                'attempts': records, 'input_tokens': tokens,
                'elapsed_seconds': time.monotonic() - started}
    return {'document_id': paper['document_id'], 'fulltext_sha256': paper['fulltext_sha256'],
            'failed': True, 'attempts': records, 'validation_error': error,
            'elapsed_seconds': time.monotonic() - started}


def recover_failed_checkpoint(config, path):
    """Explicit version-changing recovery; never relabel already scored/valid output."""
    prior = load(path)
    if prior.get('documents') or not prior.get('failures'):
        raise ValueError('Recovery requires a failed-only checkpoint with no valid documents')
    old = prior['config']
    identity = {k: v for k, v in config.items() if k != 'repair_implementation_sha256'}
    old_identity = {k: v for k, v in old.items()
                    if k not in ('repair_implementation_sha256', 'recovery')}
    if old_identity != identity or config['repair_protocol'] != 'field_local_strict_grounding_v3':
        raise ValueError('Recovery source/model/prompt/generation identity does not match')
    updated = dict(config, recovery={
        'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'prior_repair_implementation_sha256': old.get('repair_implementation_sha256'),
        'prior_recovery': old.get('recovery'),
        'historical_elapsed_seconds_included': prior.get('elapsed_seconds', 0.0)})
    return dict(config=updated, documents=[], failures=prior['failures'],
                elapsed_seconds=prior.get('elapsed_seconds', 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8010/v1')
    parser.add_argument('--prompt', type=Path, default=ROOT / 'summary-systemprompt+.txt')
    parser.add_argument('--skip-id', action='append', default=[])
    parser.add_argument('--max-new-documents', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=16000)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'qwen_summaries.json')
    parser.add_argument('--runtime', choices=('vllm', 'llama.cpp'), default='vllm')
    parser.add_argument('--alias', default='qwen38')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--weights-sha256', default='')
    parser.add_argument('--allocated-gpus', type=int, default=2)
    parser.add_argument('--repair-protocol', choices=('field_local_strict_grounding_v2',
                        'field_local_strict_grounding_v3'), default='field_local_strict_grounding_v2')
    parser.add_argument('--recover-failed-cache', type=Path,
                        help='Explicitly reuse failed-only V3 drafts across an implementation change')
    args = parser.parse_args()
    if min(args.concurrency, args.max_tokens, args.attempts) < 1 or args.max_new_documents < 0:
        parser.error('Invalid generation settings')
    prompt = system_prompt(args.prompt)
    config = {'model': args.model, 'revision': args.revision,
              'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
              'prompt_file_sha256': hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
              'temperature': 0.2, 'top_p': 0.95, 'thinking': False, 'max_tokens': args.max_tokens,
              'concurrency': args.concurrency, 'attempts': args.attempts, 'context_limit': 32768,
              'runtime': args.runtime, 'weights_sha256': args.weights_sha256,
              'allocated_gpus': args.allocated_gpus,
              'repair_protocol': args.repair_protocol,
              'repair_system_prompt_sha256': hashlib.sha256(REPAIR_SYSTEM.encode()).hexdigest(),
              'student_input': 'all_substantive_narrative_fields_without_grounding_quotes_or_citation_rankings'}
    if args.repair_protocol == 'field_local_strict_grounding_v3':
        config['repair_implementation_sha256'] = hashlib.sha256(
            (ROOT / 'summary_repair_v3.py').read_bytes()).hexdigest()
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    recovered = recover_failed_checkpoint(config, args.recover_failed_cache) if args.recover_failed_cache else None
    if recovered:
        config = recovered['config']
    output = load(path) if path.exists() else (recovered or {'config': config, 'documents': [], 'failures': [], 'elapsed_seconds': 0.0})
    if output['config'] != config:
        raise ValueError('Summary checkpoint prompt/model/generation settings have changed')
    snapshot = {
        'file_name': args.prompt.name, 'file_sha256': config['prompt_file_sha256'],
        'effective_prompt_sha256': config['system_prompt_sha256'], 'effective_system_prompt': prompt,
        'repair_system_prompt': REPAIR_SYSTEM, 'anchor_selection_system_prompt': ANCHOR_SYSTEM}
    if args.repair_protocol == 'field_local_strict_grounding_v3':
        from summary_repair_v3 import FIELD_USER_TEMPLATE, ANCHOR_USER_TEMPLATE, field_example
        snapshot.update(repair_protocol=args.repair_protocol, field_user_template=FIELD_USER_TEMPLATE,
                        anchor_user_template=ANCHOR_USER_TEMPLATE,
                        field_shape_examples={k: field_example(k) for k in KEYS},
                        repair_implementation_sha256=hashlib.sha256(
                            (ROOT / 'summary_repair_v3.py').read_bytes()).hexdigest())
    write_json_atomic(str(path.parent / 'summary_prompt_snapshot.json'), snapshot)
    papers = load(ROOT / 'data/papers.json')
    by_id = {p['document_id']: p for p in papers}
    for document in output['documents'] + output['failures']:
        if (document['document_id'] not in by_id or
                document.get('fulltext_sha256', by_id[document['document_id']]['fulltext_sha256']) !=
                by_id[document['document_id']]['fulltext_sha256']):
            raise ValueError('Source changed after summary generation')
    done = {d['document_id'] for d in output['documents']}
    pending = [p for p in papers if p['document_id'] not in done | set(args.skip_id)]
    if args.max_new_documents:
        pending = pending[:args.max_new_documents]
    backend = SummaryClient(args.base_url, args.alias, args.runtime)
    directory = path.parent / 'documents'
    directory.mkdir(exist_ok=True)
    started = time.monotonic()
    previous_failures = {d['document_id']: d for d in output['failures']}
    def process(paper):
        if args.repair_protocol == 'field_local_strict_grounding_v3':
            from summary_repair_v3 import generate, repair_document
            previous = previous_failures.get(paper['document_id'])
            if previous and previous.get('attempts'):
                return repair_document(paper, backend, previous)
            return generate(paper, backend, prompt, args.max_tokens)
        previous = previous_failures.get(paper['document_id'])
        if previous and len(previous.get('attempts', [])) <= args.attempts:
            repaired = repair_saved_document(paper, backend, previous, args.attempts)
            if repaired is not None:
                return repaired
        document = generate_document(paper, backend, prompt, args.max_tokens, args.attempts)
        # First-pass smoke failures also receive the same bounded second-pass protocol.
        if document.get('failed'):
            repaired = repair_saved_document(paper, backend, document, args.attempts)
            if repaired is not None:
                return repaired
        return document
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process, p): p for p in pending}
        for future in as_completed(futures):
            paper = futures[future]
            try:
                document = future.result()
            except Exception as error:
                document = {'document_id': paper['document_id'], 'failed': True, 'error': str(error)}
            if document.get('failed'):
                output['failures'].append(document)
                print('SUMMARY_FAILED', paper['document_id'], flush=True)
            else:
                output['documents'].append(document)
                write_json_atomic(str(directory / (paper['document_id'] + '.json')), document)
                print('SUMMARY_COMPLETE', len(output['documents']), '/', len(papers) - len(args.skip_id), paper['document_id'], flush=True)
            output['elapsed_seconds'] += time.monotonic() - started
            started = time.monotonic()
            write_json_atomic(str(path), output)
    if any(p['document_id'] not in {d['document_id'] for d in output['documents']} for p in pending):
        raise SystemExit('Some summaries failed; all failures are saved. No missing summary will be silently scored.')


if __name__ == '__main__':
    main()
