"""Grounded summary inference: JSON transport, field-local repairs, and raw journals."""
import copy
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
import urllib.error
import urllib.request

REPAIR_SYSTEM = (
    'You repair evidence-grounded scientific JSON fields using only the supplied source. '
    'Return one JSON object mapping requested top-level field names to their corrected values. '
    'Do not return other fields, markdown, commentary, or the complete summary. '
    'Preserve supported narrative statements. Each grounded entry has exactly one narrative '
    'key and a nonempty literal source quote of at most five words as its value. '
    'Copy source spelling, mathematical symbols, punctuation and PDF typography literally. '
    'Do not invent proof fragments. Remove unsupported entries; if a field is not reported, '
    'return the empty string for the entire field, not a list containing empty quotes. '
    'Citation entries have exactly a literal citation key with a justification value and a '
    'quotes key with nonempty source quotes. Never retain template placeholders. '
    'If no bibliography is supplied, return an empty string for top_influential_citations. '
    'Claims have description, supporting_evidence, contradicting_evidence, implications, '
    'in that order; the latter three are grounded lists or empty strings. '
    'Never change scientific numbers to make a quote match. No outside knowledge is allowed.'
)

ANCHOR_SYSTEM = (
    'You verify scientific statements against the supplied source. For each requested task, '
    'choose one numbered candidate proof fragment ONLY if it supports the statement and its '
    'numbers/negations. Return a JSON object mapping every task ID to the integer candidate '
    'index, or to "drop" when no candidate adequately supports the statement. Do not invent '
    'quotes, rewrite statements, or use outside knowledge. Weak topical overlap is not proof. '
    'Dropping unsupported evidence is preferable to a false citation. Return JSON only.'
)


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_json(raw):
    """Only remove an enclosing code fence, never prose or a partial JSON object."""
    from summarize import unique_object
    text = raw.strip()
    transformations = []
    if text.startswith(('```json\n', '```\n')) and text.endswith('\n```'):
        text = text.split('\n', 1)[1].rsplit('\n```', 1)[0]
        transformations.append('removed_enclosing_markdown_fence')
    return json.loads(text, object_pairs_hook=unique_object), transformations


def invalid_fields(summary, source):
    """Validate each field independently so one error cannot mask all others."""
    from summarize import KEYS, validate_summary
    if not isinstance(summary, dict) or set(summary) != set(KEYS):
        raise ValueError('Summary must have exactly the 19 Schema-v4 fields')
    errors = {}
    for field in KEYS:
        trial = dict.fromkeys(KEYS, '')
        trial['executive_summary'] = 'Validation placeholder, never saved or judged.'
        trial[field] = copy.deepcopy(summary[field])
        try:
            validate_summary(json.dumps(trial, ensure_ascii=False), source)
        except (ValueError, TypeError, KeyError) as error:
            errors[field] = str(error)
    return errors


def source_candidates(source, narrative, bad_quote, limit=16):
    """Lexical retrieval proposes real spans; it never declares an invalid quote matched."""
    stop = set('the a an and or of to in is are was were for with this that by from as be it its on at not'.split())
    def terms(value):
        return set(re.findall(r'\w+', unicodedata.normalize('NFKC', value).casefold())) - stop
    query = terms(narrative) | terms(bad_quote if isinstance(bad_quote, str) else '')
    matches = list(re.finditer(r'\S+', source))
    normalized = [terms(m.group()) for m in matches]
    counts = Counter(term for value in normalized for term in value)
    ranked = []
    for index in range(len(matches)):
        for size in (3, 4, 5):
            window = matches[index:index + size]
            if len(window) != size:
                continue
            hits = set().union(*normalized[index:index + size]) & query
            if not hits:
                continue
            score = sum(math.log(1 + len(matches) / counts[term]) for term in hits)
            ranked.append((score, -size, window[0].start(), window[-1].end()))
    ranked.sort(reverse=True)
    chosen = []
    seen = set()
    for _, _, start, end in ranked:
        quote = source[start:end]
        if quote in seen:
            continue
        seen.add(quote)
        chosen.append({'quote': quote, 'start': start, 'end': end})
        if len(chosen) >= limit:
            break
    return chosen


def quote_tasks(summary, source):
    from summarize import GROUNDED, evidence_span
    tasks = []
    def sequence(value, path):
        if not isinstance(value, list):
            return
        for index, entry in enumerate(value):
            if not isinstance(entry, dict) or len(entry) != 1:
                continue
            narrative, quote = next(iter(entry.items()))
            try:
                if not isinstance(quote, str) or not quote.strip() or len(quote.split()) > 5:
                    raise ValueError('invalid quote')
                _, _, exact = evidence_span(source, quote)
                if len(exact.split()) > 5:
                    raise ValueError('aligned quote too long')
            except ValueError:
                tasks.append(dict(id='q' + str(len(tasks)), path=path + [index], narrative=narrative,
                    invalid_quote=quote, candidates=source_candidates(source, narrative, quote)))
    for field in GROUNDED:
        sequence(summary[field], [field])
    if isinstance(summary.get('claims'), list):
        for index, claim in enumerate(summary['claims']):
            if isinstance(claim, dict):
                for field in ('supporting_evidence', 'contradicting_evidence', 'implications'):
                    sequence(claim.get(field), ['claims', index, field])
    return tasks


def restore_draft(failure):
    """Rebuild saved field-level edits; unvalidated edits remain unvalidated."""
    from summarize import KEYS
    if failure.get('draft_summary'):
        return copy.deepcopy(failure['draft_summary'])
    state = None
    for record in failure.get('attempts', []):
        try:
            value, _ = parse_json(record['response'])
        except (ValueError, KeyError):
            continue
        if record.get('phase') == 'generation' and isinstance(value, dict) and set(value) == set(KEYS):
            state = {field: value[field] for field in KEYS}
        elif (state is not None and record.get('phase') == 'field_repair'
                and isinstance(value, dict) and set(value) == set(record['requested_fields'])):
            state.update(value)
    return state


def repair_saved_document(paper, client, failure, max_attempts=6):
    """Second bounded pass selects exact spans instead of regenerating entire fields."""
    from summarize import validate_summary
    source = paper['fulltext']
    state = restore_draft(failure)
    if state is None:
        return None
    journal = copy.deepcopy(failure['attempts'])
    started = time.monotonic()
    seed = int(digest(paper['document_id'])[:8], 16) % 2147483000 + 100
    removals, anchors = [], []
    for attempt in range(max_attempts):
        errors = invalid_fields(state, source)
        if not errors:
            summary, narrative, spans = validate_summary(json.dumps(state, ensure_ascii=False), source)
            return {'document_id': paper['document_id'], 'fulltext_sha256': paper['fulltext_sha256'],
                'summary': summary, 'judge_context': narrative, 'evidence_spans': spans,
                'attempts': journal, 'input_tokens': journal[0]['input_tokens_preflight'],
                'elapsed_seconds': failure.get('elapsed_seconds', 0) + time.monotonic() - started,
                'removed_unsupported_entries': removals, 'selected_source_anchors': anchors,
                'repair_protocol': 'field_local_strict_grounding_v2_with_anchored_retry'}
        tasks = quote_tasks(state, source)[:8]  # Bound repair inputs; never truncate the paper.
        if tasks:
            public_tasks = [{k: value for k, value in task.items() if k != 'path'} for task in tasks]
            user = 'BEGIN_PAPER\n' + source + '\nEND_PAPER\n\nTASKS\n' + json.dumps(public_tasks, ensure_ascii=False)
            record = client.generate(ANCHOR_SYSTEM, user, 2048, seed + attempt)
            record.update(phase='source_anchor_selection', attempt=len(journal) + 1, tasks=tasks)
            journal.append(record)
            try:
                choices, changes = parse_json(record['response'])
                record['normalizations'] = changes
                if record['finish_reason'] == 'length' or not isinstance(choices, dict) or set(choices) != {task['id'] for task in tasks}:
                    raise ValueError('Anchor response must cover exactly all task IDs')
                # Validate every choice before applying any operation.
                for task in tasks:
                    choice = choices[task['id']]
                    if choice != 'drop' and (type(choice) is not int or not 0 <= choice < len(task['candidates'])):
                        raise ValueError('Anchor choice is not a provided source span')
                for task in reversed(tasks):
                    sequence = state
                    for component in task['path'][:-1]:
                        sequence = sequence[component]
                    index = task['path'][-1]
                    choice = choices[task['id']]
                    if choice == 'drop':
                        removals.append({'path': task['path'], 'entry': sequence[index],
                                         'reason': 'model_found_no_supporting_candidate'})
                        del sequence[index]
                    else:
                        anchor = task['candidates'][choice]
                        sequence[index] = {task['narrative']: anchor['quote']}
                        anchors.append(dict(path=task['path'], **anchor))
                # Empty grounded sequences use the prompt's empty-string representation.
                for task in tasks:
                    owner = state
                    for component in task['path'][:-2]:
                        owner = owner[component]
                    key = task['path'][-2]
                    if owner[key] == []:
                        owner[key] = ''
            except (ValueError, KeyError, TypeError) as error:
                record['validation_error'] = str(error)
                continue
        else:
            # Remaining non-quote/schema/citation errors still get a bounded field-only patch.
            user = 'BEGIN_PAPER\n' + source + '\nEND_PAPER\nFIELDS_TO_REPAIR\n' + json.dumps(
                {field: state[field] for field in errors}, ensure_ascii=False) + '\nERRORS\n' + json.dumps(errors)
            record = client.generate(REPAIR_SYSTEM, user, 4096, seed + attempt)
            record.update(phase='field_repair', attempt=len(journal) + 1, requested_fields=list(errors))
            journal.append(record)
            try:
                patch, changes = parse_json(record['response'])
                record['normalizations'] = changes
                if record['finish_reason'] == 'length' or not isinstance(patch, dict) or set(patch) != set(errors):
                    raise ValueError('Repair must cover exactly the requested fields')
                state.update(patch)
            except (ValueError, KeyError, TypeError) as error:
                record['validation_error'] = str(error)
    # The final edit also needs validation, without requiring another generation.
    errors = invalid_fields(state, source)
    if not errors:
        summary, narrative, spans = validate_summary(json.dumps(state, ensure_ascii=False), source)
        return {'document_id': paper['document_id'], 'fulltext_sha256': paper['fulltext_sha256'],
            'summary': summary, 'judge_context': narrative, 'evidence_spans': spans, 'attempts': journal,
            'input_tokens': journal[0]['input_tokens_preflight'],
            'elapsed_seconds': failure.get('elapsed_seconds', 0) + time.monotonic() - started,
            'removed_unsupported_entries': removals, 'selected_source_anchors': anchors,
            'repair_protocol': 'field_local_strict_grounding_v2_with_anchored_retry'}
    return dict(failure, attempts=journal, draft_summary=state, validation_errors=errors,
                validation_error=str(errors), elapsed_seconds=failure.get('elapsed_seconds', 0) + time.monotonic() - started)


class SummaryClient:
    def __init__(self, base_url, alias, runtime, context_limit=32768):
        self.base_url = base_url.rstrip('/').removesuffix('/v1') if hasattr(str, 'removesuffix') else base_url.rstrip('/').rsplit('/v1', 1)[0]
        self.alias = alias
        self.runtime = runtime
        self.context_limit = context_limit

    def post(self, endpoint, payload, timeout=1800):
        request = urllib.request.Request(self.base_url + endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                # Never print headers or credentials. Non-transient errors must not repeat.
                if error.code < 500 or attempt == 2:
                    raise RuntimeError('Summary endpoint HTTP ' + str(error.code)) from error
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)

    def token_count(self, messages):
        if self.runtime == 'llama.cpp':
            rendered = self.post('/apply-template', {'messages': messages,
                'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})
            return len(self.post('/tokenize', {'content': rendered['prompt'], 'add_special': False})['tokens'])
        return self.post('/tokenize', {'model': self.alias, 'messages': messages,
            'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}})['count']

    def generate(self, system, user, max_tokens, seed):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        tokens = self.token_count(messages)
        if tokens + max_tokens > self.context_limit:
            raise ValueError('Input plus output budget exceeds context; no source is truncated')
        payload = {'model': self.alias, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': 0.2, 'top_p': 0.95, 'seed': seed,
            'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_object'}}
        if self.runtime == 'llama.cpp':
            payload.update(top_k=0, min_p=0.0, repeat_penalty=1.0)
        started = time.monotonic()
        body = self.post('/v1/chat/completions', payload)
        choice = body['choices'][0]
        raw = choice['message'].get('content')
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError('Model returned no final summary content')
        return {'response': raw, 'finish_reason': choice.get('finish_reason'),
            'usage': body.get('usage', {}), 'input_tokens_preflight': tokens,
            'elapsed_seconds': time.monotonic() - started, 'max_tokens': max_tokens,
            'seed': seed, 'system_prompt_sha256': digest(system), 'user_prompt_sha256': digest(user)}


def generate_document(paper, client, prompt, max_tokens, attempts):
    from summarize import KEYS, validate_summary, grounding_errors
    source = paper['fulltext']
    user = ('Apply the system prompt to the supplied dataset paper text only. The dataset may '
        'omit the bibliography; if it does, top_influential_citations must be "". '
        'For unreported data/code or ethics, use "" for the entire field. '
        'Do not reproduce template placeholders. Copy proof quotes literally, including symbols.\n\n'
        'BEGIN_PAPER\n' + source + '\nEND_PAPER')
    journal = []
    state = None
    errors = {}
    started = time.monotonic()
    seed = int(digest(paper['document_id'])[:8], 16) % 2147483000
    for attempt in range(attempts):
        is_repair = state is not None
        if is_repair:
            fields = {field: state[field] for field in errors}
            request = ('BEGIN_PAPER\n' + source + '\nEND_PAPER\n\nFIELDS_TO_REPAIR\n'
                + json.dumps(fields, ensure_ascii=False) + '\n\nERRORS\n'
                + json.dumps(errors, ensure_ascii=False) + '\n\nAll detected quote errors:\n'
                + '\n'.join(grounding_errors(json.dumps(state, ensure_ascii=False), source)))
            system = REPAIR_SYSTEM
            budget = min(max_tokens, 4096)
        else:
            request, system, budget = user, prompt, max_tokens
        record = client.generate(system, request, budget, seed + attempt)
        record.update(attempt=attempt + 1, phase='field_repair' if is_repair else 'generation',
                      requested_fields=list(errors) if is_repair else list(KEYS))
        journal.append(record)
        try:
            if record['finish_reason'] == 'length':
                raise ValueError('Output token budget exhausted; partial output cannot be accepted')
            obj, changes = parse_json(record['response'])
            record['normalizations'] = changes
            if is_repair:
                if not isinstance(obj, dict) or set(obj) != set(errors):
                    raise ValueError('Repair must return exactly the requested top-level fields')
                for field, value in obj.items():
                    state[field] = value
            else:
                if not isinstance(obj, dict) or set(obj) != set(KEYS):
                    raise ValueError('Generation lacks exact Schema-v4 keys')
                if tuple(obj) != KEYS:
                    record['normalizations'].append('restored_template_field_order')
                state = {field: obj[field] for field in KEYS}
            errors = invalid_fields(state, source)
            if errors:
                record['validation_errors'] = errors
                print('SUMMARY_REPAIR', paper['document_id'], attempt + 1,
                      ','.join(errors), flush=True)
                continue
            summary, narrative, spans = validate_summary(json.dumps(state, ensure_ascii=False), source)
        except (ValueError, TypeError, KeyError) as error:
            record['validation_error'] = str(error)
            print('SUMMARY_VALIDATION_RETRY', paper['document_id'], attempt + 1,
                  str(error)[:180], flush=True)
            if state is None:
                user += '\nReturn only a complete JSON object with all 19 fields in template order.'
            continue
        return {'document_id': paper['document_id'], 'fulltext_sha256': paper['fulltext_sha256'],
            'summary': summary, 'judge_context': narrative, 'evidence_spans': spans,
            'attempts': journal, 'input_tokens': journal[0]['input_tokens_preflight'],
            'elapsed_seconds': time.monotonic() - started,
            'repair_protocol': 'field_local_strict_grounding_v2'}
    return {'document_id': paper['document_id'], 'fulltext_sha256': paper['fulltext_sha256'],
        'failed': True, 'attempts': journal, 'validation_errors': errors,
        'draft_summary': state,
        'validation_error': journal[-1].get('validation_error', str(errors)),
        'elapsed_seconds': time.monotonic() - started}
