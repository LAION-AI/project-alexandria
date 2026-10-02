"""Versioned 9B repairs: lossless shape alignment, single-field patches, all quotes.

The initial system prompt, source, and final validator are unchanged. V2 remains
available for reproducing the completed 27B experiment. No fuzzy evidence acceptance.
"""
import copy
import json
import math
import time

from summary_runtime import (ANCHOR_SYSTEM, REPAIR_SYSTEM, digest, generate_document,
                             invalid_fields, parse_json, quote_tasks)

PROTOCOL = 'field_local_strict_grounding_v3'
FIELD_USER_TEMPLATE = (
    'BEGIN_PAPER\n{source}\nEND_PAPER\n\n'
    'Repair ONLY the field named below. Preserve all supported narrative statements; '
    'do not replace a detailed field with a one-sentence stub. Remove unsupported '
    'entries. Unreported/inapplicable fields must be the empty string. '
    'The example demonstrates JSON shape only, not facts to copy. '
    'Do not use objects with narrative/source/evidence property names: a grounded '
    'entry has the actual statement as its single key and the quote as its value.\n'
    'FIELD\n{field}\nCURRENT_VALUE\n{value}\nERROR\n{error}\n'
    'EXACT_OUTPUT_SHAPE_EXAMPLE\n{example}\n'
    'Return JSON with exactly this one top-level field. Copy only real source quotes '
    'of at most five words. For citation rankings, use only actual bibliography entries '
    'as citation keys, never a placeholder or a scientific statement. If no bibliography '
    'is present, return {{"top_influential_citations": ""}}; the empty citation example '
    'illustrates this absent-bibliography case.'
)
ANCHOR_USER_TEMPLATE = 'BEGIN_PAPER\n{source}\nEND_PAPER\n\nTASKS\n{tasks}'


def normalize_shapes(state):
    """Rewrap unambiguous aliases without rewriting/dropping any narrative or quote."""
    from summarize import GROUNDED
    result, changes = copy.deepcopy(state), []

    def sequence(value, path):
        if value == []:
            changes.append({'path': path, 'operation': 'empty_list_to_empty_string'})
            return ''
        if not isinstance(value, list):
            return value
        converted = []
        for index, entry in enumerate(value):
            if isinstance(entry, dict) and len(entry) == 2:
                pairs = [(n, q) for n in ('narrative', 'statement', 'text')
                         for q in ('source', 'evidence', 'quote') if set(entry) == {n, q}]
                if len(pairs) == 1:
                    n, q = pairs[0]
                    if isinstance(entry[n], str) and isinstance(entry[q], str):
                        changes.append({'path': path + [index], 'operation': 'unambiguous_entry_rewrap',
                                        'narrative_key': n, 'quote_key': q})
                        entry = {entry[n]: entry[q]}
            converted.append(entry)
        return converted

    for field in GROUNDED:
        result[field] = sequence(result[field], [field])
    claim_keys = ('description', 'supporting_evidence', 'contradicting_evidence', 'implications')
    if result.get('claims') == []:
        result['claims'] = ''
        changes.append({'path': ['claims'], 'operation': 'empty_list_to_empty_string'})
    elif isinstance(result.get('claims'), list):
        for index, claim in enumerate(result['claims']):
            if not isinstance(claim, dict):
                continue
            if set(claim) == set(claim_keys) and tuple(claim) != claim_keys:
                claim = {k: claim[k] for k in claim_keys}
                result['claims'][index] = claim
                changes.append({'path': ['claims', index], 'operation': 'claim_key_order'})
            for field in claim_keys[1:]:
                if field in claim:
                    claim[field] = sequence(claim[field], ['claims', index, field])
    citations = result.get('top_influential_citations')
    if citations == []:
        result['top_influential_citations'] = ''
        changes.append({'path': ['top_influential_citations'], 'operation': 'empty_list_to_empty_string'})
    elif isinstance(citations, list):
        for index, item in enumerate(citations):
            if (isinstance(item, dict) and set(item) == {'citation', 'justification', 'quotes'}
                    and isinstance(item['citation'], str) and item['citation'] != 'quotes'
                    and isinstance(item['justification'], str)):
                citations[index] = {item['citation']: item['justification'], 'quotes': item['quotes']}
                changes.append({'path': ['top_influential_citations', index],
                                'operation': 'unambiguous_citation_rewrap'})
    return result, changes


def sequence_shape(value):
    return value == '' or (isinstance(value, list) and bool(value) and all(
        isinstance(e, dict) and len(e) == 1 and isinstance(next(iter(e)), str)
        and bool(next(iter(e)).strip()) and isinstance(next(iter(e.values())), str) for e in value))


def structural_fields(state):
    """Inspect all entries so an early bad quote cannot hide later malformed entries."""
    from summarize import GROUNDED, KEYS
    fields = {k for k in KEYS[:5] if not isinstance(state[k], str)}
    if isinstance(state['executive_summary'], str) and not state['executive_summary'].strip():
        fields.add('executive_summary')
    fields.update(k for k in GROUNDED if not sequence_shape(state[k]))
    claims = state['claims']
    keys = ('description', 'supporting_evidence', 'contradicting_evidence', 'implications')
    if claims != '' and (not isinstance(claims, list) or not claims or any(
            not isinstance(c, dict) or tuple(c) != keys or not isinstance(c['description'], str)
            or any(not sequence_shape(c[k]) for k in keys[1:]) for c in claims)):
        fields.add('claims')
    return fields


def field_example(field):
    from summarize import GROUNDED
    grounded = [{'A supported statement from the paper.': 'literal source quote'}]
    if field in GROUNDED:
        value = grounded
    elif field == 'claims':
        value = [{'description': 'A supported claim.', 'supporting_evidence': grounded,
                  'contradicting_evidence': '', 'implications': ''}]
    elif field == 'top_influential_citations':
        value = ''
    else:
        value = 'Source-supported text.'
    return {field: value}


def _record(client, journal, system, user, budget, seed, **metadata):
    record = client.generate(system, user, budget, seed)
    record.update(attempt=len(journal) + 1, **metadata)
    journal.append(record)
    return record


def _schema_stage(state, source, client, journal, seed, normalizations):
    """Two attempts per malformed field; literal-quote-only errors use anchors later."""
    from summarize import KEYS
    for field in KEYS:
        errors = invalid_fields(state, source)
        if field not in structural_fields(state) and not (
                field == 'top_influential_citations' and field in errors):
            continue
        for _ in range(2):
            errors = invalid_fields(state, source)
            user = FIELD_USER_TEMPLATE.format(source=source, field=field,
                value=json.dumps(state[field], ensure_ascii=False), error=errors.get(field, 'Invalid nested shape'),
                example=json.dumps(field_example(field), ensure_ascii=False))
            record = _record(client, journal, REPAIR_SYSTEM, user, 4096, seed + len(journal),
                             phase='field_repair', requested_fields=[field], repair_protocol=PROTOCOL)
            try:
                patch, changes = parse_json(record['response'])
                record['normalizations'] = changes
                if record.get('finish_reason') == 'length' or not isinstance(patch, dict) or set(patch) != {field}:
                    raise ValueError('Single-field repair must be complete and cover exactly its requested field')
                proposed = copy.deepcopy(state)
                proposed[field] = patch[field]
                proposed, changes = normalize_shapes(proposed)
                normalizations.extend(changes)
                state = proposed
                errors = invalid_fields(state, source)
                record['validation_errors'] = errors
                if field not in structural_fields(state) and (field != 'top_influential_citations' or field not in errors):
                    break
            except (ValueError, KeyError, TypeError) as error:
                record['validation_error'] = str(error)
    return state


def _apply_anchors(state, tasks, choices, anchors, removals):
    """Validate all choices before mutating; do not turn unknown indices into drops."""
    if not isinstance(choices, dict) or set(choices) != {t['id'] for t in tasks}:
        raise ValueError('Anchor response must cover exactly the requested task IDs')
    for task in tasks:
        choice = choices[task['id']]
        if choice != 'drop' and (type(choice) is not int or not 0 <= choice < len(task['candidates'])):
            raise ValueError('Anchor index must identify a provided literal source candidate')
    for task in reversed(tasks):
        owner = state
        for key in task['path'][:-1]:
            owner = owner[key]
        index, choice = task['path'][-1], choices[task['id']]
        if choice == 'drop':
            removals.append({'path': task['path'], 'entry': owner[index],
                             'reason': 'model_found_no_supporting_candidate'})
            del owner[index]
        else:
            anchor = task['candidates'][choice]
            owner[index] = {task['narrative']: anchor['quote']}
            anchors.append(dict(path=task['path'], **anchor))
    return state


def repair_document(paper, client, failure):
    """Resume audited V3 drafts; for older protocols recover the original generation."""
    from summarize import KEYS, validate_summary
    state = None
    saved = failure.get('draft_summary')
    resumed = (failure.get('repair_protocol') == PROTOCOL and isinstance(saved, dict)
               and set(saved) == set(KEYS))
    if resumed:
        state = {k: copy.deepcopy(saved[k]) for k in KEYS}
    for record in ([] if resumed else failure.get('attempts', [])):
        if record.get('phase') == 'generation' and record.get('finish_reason') != 'length':
            try:
                draft, _ = parse_json(record['response'])
                if isinstance(draft, dict) and set(draft) == set(KEYS):
                    state = {k: draft[k] for k in KEYS}
                    break
            except ValueError:
                pass
    if state is None:
        return dict(failure, repair_protocol=PROTOCOL)
    started = time.monotonic()
    source = paper['fulltext']
    journal = copy.deepcopy(failure['attempts'])
    state, changes = normalize_shapes(state)
    normalizations = copy.deepcopy(failure.get('shape_normalizations', [])) if resumed else []
    normalizations.extend(changes)
    seed = int(digest(paper['document_id'])[:8], 16) % 2147483000 + 100
    state = _schema_stage(state, source, client, journal, seed, normalizations)
    initial_tasks = len(quote_tasks(state, source, include_candidates=False))
    # Bound by the actual task inventory, not six calls regardless of inventory size.
    max_calls = min(64, max(4, 2 * int(math.ceil(initial_tasks / 8)) + 2))
    anchors = copy.deepcopy(failure.get('selected_source_anchors', [])) if resumed else []
    removals = copy.deepcopy(failure.get('removed_unsupported_entries', [])) if resumed else []
    for _ in range(max_calls):
        tasks = quote_tasks(state, source, limit=8)
        if not tasks:
            break
        public_tasks = []
        for task in tasks:
            public = {k: v for k, v in task.items() if k != 'path'}
            public['candidates'] = [dict(index=i, **c, source_context=source[max(0, c['start'] - 80):c['end'] + 80])
                                    for i, c in enumerate(task['candidates'])]
            public_tasks.append(public)
        user = ANCHOR_USER_TEMPLATE.format(source=source, tasks=json.dumps(public_tasks, ensure_ascii=False))
        record = _record(client, journal, ANCHOR_SYSTEM, user, 2048, seed + len(journal),
                         phase='source_anchor_selection', tasks=public_tasks, repair_protocol=PROTOCOL)
        try:
            choices, changes = parse_json(record['response'])
            record['normalizations'] = changes
            if record.get('finish_reason') == 'length':
                raise ValueError('Anchor response exhausted its output budget')
            state = _apply_anchors(state, tasks, choices, anchors, removals)
            state, changes = normalize_shapes(state)
            normalizations.extend(changes)
        except (ValueError, KeyError, TypeError) as error:
            record['validation_error'] = str(error)
    errors = invalid_fields(state, source)
    payload = dict(document_id=paper['document_id'], fulltext_sha256=paper['fulltext_sha256'],
                   attempts=journal, repair_protocol=PROTOCOL, shape_normalizations=normalizations,
                   resumed_v3_draft=resumed,
                   selected_source_anchors=anchors, removed_unsupported_entries=removals,
                   input_tokens=journal[0]['input_tokens_preflight'],
                   elapsed_seconds=failure.get('elapsed_seconds', 0) + time.monotonic() - started,
                   repair_limits={'field_attempts': 2, 'quote_tasks_per_call': 8, 'anchor_call_limit': max_calls})
    if errors:
        return dict(payload, failed=True, draft_summary=state, validation_errors=errors,
                    validation_error=str(errors))
    summary, narrative, spans = validate_summary(json.dumps(state, ensure_ascii=False), source)
    return dict(payload, summary=summary, judge_context=narrative, evidence_spans=spans)


def generate(paper, client, prompt, max_tokens, initial_attempts=2):
    # No giant multi-field rewrite: a complete first draft goes directly to V3.
    history, elapsed = [], 0
    for index in range(initial_attempts):
        document = generate_document(paper, client, prompt, max_tokens, 1, seed_offset=index)
        history.extend(document['attempts'])
        elapsed += document.get('elapsed_seconds', 0)
        if not document.get('failed') or document.get('draft_summary'):
            break
    document['attempts'], document['elapsed_seconds'] = history, elapsed
    if document.get('failed'):
        return repair_document(paper, client, document)
    return dict(document, repair_protocol=PROTOCOL)
