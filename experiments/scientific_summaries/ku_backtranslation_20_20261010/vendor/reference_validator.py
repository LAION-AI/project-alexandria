# Extracted unchanged functions from the pinned Alexandria commit.

import copy, json



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

KEYS = ('title', 'authors', 'field_subfield', 'type_of_paper', 'executive_summary',
        'research_context', 'research_question_and_hypothesis', 'methodological_details',
        'procedures_and_architectures', 'key_results', 'interpretation_and_theoretical_implications',
        'contradictions_and_limitations', 'claims', 'data_and_code_availability',
        'robustness_and_ablation_notes', 'ethical_considerations', 'key_figures_tables',
        'top_influential_citations', 'three_takeaways')

GROUNDED = KEYS[5:12] + KEYS[13:17] + ('three_takeaways',)

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
