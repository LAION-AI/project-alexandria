"""Tests for the frozen existing-summary benchmark, without model/network access."""
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries/evaluate.py'
SPEC = importlib.util.spec_from_file_location('summaries_evaluation', SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_exact_evidence_alignment():
    source = 'The ﬁtted\n  model gives 42 units.'
    start, end, quote = evaluation.evidence_span(source, 'The fitted model gives 42 units.')
    assert source[start:end] == quote == source
    with pytest.raises(ValueError):
        evaluation.evidence_span(source, 'The fitted model gives 43 units.')
    with pytest.raises(ValueError):
        evaluation.evidence_span(source, '')


def make_questions():
    return [{'question': 'Which paper-specific value in experiment ' + str(index) + '?',
             'options': {'A': '42 units', 'B': '41 units', 'C': '43 units', 'D': '44 units'},
             'answer': 'A', 'rationale': 'The experiment reports 42 units.',
             'evidence_quote': '42 units', 'difficulty': 'hard', 'category': 'result'}
            for index in range(10)]


def test_balanced_questions_and_source_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    qa = tmp_path / 'data/qa'
    qa.mkdir(parents=True)
    original = {'document_id': 'sample', 'questions': make_questions()}
    path = qa / 'sample.json'
    path.write_text(json.dumps(original))
    paper = {'document_id': 'sample', 'fulltext': 'Result: 42 units.', 'existing_summary': '42 units.'}
    questions = evaluation.questions_for(paper, 1)
    assert ''.join(q['answer'] for q in questions) == 'CDABCDABCD'
    assert all(q['options'][q['answer']] == '42 units' for q in questions)
    assert all(paper['fulltext'][q['evidence_start']:q['evidence_end']] == '42 units'
               for q in questions)
    assert json.loads(path.read_text()) == original
    original['questions'][0]['answer'] = ''
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        evaluation.questions_for(paper, 1)


def test_cluster_bootstrap_counts_invalid_as_incorrect():
    rows = [{'gold': 'A', 'predictions': {'no_context': None, 'original': 'A', 'summary': 'B'}}
            for _ in range(10)]
    stats = evaluation.statistics([{'rows': rows}])
    assert stats['no_context']['invalid'] == 10
    assert stats['no_context']['accuracy'] == 0
    assert stats['original']['ci95'] == [1.0, 1.0]
    assert stats['summary_minus_original']['ci95'] == [-1.0, -1.0]


def test_completed_checkpoint_reconciles_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluation.sys, 'argv', ['evaluate.py'])
    monkeypatch.setattr(evaluation, 'OpenAICompatibleBackend', lambda *a, **kw: None)
    qa = tmp_path / 'data/qa'
    qa.mkdir(parents=True)
    paper = {'document_id': 'sample', 'fulltext': 'Result: 42 units.', 'existing_summary': '42 units.'}
    (tmp_path / 'data/papers.json').write_text(json.dumps([paper]))
    (qa / 'sample.json').write_text(json.dumps({'document_id': 'sample', 'questions': make_questions()}))
    questions = evaluation.questions_for(paper, 0)
    stale = [dict(q, evidence_quote='wrong stale evidence') for q in questions]
    contexts = {'no_context': '', 'original': paper['fulltext'], 'summary': paper['existing_summary']}
    rows = [{'gold': q['answer'], 'source': 'arxiv',
             'predictions': {c: q['answer'] for c in evaluation.CONDITIONS},
             'responses': {c: {'prompt_sha256': hashlib.sha256(evaluation.historical_answer_prompt(
                 q['formatted_question'], contexts[c]).encode()).hexdigest()} for c in evaluation.CONDITIONS}}
            for q in questions]
    results = {'protocol': {}, 'documents': [{'document_id': 'sample', 'author_model': 'gpt-6-luna',
                             'questions': stale, 'rows': rows}], 'elapsed_judge_seconds': 1}
    (tmp_path / 'results.json').write_text(json.dumps(results))
    evaluation.main()
    updated = json.loads((tmp_path / 'results.json').read_text())
    assert updated['documents'][0]['questions'] == questions


def test_ku_condition_preserves_existing_controls(monkeypatch):
    monkeypatch.setattr(evaluation, 'CONDITIONS', ('no_context', 'original', 'summary', 'knowledge_units'))
    monkeypatch.setattr(evaluation, 'count_tokens', lambda *a: 20)
    paper = {'document_id': 'sample', 'viewer_config': 'arxiv', 'fulltext': 'Full text',
             'existing_summary': 'Existing summary', 'knowledge_units': 'KU FACTS'}
    questions = [{'question_index': i, 'formatted_question': 'Question ' + str(i), 'answer': 'A'}
                 for i in range(10)]
    previous = {'max_prompt_tokens': 100, 'rows': [
        {'predictions': {'no_context': 'B', 'original': 'A', 'summary': 'C'},
         'responses': {c: {'old': True, 'prompt_sha256': hashlib.sha256(evaluation.historical_answer_prompt(
             q['formatted_question'], {'no_context': '', 'original': paper['fulltext'],
                                      'summary': paper['existing_summary']}[c]).encode()).hexdigest()}
                       for c in ('no_context', 'original', 'summary')}} for q in questions]}

    class Backend:
        base_url = 'http://localhost/v1'
        calls = []

        def generate_batch(self, system, prompts, max_tokens):
            self.calls.extend(prompts)
            return [';A;'] * len(prompts)

    backend = Backend()
    document = evaluation.run_document(paper, questions, backend, previous, context_limit=32768)
    assert len(backend.calls) == 10
    assert all('KU FACTS' in p for p in backend.calls)
    for row in document['rows']:
        assert row['predictions'] == {'no_context': 'B', 'original': 'A', 'summary': 'C', 'knowledge_units': 'A'}
        assert row['responses']['original']['old'] is True
        assert row['responses']['knowledge_units']['context_limit'] == 32768


def summaries_module():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location('summary_generation', SCRIPT.parent / 'summarize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_prompt_wrapper_is_literal_and_never_executed(tmp_path):
    module = summaries_module()
    path = tmp_path / 'prompt.txt'
    path.write_text('SYSTEM_PROMPT_SUMMARY = r"""Role\\nRaw prompt"""')
    assert module.system_prompt(path) == r'Role\nRaw prompt'
    path.write_text('SYSTEM_PROMPT_SUMMARY = str(123)')
    with pytest.raises(ValueError):
        module.system_prompt(path)


def test_grounded_summary_schema_and_quotes():
    module = summaries_module()
    summary = {k: '' for k in module.KEYS}
    summary['executive_summary'] = 'The model gave a measured result.'
    summary['key_results'] = [{'The measured result was 42 units.': '42 units'}]
    value, context, spans = module.validate_summary(json.dumps(summary), 'Result: 42 units.')
    assert value == summary
    assert 'The measured result was 42 units.' in context
    assert len(spans) == 1 and spans[0]['quote'] == '42 units'
    summary['key_results'] = [{'The measured result was 43 units.': '43 units'}]
    with pytest.raises(ValueError):
        module.validate_summary(json.dumps(summary), 'Result: 42 units.')
    summary['key_results'] = [{'Claim.': 'one two three four five six'}]
    with pytest.raises(ValueError):
        module.validate_summary(json.dumps(summary), 'one two three four five six')


def test_summary_typographic_alignment_is_recorded():
    module = summaries_module()
    summary = {k: '' for k in module.KEYS}
    summary['executive_summary'] = 'A model was fitted.'
    summary['methodological_details'] = [{'The analysis fitted a model.': 'fitted model'}]
    value, _, spans = module.validate_summary(json.dumps(summary), 'A ﬁtted\nmodel was used.')
    assert value['methodological_details'][0]['The analysis fitted a model.'] == 'ﬁtted\nmodel'
    assert spans[0]['typographic_alignment'] is True
    assert spans[0]['original_quote'] == 'fitted model'


def test_summary_repair_reports_all_grounding_errors_without_replacing_evidence():
    module = summaries_module()
    summary = {k: '' for k in module.KEYS}
    summary['executive_summary'] = 'Measured results.'
    summary['key_results'] = [{'Claim one.': '43 units'}, {'Claim two.': 'missing text'},
                              {'Valid claim.': '42 units'}]
    summary['top_influential_citations'] = [{'Invented citation': 'Basis.', 'quotes': ['not present']}]
    raw = json.dumps(summary)
    errors = module.grounding_errors(raw, 'Result: 42 units.')
    assert len(errors) == 4
    assert 'key_results[0]' in errors[0] and 'key_results[1]' in errors[1]
    assert 'citation' in errors[2] and 'quotes[0]' in errors[3]
    assert json.loads(raw) == summary
    with pytest.raises(ValueError):
        module.validate_summary(raw, 'Result: 42 units.')


def runtime_module():
    summaries_module()
    import summary_runtime
    return summary_runtime


def test_summary_runtime_fence_removal_is_conservative():
    module = runtime_module()
    value, changes = module.parse_json('```json\n{"value":42}\n```')
    assert value == {'value': 42}
    assert changes == ['removed_enclosing_markdown_fence']
    with pytest.raises(ValueError):
        module.parse_json('Here is the JSON:\n{"value":42}')
    with pytest.raises(ValueError):
        module.parse_json('{"value":42,"value":43}')


def test_summary_runtime_checks_all_fields_not_only_first_failure():
    runtime = runtime_module()
    module = summaries_module()
    value = dict.fromkeys(module.KEYS, '')
    value['executive_summary'] = 'The results were measured.'
    value['key_results'] = [{'The result was 43 units.': '43 units'}]
    value['ethical_considerations'] = [{'No ethics are reported.': ''}]
    assert set(runtime.invalid_fields(value, 'Result: 42 units.')) == {'key_results', 'ethical_considerations'}


def test_summary_runtime_repairs_only_requested_fields_and_preserves_narrative():
    runtime = runtime_module()
    module = summaries_module()
    value = dict.fromkeys(module.KEYS, '')
    value['executive_summary'] = 'The experiment measured 42 units.'
    value['key_results'] = [{'The result was 42 units.': '43 units'}]

    class Client:
        requests = []

        def generate(self, system, user, max_tokens, seed):
            self.requests.append((system, user, max_tokens, seed))
            if len(self.requests) == 1:
                response = json.dumps(value)
            elif len(self.requests) == 2:
                # A repair is not allowed to quietly modify an unrequested summary field.
                response = json.dumps({'key_results': [{'The result was 42 units.': '42 units'}],
                                       'executive_summary': 'Injected change.'})
            else:
                response = json.dumps({'key_results': [{'The result was 42 units.': '42 units'}]})
            return {'response': response, 'finish_reason': 'stop', 'input_tokens_preflight': 100}

    client = Client()
    result = runtime.generate_document({'document_id': 'sample', 'fulltext': 'Result: 42 units.',
        'fulltext_sha256': hashlib.sha256(b'Result: 42 units.').hexdigest()}, client, 'USER PROMPT', 16000, 4)
    assert not result.get('failed')
    assert result['summary']['executive_summary'] == value['executive_summary']
    assert len(result['attempts']) == 3
    assert result['attempts'][1]['validation_error'].startswith('Repair must return exactly')
    assert client.requests[0][0] == 'USER PROMPT'
    assert client.requests[1][0] == runtime.REPAIR_SYSTEM
    assert client.requests[1][2] == 4096
    assert len({r[3] for r in client.requests}) == 3
    assert result['evidence_spans'][0]['quote'] == '42 units'


def test_summary_runtime_never_accepts_length_limited_output():
    runtime = runtime_module()
    module = summaries_module()
    value = dict.fromkeys(module.KEYS, '')
    value['executive_summary'] = 'A summary.'

    class Client:
        def generate(self, *args):
            return {'response': json.dumps(value), 'finish_reason': 'length', 'input_tokens_preflight': 100}

    result = runtime.generate_document({'document_id': 'sample', 'fulltext': 'Paper.',
        'fulltext_sha256': 'abc'}, Client(), 'USER PROMPT', 16000, 2)
    assert result['failed']
    assert all('budget exhausted' in r['validation_error'] for r in result['attempts'])


def test_source_anchor_retry_uses_only_exact_source_candidates():
    runtime = runtime_module()
    module = summaries_module()
    source = 'Result: 42 units were measured in the experiment.'
    value = dict.fromkeys(module.KEYS, '')
    value['executive_summary'] = 'The measured result was 42 units.'
    value['key_results'] = [{'The measured result was 42 units.': '43 units'}]
    initial = {'response': json.dumps(value), 'phase': 'generation', 'input_tokens_preflight': 100}
    failure = {'document_id': 'sample', 'failed': True, 'draft_summary': value,
               'attempts': [initial], 'elapsed_seconds': 10}

    class Client:
        def generate(self, system, user, max_tokens, seed):
            assert system == runtime.ANCHOR_SYSTEM
            tasks = json.loads(user.split('\n\nTASKS\n')[1])
            task = tasks[0]
            for candidate in task['candidates']:
                assert source[candidate['start']:candidate['end']] == candidate['quote']
            index = next(i for i, candidate in enumerate(task['candidates']) if '42' in candidate['quote'])
            return {'response': json.dumps({task['id']: index}), 'finish_reason': 'stop', 'input_tokens_preflight': 100}

    document = runtime.repair_saved_document({'document_id': 'sample', 'fulltext': source,
        'fulltext_sha256': 'abc'}, Client(), failure, 2)
    assert not document.get('failed')
    assert document['summary']['executive_summary'] == value['executive_summary']
    assert document['selected_source_anchors']
    assert '43 units' not in document['summary']['key_results'][0].values()


def test_source_anchor_drop_is_logged_and_empty_sequence_is_not_accepted():
    runtime = runtime_module()
    module = summaries_module()
    value = dict.fromkeys(module.KEYS, '')
    value['executive_summary'] = 'The source reports 42 units.'
    value['key_results'] = [{'An unsupported result was 99 units.': '99 units'}]
    failure = {'failed': True, 'draft_summary': value, 'attempts': [{'response': json.dumps(value),
        'phase': 'generation', 'input_tokens_preflight': 100}], 'elapsed_seconds': 10}

    class Client:
        def generate(self, system, user, max_tokens, seed):
            return {'response': '{"q0":"drop"}', 'finish_reason': 'stop', 'input_tokens_preflight': 100}

    document = runtime.repair_saved_document({'document_id': 'sample', 'fulltext': 'Result: 42 units.',
        'fulltext_sha256': 'abc'}, Client(), failure, 2)
    assert document['summary']['key_results'] == ''
    assert document['removed_unsupported_entries'][0]['entry'] == value['key_results'][0]


def test_restore_failed_draft_replays_only_requested_field_edits():
    runtime = runtime_module()
    module = summaries_module()
    value = dict.fromkeys(module.KEYS, '')
    value['executive_summary'] = 'Original executive narrative.'
    failure = {'attempts': [dict(response=json.dumps(value), phase='generation'),
        dict(response='{"key_results":[{"Result.":"42 units"}]}', phase='field_repair', requested_fields=['key_results']),
        dict(response='{"executive_summary":"Unrequested change."}', phase='field_repair', requested_fields=['key_results'])]}
    recovered = runtime.restore_draft(failure)
    assert recovered['executive_summary'] == 'Original executive narrative.'
    assert recovered['key_results'] == [{'Result.': '42 units'}]
