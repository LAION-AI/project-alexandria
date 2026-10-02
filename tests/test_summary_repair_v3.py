"""Strict V3 regression tests: no model, GPU, or network required."""
import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries'
sys.path.insert(0, str(ROOT))
repair = importlib.import_module('summary_repair_v3')
summary = importlib.import_module('summarize')
runtime = importlib.import_module('summary_runtime')


def draft():
    value = dict.fromkeys(summary.KEYS, '')
    value['executive_summary'] = 'The measured result was 42 units.'
    return value


def paper():
    text = 'Result: 42 units were measured in the experiment.'
    return {'document_id': 'sample', 'fulltext': text, 'fulltext_sha256': runtime.digest(text)}


def failed(value):
    return {'failed': True, 'draft_summary': copy.deepcopy(value), 'elapsed_seconds': 0,
            'attempts': [{'phase': 'generation', 'finish_reason': 'stop',
                          'response': json.dumps(value), 'input_tokens_preflight': 100}]}


def response(value):
    return {'response': json.dumps(value), 'finish_reason': 'stop', 'input_tokens_preflight': 100}


def test_unambiguous_alias_conversion_preserves_strings_and_does_not_validate_them():
    value = draft()
    value['key_results'] = [{'narrative': 'The result was 99 units.', 'evidence': '99 units'}]
    value['claims'] = [{'implications': [], 'contradicting_evidence': '',
                        'supporting_evidence': [{'statement': 'Measured 42 units.', 'quote': '42 units'}],
                        'description': 'Measured result.'}]
    value['top_influential_citations'] = [{'citation': 'Missing source citation', 'justification': 'Method.',
                                         'quotes': ['not present']}]
    normalized, changes = repair.normalize_shapes(value)
    assert normalized['key_results'] == [{'The result was 99 units.': '99 units'}]
    assert value['key_results'][0]['narrative'] == 'The result was 99 units.'
    assert normalized['claims'][0]['supporting_evidence'] == [{'Measured 42 units.': '42 units'}]
    assert normalized['claims'][0]['implications'] == ''
    assert tuple(normalized['claims'][0]) == ('description', 'supporting_evidence', 'contradicting_evidence', 'implications')
    assert len(changes) >= 4
    errors = runtime.invalid_fields(normalized, paper()['fulltext'])
    assert {'key_results', 'top_influential_citations'} <= set(errors)


def test_ambiguous_or_extra_alias_keys_are_not_silently_discarded():
    value = draft()
    value['key_results'] = [{'narrative': 'Result.', 'source': '42 units', 'confidence': .9}]
    normalized, _ = repair.normalize_shapes(value)
    assert normalized['key_results'] == value['key_results']
    assert 'key_results' in repair.structural_fields(normalized)


def test_all_entries_are_checked_for_structure_even_after_an_earlier_invalid_quote():
    value = draft()
    value['key_results'] = [{'First result.': ''}, 'Malformed second entry']
    assert 'key_results' in repair.structural_fields(value)


def test_more_than_six_quote_batches_complete_without_losing_supported_narrative():
    value = draft()
    value['key_results'] = [{f'Measurement {i} reports 42 units.': '43 units'} for i in range(65)]

    class Client:
        calls = 0

        def generate(self, system, user, budget, seed):
            assert system == runtime.ANCHOR_SYSTEM
            tasks = json.loads(user.split('\n\nTASKS\n')[1])
            assert 0 < len(tasks) <= 8
            self.calls += 1
            choices = {}
            for t in tasks:
                choices[t['id']] = next(c['index'] for c in t['candidates'] if '42' in c['quote'])
                assert all(paper()['fulltext'][c['start']:c['end']] == c['quote'] for c in t['candidates'])
            return response(choices)

    client = Client()
    result = repair.repair_document(paper(), client, failed(value))
    assert not result.get('failed')
    assert client.calls == 9
    assert len(result['summary']['key_results']) == 65
    assert list(result['summary']['key_results'][64]) == ['Measurement 64 reports 42 units.']
    assert len(result['selected_source_anchors']) == 65
    assert result['summary']['executive_summary'] == value['executive_summary']


def test_schema_repairs_request_one_field_with_an_explicit_shape():
    value = draft()
    value['key_results'] = ['The result was 42 units.']
    value['ethical_considerations'] = ['No reported ethics.']

    class Client:
        fields = []

        def generate(self, system, user, budget, seed):
            assert system == runtime.REPAIR_SYSTEM
            field = user.split('\nFIELD\n')[1].split('\n')[0]
            self.fields.append(field)
            assert 'EXACT_OUTPUT_SHAPE_EXAMPLE' in user
            return response({field: [{'The result was 42 units.': '42 units'}] if field == 'key_results' else ''})

    client = Client()
    result = repair.repair_document(paper(), client, failed(value))
    assert not result.get('failed')
    assert client.fields == ['key_results', 'ethical_considerations']
    assert result['summary']['ethical_considerations'] == ''


def test_invalid_anchor_batch_never_partially_mutates_state():
    value = draft()
    value['key_results'] = [{'A.': '43 units'}, {'B.': '43 units'}]
    tasks = runtime.quote_tasks(value, paper()['fulltext'])
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match='provided literal source'):
        repair._apply_anchors(value, tasks, {'q0': 'drop', 'q1': 999}, [], [])
    assert value == before


def test_unsupported_entries_can_be_dropped_but_removals_remain_auditable():
    value = draft()
    value['ethical_considerations'] = [{'No ethics approval is reported.': ''}]

    class Client:
        def generate(self, *args):
            return response({'q0': 'drop'})

    result = repair.repair_document(paper(), Client(), failed(value))
    assert result['summary']['ethical_considerations'] == ''
    assert result['removed_unsupported_entries'][0]['entry'] == value['ethical_considerations'][0]


def test_original_generation_is_preferred_to_destructively_shortened_old_draft():
    value = draft()
    value['key_results'] = [{'The detailed result was 42 units.': '42 units'}]
    failure = failed(value)
    failure['draft_summary']['key_results'] = [{'Tiny stub.': '42 units'}]
    result = repair.repair_document(paper(), object(), failure)
    assert result['summary']['key_results'] == value['key_results']


def test_limited_task_inventory_does_not_compute_unused_candidates(monkeypatch):
    value = draft()
    value['key_results'] = [{str(i): '43 units'} for i in range(20)]
    calls = []
    monkeypatch.setattr(runtime, 'source_candidates', lambda *a: calls.append(a) or [])
    assert len(runtime.quote_tasks(value, paper()['fulltext'], include_candidates=False)) == 20
    assert not calls
    assert len(runtime.quote_tasks(value, paper()['fulltext'], limit=8)) == 8
    assert len(calls) == 8


def test_citation_shape_example_cannot_introduce_fake_citation_text():
    assert repair.field_example('top_influential_citations') == {'top_influential_citations': ''}
    assert 'actual bibliography entries' in repair.FIELD_USER_TEMPLATE


def test_v3_resume_preserves_repaired_narrative_and_audit_history():
    initial = draft()
    initial['key_results'] = [{'Detailed result: 42 units.': '43 units'}]
    failure = failed(initial)
    failure['repair_protocol'] = repair.PROTOCOL
    failure['draft_summary']['key_results'][0]['Detailed result: 42 units.'] = '42 units'
    failure['draft_summary']['top_influential_citations'] = [{'Fake citation': 'Method.', 'quotes': ['42 units']}]
    failure['selected_source_anchors'] = [{'path': ['key_results', 0], 'quote': '42 units'}]
    failure['removed_unsupported_entries'] = [{'entry': {'Unsupported': 'fake'}}]
    failure['shape_normalizations'] = [{'operation': 'prior_rewrap'}]

    class Client:
        calls = 0

        def generate(self, system, user, *args):
            self.calls += 1
            assert system == runtime.REPAIR_SYSTEM
            assert '\nFIELD\ntop_influential_citations\n' in user
            return response({'top_influential_citations': ''})

    client = Client()
    result = repair.repair_document(paper(), client, failure)
    assert not result.get('failed') and result['resumed_v3_draft']
    assert client.calls == 1
    assert result['summary']['key_results'] == [{'Detailed result: 42 units.': '42 units'}]
    for key in ('selected_source_anchors', 'removed_unsupported_entries', 'shape_normalizations'):
        assert result[key] == failure[key]


def test_explicit_failed_cache_recovery_checks_identity_and_preserves_provenance(tmp_path):
    config = dict(model='pinned', repair_protocol=repair.PROTOCOL, repair_implementation_sha256='new')
    old = dict(config, repair_implementation_sha256='old')
    path = tmp_path / 'failed.json'
    cache = dict(config=old, documents=[], failures=[failed(draft())], elapsed_seconds=12.5)
    path.write_text(json.dumps(cache))
    recovered = summary.recover_failed_checkpoint(config, path)
    assert recovered['elapsed_seconds'] == 12.5
    assert recovered['config']['repair_implementation_sha256'] == 'new'
    assert recovered['config']['recovery']['prior_repair_implementation_sha256'] == 'old'
    assert recovered['config']['recovery']['checkpoint_sha256'] == runtime.digest(path.read_text())
    with pytest.raises(ValueError, match='identity'):
        summary.recover_failed_checkpoint(dict(config, model='different'), path)
    cache['documents'] = [{'already_valid': True}]
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError, match='no valid documents'):
        summary.recover_failed_checkpoint(config, path)
