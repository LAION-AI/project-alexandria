"""Offline tests for the standalone methods report and scaling arithmetic."""
import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries/pipeline_report.py'
SPEC = importlib.util.spec_from_file_location('summary_pipeline_report', SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class Structure(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.assets = [], [], []
        self.language = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'html':
            self.language = values.get('lang')
        if 'id' in values:
            self.ids.append(values['id'])
        if tag == 'a':
            self.links.append(values['href'])
        if tag in ('script', 'img', 'iframe') or tag == 'link' and values.get('rel') == 'stylesheet':
            self.assets.append(values)


def attempt(phase, duration=1):
    return dict(phase=phase, seed=123, response='{}', elapsed_seconds=duration,
                user_prompt_sha256='u', system_prompt_sha256='s',
                usage={'prompt_tokens': 100, 'completion_tokens': 50})


def test_gpu_hours_use_tokens_and_gpu_count_correctly():
    assert report.gpu_hours(3600, 1000, 100, (1000, 100, 1)) == 2
    assert report.gpu_hours(7200, 1000, 100, (1000, 100, 1)) == 4
    assert report.gpu_hours(3600, 1000, 100, (1000, 100, .5)) == 4
    with pytest.raises(ValueError):
        report.gpu_hours(1, 1, 1, (1, 1, 1.1))


def test_saved_attempt_copies_are_deduplicated_but_new_calls_are_not():
    a, b = attempt('generation'), attempt('field_repair', 2)
    d = dict(document_id='one', fulltext_sha256=report.sha('paper text'),
             attempts=[a, b], judge_context='summary text')
    cache = dict(documents=[d], failures=[dict(document_id='one', attempts=[a, b, attempt('field_repair', 3)])],
                 elapsed_seconds=3600, config={'allocated_gpus': 2})
    values = report.measured_workload(cache, [{'document_id': 'one', 'fulltext': 'paper text'}])
    assert values['distinct_saved_calls'] == 3
    assert values['initial_input_tokens'] == 100
    assert values['validated_input_tokens'] == 300
    assert values['successful_journal_calls_per_paper'] == 2
    assert values['checkpointed_gpu_hours_proxy'] == 2


def test_missing_usage_or_changed_source_is_not_silently_estimated():
    a = attempt('generation')
    a['usage'] = {}
    cache = dict(documents=[dict(document_id='one', fulltext_sha256=report.sha('text'),
                                 attempts=[a], judge_context='text')], elapsed_seconds=1,
                 config={'allocated_gpus': 1})
    with pytest.raises(ValueError, match='Missing measured token'):
        report.measured_workload(cache, [{'document_id': 'one', 'fulltext': 'text'}])
    with pytest.raises(ValueError, match='Source text changed'):
        report.measured_workload(cache, [{'document_id': 'one', 'fulltext': 'changed'}])


def test_complete_prompt_inventory_matches_frozen_system_prompts():
    items = {item[0]: item for item in report.prompt_inventory()}
    snapshot = report.load(report.ROOT / 'summary_runs/qwen27b/summary_prompt_snapshot.json')
    assert items['summary-system'][2] == snapshot['effective_system_prompt']
    assert items['repair-system'][2] == snapshot['repair_system_prompt']
    assert items['anchor-system'][2] == snapshot['anchor_selection_system_prompt']
    assert report.sha(items['summary-system'][2]) == snapshot['effective_prompt_sha256']
    assert '{paper_text}' in items['summary-user'][2]
    assert '{tasks_json}' in items['anchor-user'][2]
    assert items['judge-system'][2] == 'You are a very smart very intelligence assistant who is very helpful.'
    assert '{context}' in items['judge-user'][2] and '{question}' in items['judge-user'][2]
    assert 'historical_sanitize' in items['judge-sanitizer'][2]


def test_built_html_is_standalone_english_and_has_working_local_links():
    output, provenance = report.build()
    parser = Structure()
    parser.feed(output)
    assert parser.language == 'en'
    assert len(parser.ids) == len(set(parser.ids))
    assert not parser.assets
    assert '@@COMMIT@@' not in output
    for href in parser.links:
        if href.startswith('#'):
            assert href[1:] in parser.ids
        elif not href.startswith(('https://', 'http://')):
            assert (report.ROOT / href).is_file(), href
    assert len(provenance['prompts']) >= 16
    assert {'v3-field-user', 'v3-anchor-user', 'v3-field-examples'} <= {p['id'] for p in provenance['prompts']}
    assert len([p for p in provenance['input_sha256'] if p.startswith('data/qa/')]) == 97
    assert provenance['measured_workload']['documents'] == 97
    assert provenance['measured_workload']['distinct_saved_calls'] == 451


def test_built_estimates_are_ordered_and_match_prior_calculations():
    _, data = report.build()
    assert len(data['estimates']) == 4
    for row in data['estimates']:
        costs = row['gpu_hours']
        assert costs['fast'] < costs['central'] < costs['slow']
    central = [round(row['gpu_hours']['central']) for row in data['estimates']]
    assert central == [82649, 198830, 30394, 74352]


def test_companion_is_machine_readable_and_checksummed():
    data_path = report.ROOT / 'summary_pipeline.json'
    saved = report.load(data_path)
    for item in saved['prompts']:
        assert report.sha(item['text']) == item['sha256']
    for line in (report.ROOT / 'summary_pipeline.SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert report.sha((report.ROOT / name).read_bytes()) == digest
    assert json.loads(data_path.read_text())['as_of'] == '2026-10-02'
