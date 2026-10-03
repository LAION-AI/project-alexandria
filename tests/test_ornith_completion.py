import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries'))
from finish_ornith9b import compact_journals, output_schema, field_schema
from summary_runtime import ANCHOR_SYSTEM, REPAIR_SYSTEM
from summarize import KEYS


def test_anchor_schema_allows_only_exact_task_ids_and_candidate_indices():
    tasks = [{'id': 'q0', 'candidates': [{'quote': 'A'}]}, {'id': 'q1', 'candidates': []}]
    schema = output_schema([{'content': ANCHOR_SYSTEM}, {'content': '\n\nTASKS\n' + json.dumps(tasks)}])
    assert schema['required'] == ['q0', 'q1']
    assert schema['additionalProperties'] is False
    assert schema['properties'] == {'q0': {'enum': [0, 'drop']}, 'q1': {'enum': ['drop']}}


def test_generation_schema_requires_all_fields_and_bounds_repetition():
    schema = output_schema([{'content': 'Original generation prompt'}, {'content': 'Source'}])
    assert tuple(schema['properties']) == KEYS
    assert schema['required'] == list(KEYS)
    assert schema['additionalProperties'] is False
    assert field_schema('research_context')['anyOf'][1]['maxItems'] == 12
    claim = field_schema('claims')['anyOf'][1]['items']
    assert claim['required'] == ['description', 'supporting_evidence', 'contradicting_evidence', 'implications']


def test_field_repair_schema_cannot_return_other_fields():
    schema = output_schema([{'content': REPAIR_SYSTEM}, {'content': '\nFIELD\nclaims\nOther data'}])
    assert schema['required'] == ['claims']
    assert schema['additionalProperties'] is False


def test_sharded_journals_keep_full_raw_calls_and_lean_attempt_metadata(tmp_path):
    call = dict(response='Raw generation', tasks=[{'id': 'q0'}], phase='generation', seed=7)
    record = dict(document_id='sample', attempts=[call], judge_context='Preserved narrative')
    cache = dict(documents=[record], failures=[dict(record, failed=True)])
    compact = compact_journals(cache, tmp_path)
    for group in ('documents', 'failures'):
        lean = compact[group][0]
        assert lean['attempts'] == [dict(phase='generation', seed=7)]
        file = tmp_path / lean['raw_journal_file']
        assert json.loads(file.read_text()) == cache[group][0]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == lean['raw_journal_sha256']
    assert cache['documents'][0]['attempts'][0]['response'] == 'Raw generation'
