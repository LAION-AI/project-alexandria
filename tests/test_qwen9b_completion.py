import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries'))
from finish_qwen9b import normalize_claim_shapes, validate_existing
from summary_repair_v3 import structural_fields
from summary_runtime import invalid_fields
from summarize import KEYS, validate_summary


def test_bare_claim_narratives_are_preserved_but_not_accepted_without_evidence():
    state = dict.fromkeys(KEYS, '')
    state['executive_summary'] = 'Measured result.'
    state['claims'] = [dict(description='Result.', supporting_evidence=['Measured 42 units.'],
                            contradicting_evidence='', implications='Precision improved.')]
    failure = dict(draft_summary=state)
    before = copy.deepcopy(failure)
    normalized = normalize_claim_shapes(failure)
    claim = normalized['draft_summary']['claims'][0]
    assert claim['supporting_evidence'] == [{'Measured 42 units.': ''}]
    assert claim['implications'] == [{'Precision improved.': ''}]
    assert failure == before
    assert 'claims' not in structural_fields(normalized['draft_summary'])
    assert 'claims' in invalid_fields(normalized['draft_summary'], 'Measured 42 units. Precision improved.')
    assert len(normalized['shape_normalizations']) == 2


def test_existing_summary_validation_rejects_changed_narrative():
    state = dict.fromkeys(KEYS, '')
    state['executive_summary'] = 'Measured result.'
    summary, narrative, spans = validate_summary(json.dumps(state), 'Source.')
    document = dict(document_id='sample', fulltext_sha256='same', summary=summary,
                    judge_context=narrative, evidence_spans=spans)
    cache = dict(documents=[document])
    papers = [dict(document_id='sample', fulltext_sha256='same', fulltext='Source.')]
    assert validate_existing(cache, papers, {'sample'}) == {'sample'}
    document['judge_context'] = 'Changed'
    with pytest.raises(ValueError, match='changed'):
        validate_existing(cache, papers, {'sample'})
