import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scientific_summaries'))
from export_testset import build
from run_ornith9b import prepare_draft
from summary_repair_v3 import PROTOCOL
from summary_runtime import invalid_fields
from summarize import KEYS


def test_testset_is_exact_frozen_evaluated_cohort():
    bundle, manifest = build()
    assert bundle['paper_count'] == 97 and bundle['question_count'] == 970
    assert manifest['subsets'] == {'arxiv': 50, 'bethgelab': 47}
    assert len(manifest['source_file_sha256']) == 100
    for paper in bundle['papers']:
        assert len(paper['questions']) == 10
        for question in paper['questions']:
            assert set(question['options']) == set('ABCD')
            assert question['answer'] in 'ABCD'
            assert paper['fulltext'][question['evidence_start']:question['evidence_end']] == question['evidence_quote']


def test_ornith_preserves_original_v2_detail_and_does_not_accept_unverified_claims():
    state = dict.fromkeys(KEYS, '')
    state['executive_summary'] = 'Detailed original narrative.'
    state['claims'] = [dict(description='Result.', supporting_evidence=['42 units were measured.'],
                            contradicting_evidence='', implications='Precision improved.')]
    shortened = copy.deepcopy(state)
    shortened['executive_summary'] = 'Shortened stub.'
    failure = dict(draft_summary=shortened, attempts=[dict(phase='generation', finish_reason='stop', response=json.dumps(state))])
    prepared = prepare_draft(failure)
    assert prepared['draft_summary']['executive_summary'] == 'Detailed original narrative.'
    assert prepared['repair_protocol'] == PROTOCOL
    assert prepared['draft_summary']['claims'][0]['supporting_evidence'] == [{'42 units were measured.': ''}]
    assert 'claims' in invalid_fields(prepared['draft_summary'], '42 units were measured. Precision improved.')
    assert failure['draft_summary']['executive_summary'] == 'Shortened stub.'


def test_ornith_v3_resume_does_not_restore_already_corrected_bad_quotes():
    state = dict.fromkeys(KEYS, '')
    state['executive_summary'] = 'Detailed original narrative.'
    state['key_results'] = [{'Measured result.': '43 units'}]
    corrected = copy.deepcopy(state)
    corrected['key_results'] = [{'Measured result.': '42 units'}]
    failure = dict(repair_protocol=PROTOCOL, draft_summary=corrected,
                   attempts=[dict(phase='generation', finish_reason='stop', response=json.dumps(state))])
    assert prepare_draft(failure)['draft_summary']['key_results'] == corrected['key_results']
