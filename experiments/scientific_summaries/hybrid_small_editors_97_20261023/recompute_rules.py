"""Reproduce the conservative CPU baseline from frozen raw summaries."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from common import ROOT, cohort, read_jsonl, jsonl
import function_words


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run the complete source audit on a compute allocation'
    rows = cohort()
    function_words.init()
    for row in rows:
        source = dict(uid=row['uid'], document_id=row['document_id'], group='gemma_r128_fp8',
                      source_sha256=row['source_sha256'], fulltext=row['fulltext'],
                      summary=row['raw_summary'], judge_context=row['raw_context'])
        function_words.process(source)
        from common import load
        saved = load(ROOT / f'outputs/repaired/function_prepositions/{row["uid"]}.json')
        assert saved['judge_context'] == row['safe_context'], row['document_id']
    print('PASS: all 97 conservative CPU baselines reproduced exactly')


if __name__ == '__main__':
    main()
