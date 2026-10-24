"""Run a new full-source audit in a separate output directory on compute."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from common import CONDITIONS, cohort, condition_file, load
from copy_overlap_eval import audit_conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run full source audits on compute'
    assert not args.out.exists(), 'Choose a fresh output directory'
    rows = cohort()
    contexts = {c: {r['document_id']: load(condition_file(c, r['uid'])) for r in rows} for c in CONDITIONS}
    audit_conditions(rows, contexts, args.out)


if __name__ == '__main__':
    main()
