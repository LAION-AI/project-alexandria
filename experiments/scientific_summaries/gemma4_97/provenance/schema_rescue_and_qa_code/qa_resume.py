import hashlib
import os
import time
from common import ROOT,load,write
from orchestrate import status,start_server,stop,OWNED
from score import score_all
from audit_final import main as audit

def main():
    if not load(ROOT/'outputs/schema_reconciliation.json')['passed']:
        raise RuntimeError('Source-only reconciliation incomplete')
    started=time.monotonic()
    write(ROOT/'outputs/qa_code_snapshot.json',dict(job_id=os.getenv('SLURM_JOB_ID'),
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'rescue_code').glob('*') if p.is_file()}))
    status('loading_fixed_qa_judge_after_schema_rescue')
    judge=start_server(0,'judge','qwen25')
    status('qa_evaluation',new_conditions=4,papers=97,questions_per_condition=970)
    score_all(judge['endpoint']);stop(judge);audit()
    write(ROOT/'outputs/complete.json',dict(job_id=os.getenv('SLURM_JOB_ID'),generation_job_id='2166049',
        allocation_job_ids=['2166049',os.getenv('SLURM_JOB_ID')],qa_elapsed_seconds=time.monotonic()-started,
        allocated_gpus=4,new_model_conditions=4,comparison_conditions=14,
        intervention='Original repeated generation stopped after verified source-only finite-schema rescue candidates covered all raw/corrected outputs.'))
    status('complete',papers=97,questions=970,report=str(ROOT/'outputs/report.md'))

if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=repr(error));raise
    finally:
        for server in list(reversed(OWNED)):stop(server)
