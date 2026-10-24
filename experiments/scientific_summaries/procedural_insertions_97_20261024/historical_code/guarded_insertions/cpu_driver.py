"""Re-evaluate a conservative whole-paper fallback without new model calls."""
import os,time,hashlib
from common import ROOT,load,write

def main():
    tick=time.monotonic()
    try:
        protocol=load(ROOT/'inputs/protocol.json')
        assert all(hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==sha for name,sha in protocol['code_sha256'].items())
        from postprocess import main as cpu
        cpu()
        from score_cached import main as qa
        qa()
        from review import main as review
        review()
        assert load(ROOT/'outputs/review_complete.json')['new_unique_pairs']==0
        from finish import main as finish
        finish()
        write(ROOT/'outputs/status.json',dict(phase='complete',job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-tick,additional_model_calls=0))
    except BaseException as e:
        write(ROOT/'outputs/failure.json',dict(error=repr(e)));raise

if __name__=='__main__':main()
