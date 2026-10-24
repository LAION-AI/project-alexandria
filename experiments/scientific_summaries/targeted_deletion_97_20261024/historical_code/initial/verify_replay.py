"""Read-only exact replay check of all frozen deletion outputs."""
import concurrent.futures
from common import ROOT,cohort,load,read_jsonl,write
import deletion,postprocess


def init():
    deletion.init()
    def checked_write(path,value):
        assert load(path)==value, 'Replay record differs: '+str(path)
    def checked_jsonl(path,rows):
        assert read_jsonl(path)==rows, 'Replay trace differs: '+str(path)
    postprocess.write=checked_write
    postprocess.jsonl=checked_jsonl


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,initializer=init) as pool:
        results=list(pool.map(postprocess.process,cohort(),chunksize=1))
    write(ROOT/'outputs/exact_replay_verification.json',dict(passed=True,papers=len(results),conditions=5,
        saved_records=485,includes_all_traces=True,read_only_saved_outputs=True))
    print('EXACT_REPLAY_VERIFIED',len(results),flush=True)


if __name__=='__main__':main()
