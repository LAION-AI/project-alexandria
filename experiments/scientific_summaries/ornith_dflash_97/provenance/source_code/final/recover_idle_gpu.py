"""Recover the completed Ornith cohort on a released GPU in the same allocation."""
import concurrent.futures
import socket
from common import ROOT,load,write
from finalize import recover,rebuild
from orchestrate import start_server,stop,OWNED,SERVER_BASE

def main():
    model='ornith_dflash'
    gpu=1  # Primary QA uses GPU 0; the Qwen baseline continues on GPU 3.
    completion=load(ROOT/'outputs/cohorts'/model/'complete.json')
    if completion['papers']!=97:
        raise ValueError('Ornith cohort still running')
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1',SERVER_BASE+gpu))==0:
            raise RuntimeError('The original Ornith server has not released its port')
    sources=load(ROOT/'inputs/eval_sources_only.json')
    pending=[p for p in sources if not all(load(ROOT/'outputs/cohorts'/model/'documents'/p['document_id']/'document.json').get(c) for c in ('raw','corrected'))]
    if pending:
        server=start_server(gpu,'dflash8','summary-model')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda p:recover(p,model,server['endpoint']),pending))
        stop(server)
    rebuild(model,sources)
    write(ROOT/'outputs/idle_gpu_recovery.json',dict(model=model,recovery_papers=len(pending),complete=True))

if __name__=='__main__':
    try:
        main()
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
