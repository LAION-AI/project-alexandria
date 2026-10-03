"""Test a shorter draft block on an idle GPU while fixed QA runs separately."""
import socket
from common import ROOT,load,write
from orchestrate import start_server,stop,OWNED,SERVER_BASE
from benchmark import run_benchmark

def main():
    gpu=2
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1',SERVER_BASE+gpu))==0:
            raise RuntimeError('Previous server still occupies the tuning port')
    server=start_server(gpu,'dflash4','ornith')
    result=run_benchmark(server['endpoint'],'dflash4',gpu)
    stop(server)
    write(ROOT/'outputs/supplemental_tuning.json',dict(mode='dflash4',complete=result['complete'],
        reason='Measured draft acceptance below 30 percent; test less verification work per proposal',
        qa_configuration='dflash8',quality_sampling_unchanged=True,
        note='Only non-holdout throughput examples used; QA remains on the previously frozen DFlash8 configuration'))

if __name__=='__main__':
    try:
        main()
    finally:
        for item in list(reversed(OWNED)):
            stop(item)
