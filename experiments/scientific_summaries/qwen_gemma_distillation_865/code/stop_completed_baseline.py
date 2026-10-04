"""Retire only the owned baseline server after all 97 baseline results are saved."""
import os
from pathlib import Path
import signal
from common import ROOT,load

assert os.environ['SLURM_JOB_ID']=='2171850'
performance=load(ROOT/'outputs/evaluation/gemma12_base-generation-performance.json')
assert performance['papers']==97
paths=list((ROOT/'outputs/evaluation/generation/gemma12_base').glob('*/result.json'))
assert len(paths)==97 and all(load(p)['status']=='generated' for p in paths)
owned=[]
for folder in Path('/proc').glob('[0-9]*'):
    try:
        args=(folder/'cmdline').read_bytes().decode().split('\0')
        if 'vllm.entrypoints.openai.api_server' not in args:continue
        if '--port' not in args or args[args.index('--port')+1]!='19200':continue
        if str(ROOT/'models/gemma-4-12b-it') not in args:continue
        pid=int(folder.name)
        assert os.getpgid(pid)==pid
        owned.append(pid)
    except (OSError,UnicodeError):continue
assert len(owned)<=1
for pid in owned:
    os.killpg(pid,signal.SIGTERM)
    print('Retired completed owned baseline server:',pid,flush=True)
