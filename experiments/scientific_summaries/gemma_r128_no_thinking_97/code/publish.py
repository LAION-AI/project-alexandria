"""Persist experiment evidence and copy English metadata/results to the authorized repo."""
from pathlib import Path
import shutil
import subprocess
from common import ROOT,load,write

DEST=Path('/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-gemma-r128-no-thinking-20261004')
REPO=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria/experiments/scientific_summaries/gemma_r128_no_thinking_97')


def publish():
    DEST.mkdir(parents=True,exist_ok=True);REPO.mkdir(parents=True,exist_ok=True)
    if (ROOT/'outputs/evaluation/complete.json').exists():
        from postprocess_copy_overlap import update
        update(ROOT)
    for name in ['code','inputs','outputs']:
        # No original model files, adapted/merged model weights, or compilation caches.
        subprocess.run(['rsync','-a','--exclude=__pycache__/','--exclude=*.tmp',str(ROOT/name)+'/',str(DEST/name)+'/'],check=True)
    for name in ['README.md','job.json','STATUS.json']:
        p=ROOT/name
        if p.exists():shutil.copy2(p,DEST/name);shutil.copy2(p,REPO/name)
    (REPO/'code').mkdir(exist_ok=True)
    for p in (ROOT/'code').glob('*'):
        if p.is_file() and p.suffix in ['.py','.sbatch']:shutil.copy2(p,REPO/'code'/p.name)
    for name in ['optimization_cohort.json','runtime.json']:
        shutil.copy2(ROOT/'inputs'/name,REPO/name)
    for name in ['merge.json','arm_status.json','runtime_errors.json','allocation_accounting.json','token_proportional_preestimate.json']:
        p=ROOT/'outputs'/name
        if p.exists():shutil.copy2(p,REPO/name)
    for name in ['optimization','server_commands']:
        source=ROOT/'outputs'/name
        if source.exists():
            target=REPO/name;target.mkdir(exist_ok=True)
            for p in source.glob('*.json'):shutil.copy2(p,target/p.name)
    for source in (ROOT/'outputs/probes').glob('*/c*.json') if (ROOT/'outputs/probes').exists() else []:
        target=REPO/'probe_timings'/source.parent.name;target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target/source.name)
    output=ROOT/'outputs/evaluation';target=REPO/'evaluation';target.mkdir(exist_ok=True)
    if (output/'protocol.json').exists():shutil.copy2(output/'protocol.json',target/'protocol.json')
    if (output/'complete.json').exists():
        for p in output.iterdir():
            if p.is_file() and p.suffix in ['.json','.jsonl','.md','.csv']:shutil.copy2(p,target/p.name)


if __name__=='__main__':publish()
