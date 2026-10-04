"""Copy completed English results into the authorized project checkout without credentials."""
from pathlib import Path
import shutil

from common import ROOT,load,write

DEST=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria/experiments/scientific_summaries/qwen_gemma_distillation_865')


def publish():
    DEST.mkdir(exist_ok=True)
    for folder in ['ornith','evaluation']:
        source=ROOT/'outputs'/folder
        if not (source/'complete.json').exists():continue
        target=DEST/folder;target.mkdir(exist_ok=True)
        names=['performance.json','complete.json'] if folder=='ornith' else [
            'RESULTS.md','report.json','protocol.json','qa-results.json','complete.json',
            'gemma12_base-generation-performance.json','gemma12_r64-generation-performance.json',
            'gemma12_r128-generation-performance.json',
            'gemma12_base-summaries.jsonl','gemma12_r64-summaries.jsonl','gemma12_r128-summaries.jsonl']
        for name in names:
            path=source/name
            if path.exists():shutil.copy2(path,target/name)
    for rank in [64,128]:
        source=ROOT/'outputs'/('train-r'+str(rank))
        if not (source/'result.json').exists():continue
        record=load(source/'result.json')
        record.pop('trainable_module_names',None)
        write(DEST/('training-r'+str(rank)+'.json'),record)
        shutil.copy2(source/'timings.jsonl',DEST/('training-r'+str(rank)+'-timings.jsonl'))
    for name in ['allocation_accounting.json','hf_upload.json']:
        path=ROOT/'outputs'/name
        if path.exists():shutil.copy2(path,DEST/name)


if __name__=='__main__':publish()
