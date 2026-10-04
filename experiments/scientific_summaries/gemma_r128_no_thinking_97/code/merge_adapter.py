"""Merge the completed adapter into a separate BF16 checkpoint on one GPU."""
import os
import shutil
import time
import torch
from transformers import AutoModelForImageTextToText
from peft import PeftModel
from common import ROOT, load, write, digest


def main():
    out=ROOT/'models/merged-r128-bf16'
    if (ROOT/'outputs/merge.json').exists():
        assert (out/'config.json').exists()
        return
    start=time.monotonic()
    base=AutoModelForImageTextToText.from_pretrained(ROOT/'models/gemma-4-12b-it',
        local_files_only=True,dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='eager')
    model=PeftModel.from_pretrained(base,ROOT/'models/r128-adapter',is_trainable=False)
    model.cuda().eval()
    # Keep the FP32 adapter until merge; save all resulting weights as BF16.
    merged=model.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.bfloat16)
    out.mkdir(parents=True,exist_ok=True)
    merged.save_pretrained(out,safe_serialization=True,max_shard_size='5GB')
    for path in (ROOT/'models/gemma-4-12b-it').iterdir():
        if path.is_file() and path.suffix in ['.json','.jinja','.model'] and path.name not in ['config.json','model.safetensors.index.json']:
            shutil.copy2(path,out/path.name)
    write(ROOT/'outputs/merge.json',dict(status='complete',model=str(out),dtype='BF16',
        base_model='google/gemma-4-12B-it',base_revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
        adapter_sha256=digest(ROOT/'models/r128-adapter/adapter_model.safetensors'),
        adapter_config_sha256=digest(ROOT/'models/r128-adapter/adapter_config.json'),
        elapsed_seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID'),
        inference_numerical_equivalence_assumed=False,
        weight_files={p.name:digest(p) for p in out.glob('*.safetensors')}))


if __name__=='__main__':main()
