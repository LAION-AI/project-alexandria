"""Download only pinned weights/tokenizers; prepare source-only frozen inputs."""
import importlib.metadata
import platform
import time
import shutil
from common import ROOT,load,write
from prepare import main as prepare
from critical_values import VERSION as CRITICAL_VERSION


def main():
    start=time.monotonic()
    pins=load(ROOT/'inputs/model_pins.json')
    for label,metadata in pins.items():
        folder=ROOT/'models'/label
        assert (ROOT/'inputs/models_staged.json').exists(), 'Stage model files on a login node first'
        assert (folder/'config.json').exists() and list(folder.glob('*.safetensors')), label
        print('Pinned model ready',label,metadata['revision'],flush=True)
    write(ROOT/'inputs/runtime.json',dict(platform=platform.machine(),
          packages={k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','tokenizers','huggingface_hub']},
          provision_seconds=time.monotonic()-start))
    write(ROOT/'inputs/job_history.json',load(ROOT/'job.json'))
    snapshot=ROOT/'inputs/executed_code';snapshot.mkdir(exist_ok=True)
    for file in (ROOT/'code').glob('*.py'):shutil.copy2(file,snapshot/file.name)
    # Archive earlier guard outputs before workers start, so QA never races stale acceptance.
    quality=ROOT/'outputs/quality'
    if quality.exists():
        for folder in quality.iterdir():
            marker=folder/'complete.json'
            if folder.is_dir() and (not marker.exists() or load(marker).get('critical_values_version')!=CRITICAL_VERSION):
                dest=ROOT/'outputs/history'/('quality-before-'+CRITICAL_VERSION)/folder.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                assert not dest.exists()
                shutil.move(str(folder),str(dest))
    qa=ROOT/'outputs/qa'
    if qa.exists():
        for folder in qa.iterdir():
            marker=folder/'complete.json'
            if folder.is_dir() and (not marker.exists() or load(marker).get('critical_values_version')!=CRITICAL_VERSION):
                dest=ROOT/'outputs/history'/('qa-before-'+CRITICAL_VERSION)/folder.name
                dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
                shutil.move(str(folder),str(dest))
    prepare()
    write(ROOT/'outputs/provision_complete.json',dict(complete=True,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
