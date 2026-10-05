"""Download only pinned weights/tokenizers; prepare source-only frozen inputs."""
import importlib.metadata
import platform
import time
from common import ROOT,load,write
from prepare import main as prepare


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
    prepare()
    write(ROOT/'outputs/provision_complete.json',dict(complete=True,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':main()
