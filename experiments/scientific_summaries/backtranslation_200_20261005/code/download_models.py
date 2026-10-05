"""Stage pinned model files on a login node; never load a model here."""
import time
from huggingface_hub import snapshot_download
from common import ROOT, load, write

def main():
    started = time.monotonic()
    for label, metadata in load(ROOT / 'inputs/model_pins.json').items():
        print('Downloading', label, metadata['revision'], flush=True)
        snapshot_download(metadata['model_id'], revision=metadata['revision'],
                          local_dir=ROOT / 'models' / label, max_workers=2,
                          allow_patterns=['*.safetensors', '*.json', '*.spm', '*.model', '*.jinja'])
        print('Staged', label, flush=True)
    write(ROOT / 'inputs/models_staged.json', {'complete': True, 'seconds': time.monotonic() - started})

if __name__ == '__main__':
    main()
