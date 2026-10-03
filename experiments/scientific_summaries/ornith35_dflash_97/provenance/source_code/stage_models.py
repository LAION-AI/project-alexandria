import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', '1')

def stage(name):
    identity = json.loads((ROOT / 'inputs' / (name + '-hf-api.json')).read_text())
    directory = ROOT / 'models' / {'target': 'Ornith-1.5-35B-A3B', 'draft': 'Ornith-1.5-35B-A3B-DFlash', 'judge': 'Qwen2.5-7B-Instruct'}[name]
    revision = identity['sha']
    info = HfApi().model_info(identity['id'], revision=revision, files_metadata=True, token=False)
    if info.sha != revision:
        raise ValueError('Revision changed')
    snapshot_download(identity['id'], revision=revision, local_dir=str(directory), token=False, max_workers=6)
    files = {}
    for entry in info.siblings:
        path = directory / entry.rfilename
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        sha = digest.hexdigest()
        if entry.lfs and sha != entry.lfs.sha256:
            raise ValueError('Weight checksum mismatch: ' + entry.rfilename)
        files[entry.rfilename] = dict(bytes=path.stat().st_size, sha256=sha)
    manifest = dict(model=identity['id'], revision=revision, files=files)
    (ROOT / 'inputs' / (name + '-weights-manifest.json')).write_text(json.dumps(manifest, indent=2))
    print('STAGED', name, revision, sum(x['bytes'] for x in files.values()), flush=True)
    return manifest

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(stage, ('target', 'draft')))
    (ROOT / 'inputs/models_ready.json').write_text(json.dumps(dict(models=results), indent=2))
