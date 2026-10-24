"""Materialize hash-verified evidence, or inputs for a new inference run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true')
    parser.add_argument('--download-models', action='store_true')
    args = parser.parse_args()
    root = Path(os.environ.get('ALEXANDRIA_HYBRID_ROOT', HERE)).resolve()
    if root != HERE:
        for name in ['code', 'vendor']:
            shutil.copytree(HERE / name, root / name, dirs_exist_ok=True)
    manifest = json.loads((HERE / 'evidence_manifest.json').read_text())
    for bundle in manifest['bundles']:
        if args.inputs_only and bundle['group'] != 'inputs':
            continue
        path = HERE / bundle['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle['sha256'], path
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                target = (root / member.name).resolve()
                assert member.isfile() and target.is_relative_to(root), member.name
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == manifest['files'][member.name]['sha256']
                if target.exists():
                    assert target.read_bytes() == data, f'Refusing to overwrite changed file: {target}'
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
        print('Materialized', bundle['group'], flush=True)
    if not args.inputs_only:
        for path in (HERE / 'results').glob('*.json'):
            target = root / 'outputs' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == path.read_bytes(), target
            else:
                shutil.copy2(path, target)
    if args.download_models:
        from huggingface_hub import snapshot_download
        pins = json.loads((HERE / 'model_pins.json').read_text())
        for name, pin in pins.items():
            snapshot_download(pin['id'], revision=pin['revision'], local_dir=str(root / 'models' / name))
        output = root / 'outputs/model_download_complete.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'complete': True, 'pins': pins}, indent=2) + '\n')


if __name__ == '__main__':
    main()
