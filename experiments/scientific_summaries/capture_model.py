"""Record local judge-weight fingerprints without storing weights or machine credentials."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    fingerprints = {}
    names = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
             'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json']
    names += sorted(p.name for p in args.model_dir.glob('model-*.safetensors'))
    if not any(n.endswith('.safetensors') for n in names):
        raise SystemExit('No model shards found.')
    for name in names:
        path = args.model_dir / name
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        fingerprints[name] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    payload = {'model': 'Qwen/Qwen2.5-7B-Instruct', 'served_precision': 'BF16',
               'fingerprint_method': 'SHA256 of actual locally served files',
               'files': fingerprints}
    (ROOT / 'judge_model_fingerprints.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('Saved judge_model_fingerprints.json; no weights or credentials copied.')


if __name__ == '__main__':
    main()
