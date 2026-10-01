"""Fingerprint the existing runtimes and requested weights without loading GPU models."""
import hashlib
import subprocess
from pathlib import Path

from evaluate import ROOT
from run_summary_comparison import MODELS
from project_alexandria.io import write_json_atomic


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    binary = Path('/home/c4r33u19/moss15v2/llama.cpp/build/bin/llama-server')
    version = subprocess.check_output([str(binary), '--version'], stderr=subprocess.STDOUT, text=True).strip()
    vllm = subprocess.check_output(['/home/rvv/llm-inference/.venv27/bin/python', '-c',
        'import importlib.metadata; print(importlib.metadata.version("vllm"))'], text=True).strip()
    models = []
    for model in MODELS[1:]:
        path = Path('/home/c4r33u19/models/scientific-summary-gguf') / model['filename']
        value = sha256(path)
        if value != model['weights_sha256']:
            raise ValueError('GGUF checksum mismatch: ' + model['name'])
        models.append(dict(model=model['model'], revision=model['revision'], file_name=path.name,
                           sha256=value, bytes=path.stat().st_size))
    write_json_atomic(str(ROOT / 'summary_runtime_fingerprints.json'), dict(
        llama_cpp_version=version, llama_server_binary_sha256=sha256(binary), vllm_version=vllm,
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total',
            '--format=csv,noheader'], text=True).splitlines(), models=models))
    print('SUMMARY_RUNTIME_FINGERPRINTS_VERIFIED', len(models), flush=True)


if __name__ == '__main__':
    main()
