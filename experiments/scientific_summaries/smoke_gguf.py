"""CPU-only transport smoke: verify requested GGUFs without interrupting GPU jobs."""
import json
import os
import subprocess
import time
import urllib.request

from evaluate import ROOT
from run_summary_comparison import MODELS, stop
from summary_runtime import SummaryClient, parse_json
from project_alexandria.io import write_json_atomic


def main():
    binary = '/home/c4r33u19/moss15v2/llama.cpp/build/bin/llama-server'
    records = []
    for model in MODELS[1:]:
        command = [binary, '--model', '/home/c4r33u19/models/scientific-summary-gguf/' + model['filename'],
            '--alias', 'summary-model', '--host', '127.0.0.1', '--port', '8012',
            '--gpu-layers', '0', '--ctx-size', '8192', '--parallel', '1', '--threads', '8',
            '--threads-batch', '8', '--flash-attn', 'off', '--jinja', '--reasoning', 'off',
            '--chat-template-kwargs', '{"enable_thinking":false}']
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        with (ROOT / ('smoke_' + model['name'] + '.log')).open('a') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=log, start_new_session=True)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError('CPU GGUF startup failed: ' + model['name'])
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8012/v1/models', timeout=2) as response:
                            if any(item['id'] == 'summary-model' for item in json.load(response)['data']):
                                break
                    except (OSError, ValueError):
                        pass
                    time.sleep(2)
                else:
                    raise RuntimeError('CPU GGUF startup timed out')
                client = SummaryClient('http://127.0.0.1:8012/v1', 'summary-model', 'llama.cpp', 8192)
                record = client.generate('Return only JSON.', 'Return a JSON object with one key value and the integer 42.', 64, 42)
                value, _ = parse_json(record['response'])
                if value != {'value': 42}:
                    raise ValueError('GGUF JSON transport smoke returned unexpected content')
                records.append(dict(model=model['name'], passed=True, runtime='llama.cpp CPU-only',
                    tokenizer_preflight_tokens=record['input_tokens_preflight'], finish_reason=record['finish_reason'],
                    usage=record['usage'], elapsed_seconds=record['elapsed_seconds']))
                write_json_atomic(str(ROOT / 'gguf_smoke_results.json'), records)
                print('GGUF_SMOKE_PASSED', model['name'], flush=True)
            finally:
                stop(process)


if __name__ == '__main__':
    main()
