"""Start the resumable queue detached from terminal/tool session lifetime."""
import argparse
import subprocess
import sys
import time

from evaluate import ROOT
from run_summary_comparison import MODELS


def launch(script, arguments, log_name):
    with (ROOT / log_name).open('ab') as log:
        process = subprocess.Popen([sys.executable, str(ROOT / script)] + arguments,
            cwd=str(ROOT.parents[1]), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True, close_fds=True)
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError(f'{script} exited during launch; inspect {log_name}')
    print(f'STARTED {script} pid={process.pid} log={log_name}', flush=True)
    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-qwen-server-pid', type=int)
    parser.add_argument('--only-model', action='append', choices=[m['name'] for m in MODELS])
    parser.add_argument('--publish', action='store_true',
                        help='Publish audited results; use only with user authorization')
    args = parser.parse_args()
    forwarded = (['--reuse-qwen-server-pid', str(args.reuse_qwen_server_pid)]
                 if args.reuse_qwen_server_pid else [])
    forwarded += [argument for model in args.only_model or [] for argument in ('--only-model', model)]
    launch('run_summary_comparison.py', forwarded, 'summary_queue.log')
    if args.publish:
        launch('publish_summary_results.py', ['--watch'], 'summary_publication.log')


if __name__ == '__main__':
    main()
