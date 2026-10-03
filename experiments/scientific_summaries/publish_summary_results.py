"""Publish audited completed model results, without staging unrelated work or secrets."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from evaluate import ROOT, load
from project_alexandria.io import write_json_atomic
from queue_control import AlreadyRunning, exclusive_lock

REPO = ROOT.parents[1]
SECRET = re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def git(arguments):
    result = subprocess.run(['git'] + arguments, cwd=str(REPO), capture_output=True, text=True)
    if result.returncode:
        message = SECRET.sub(b'[REDACTED]', (result.stdout + result.stderr).encode()).decode()
        message = re.sub(r'(https?://)[^/@\s]+@', r'\1[REDACTED]@', message)
        raise RuntimeError('Git publication failed: ' + message[:1200])
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    ledger_path = ROOT / 'published_summary_results.json'
    ledger = load(ledger_path) if ledger_path.exists() else {'models': []}
    while True:
        report_path = ROOT / 'summary_comparison.json'
        report = load(report_path) if report_path.exists() else {'models': {}, 'complete': False}
        names = sorted(report['models'])
        if set(names) - set(ledger['models']):
            subprocess.run([sys.executable, str(ROOT / 'summary_comparison_report.py')], check=True)
            paths = [ROOT / name for name in ('summary_comparison.html', 'summary_comparison.json',
                'summary_comparison.csv', 'summary_comparison.SHA256SUMS', 'summary_comparison_manifest.json')]
            for name in names:
                directory = ROOT / 'summary_runs' / name
                paths += [directory / item for item in ('summaries.json', 'results.json',
                    'qa_evaluated.json', 'summary_prompt_snapshot.json', 'documents')]
                for item in ('failure_journals', 'completion_schema_snapshot.json'):
                    if (directory / item).exists():
                        paths.append(directory / item)
                cache = load(directory / 'summaries.json')
                for record in cache['documents'] + cache.get('failures', []):
                    if record.get('raw_journal_file'):
                        journal = (directory / record['raw_journal_file']).resolve()
                        journal.relative_to(directory.resolve())
                        if hashlib.sha256(journal.read_bytes()).hexdigest() != record['raw_journal_sha256']:
                            raise ValueError('Raw journal hash changed; publication stopped')
            for path in paths:
                files = path.rglob('*') if path.is_dir() else [path]
                for file in files:
                    if file.is_file() and SECRET.search(file.read_bytes()):
                        raise ValueError('Potential secret; publication stopped, file: ' + str(file.relative_to(REPO)))
                    if file.is_file() and file.stat().st_size > 100 * 1024 * 1024:
                        raise ValueError('File exceeds GitHub limit; preserve raw calls in smaller shards: '
                                         + str(file.relative_to(REPO)))
            targets = [str(path.relative_to(REPO)) for path in paths]
            git(['add', '--'] + targets)
            changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--'] + targets, cwd=str(REPO))
            if changed.returncode == 1:
                git(['commit', '--only', '-m', 'Publish audited scientific summary results: ' + ', '.join(names), '--'] + targets)
            elif changed.returncode != 0:
                raise RuntimeError('Could not inspect staged result changes')
            git(['push', 'origin', 'main'])
            ledger = {'models': names, 'commit': git(['rev-parse', 'HEAD'])}
            write_json_atomic(str(ledger_path), ledger)
            print('PUBLISHED_SUMMARY_RESULTS', names, ledger['commit'][:12], flush=True)
        phase_path = ROOT / 'summary_comparison_status.json'
        phase = load(phase_path)['phase'] if phase_path.exists() else None
        if not args.watch or report.get('complete') or phase in ('failed', 'complete'):
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        with exclusive_lock(ROOT / '.summary_publisher.lock'):
            main()
    except AlreadyRunning as error:
        print(str(error), flush=True)
        raise SystemExit(2)
