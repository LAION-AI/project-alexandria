"""Finalize a completed Slurm benchmark into the authorized repository."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/e/home/jusers/schuhmann1/jupiter/project-alexandria')
PACKAGE=REPO/'experiments/scientific_summaries/gemma4_97'
PRESERVED=Path('/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-gemma4-eval-97-20261003')

def run(argv,cwd=REPO):
    subprocess.run(argv,cwd=cwd,check=True)

def main():
    completion=json.loads((ROOT/'outputs/complete.json').read_text())
    job=completion['job_id']
    for _ in range(12):
        state=subprocess.check_output(['sacct','-X','-j',job,'--format=State','--noheader','--parsable2'],text=True).strip().strip('|')
        if state=='COMPLETED':break
        if state not in ('RUNNING','COMPLETING'):raise RuntimeError('Benchmark allocation failed: '+state)
        time.sleep(5)
    else:raise RuntimeError('Scheduler has not finalized the completed job')
    expected=json.loads((ROOT/'inputs/publication_inputs.json').read_text())
    for name,digest in expected.items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Repository document changed while benchmark was running: '+name)
    run(['python3',str(PACKAGE/'package_results.py'),'--run-dir',str(ROOT)])
    run(['python3',str(PACKAGE/'verify_results.py')])
    status_path=ROOT/'STATUS.json'
    status=json.loads(status_path.read_text())
    status.update(state='benchmark_complete',note='All Gemma QA and throughput conditions passed final verification.',publication='Preserving verified results and creating local commit')
    status_path.write_text(json.dumps(status,indent=2))
    shutil.copytree(ROOT,PRESERVED,dirs_exist_ok=True,symlinks=True,
        ignore=shutil.ignore_patterns('models','cache','__pycache__'))
    vendor=PRESERVED/'inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a'
    target=Path('/e/data1/datasets/playground/mmlaion/schuhmann1/scientific-ornith-dflash-eval-97-20261003/inputs/project-alexandria-5aac4b5ba2a78b20637e8ab960fe79d01ecd769a')
    if not vendor.is_symlink() or not target.is_dir():raise RuntimeError('Durable protocol snapshot unavailable')
    vendor.unlink();vendor.symlink_to(target,target_is_directory=True)
    for name in ['outputs/report.json','outputs/qa-results.json','outputs/final_audit.json','outputs/complete.json']:
        if hashlib.sha256((ROOT/name).read_bytes()).digest()!=hashlib.sha256((PRESERVED/name).read_bytes()).digest():
            raise RuntimeError('Preservation checksum mismatch')
    readme=REPO/'experiments/scientific_summaries/README.md'
    text=readme.read_text()
    header='## Completed Gemma 4 E4B IT / 12B IT results'
    if header not in text:
        report=json.loads((ROOT/'outputs/report.json').read_text())
        models=report['models']
        section=header+'\n\nThe [English Gemma findings/results](gemma4_97/README.md) cover both models on the same 97 papers / 970 MCQs, raw and corrected summaries, all 14 comparison scores, 48 batch throughput probes, paired confidence intervals, full traces and scheduler times. '
        section+='E4B scores **'+format(100*models['gemma4_e4b_raw']['accuracy'],'.2f')+'% raw** / **'+format(100*models['gemma4_e4b_corrected']['accuracy'],'.2f')+'% corrected**; '
        section+='12B scores **'+format(100*models['gemma4_12b_raw']['accuracy'],'.2f')+'% raw** / **'+format(100*models['gemma4_12b_corrected']['accuracy'],'.2f')+'% corrected**. See the [shared GH200 comparison](ORNITH_DFLASH_GH200_RESULTS.md).\n\n'
        first,rest=text.split('\n',1);readme.write_text('# Scientific summaries: frozen cohort and model comparisons\n\n'+section+rest.lstrip())
    root_readme=REPO/'README.md';text=root_readme.read_text()
    line='- [Gemma 4 E4B IT / 12B IT: 97-paper QA and GH200 throughput](experiments/scientific_summaries/gemma4_97/README.md)'
    if line not in text:root_readme.write_text(text.rstrip()+'\n\n'+line+'\n')
    run(['git','diff','--check'])
    relative=['experiments/scientific_summaries/gemma4_97','experiments/scientific_summaries/ORNITH_DFLASH_GH200_RESULTS.md','experiments/scientific_summaries/README.md','README.md']
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!='findings/ornith-dflash-97-20261003':raise RuntimeError('Working branch changed; results verified but not committed')
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
    if staged:raise RuntimeError('Existing staged changes; results verified but not committed')
    run(['git','add',*relative])
    run(['git','-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit','-m','Publish complete 97-paper Gemma 4 E4B and 12B QA and GH200 throughput'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    status=dict(state='complete_local',commit=commit,preserved_run=str(PRESERVED),report=str(PACKAGE/'README.md'),
        github_push='pending authentication; no token is stored or requested in this file')
    (ROOT/'outputs/publication_status.json').write_text(json.dumps(status,indent=2))
    top=json.loads((ROOT/'STATUS.json').read_text());top.update(state='complete_local',publication=status)
    (ROOT/'STATUS.json').write_text(json.dumps(top,indent=2))
    shutil.copyfile(ROOT/'STATUS.json',PRESERVED/'STATUS.json')
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
