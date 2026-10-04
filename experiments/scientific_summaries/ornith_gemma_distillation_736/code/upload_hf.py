"""Publish the validated frozen release using a Hugging Face login, never a GitHub token."""
import datetime
import json
import os
import socket
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download
from common import ROOT, load, write, digest


def main():
    token=get_token()
    if not token:
        write(ROOT/'outputs/hf_upload.json',dict(status='awaiting_hf_login',credentials_saved=False))
        return
    ready=load(ROOT/'outputs/release_ready.json')
    release=ROOT/'release'
    assert ready['papers']==ready['trace_archives']==736
    assert ready['checksums_sha256']==digest(release/'SHA256SUMS')
    expected_files={}
    for line in (release/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        path=(release/name).resolve()
        assert path.is_relative_to(release) and path.is_file() and digest(path)==expected
        expected_files[name]=path.stat().st_size
    expected_files['SHA256SUMS']=(release/'SHA256SUMS').stat().st_size
    api=HfApi(token=token)
    identity=api.whoami()
    request=load(ROOT/'inputs/hf_target.json') if (ROOT/'inputs/hf_target.json').exists() else {}
    namespace=request.get('namespace',identity['name'])
    repo_id=namespace+'/scientific-summary-distillation-Ornith-1.5-9B-736-20261004'
    receipt=ROOT/'outputs/hf_upload.json'
    previous=load(receipt) if receipt.exists() else {}
    exists=api.repo_exists(repo_id,repo_type='dataset')
    if exists and previous.get('created_repo_id')!=repo_id:
        remote=Path(hf_hub_download(repo_id,'manifest.json',repo_type='dataset',token=token))
        assert digest(remote)==digest(release/'manifest.json'), 'Existing repository belongs to a different release'
    api.create_repo(repo_id,repo_type='dataset',private=False,exist_ok=True)
    state=dict(status='uploading',repo_id=repo_id,created_repo_id=repo_id,
               url='https://huggingface.co/datasets/'+repo_id,
               release_manifest_sha256=digest(release/'manifest.json'),credentials_saved=False,
               job_id=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),pid=os.getpid(),
               started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(receipt,state)
    api.upload_large_folder(repo_id,release,repo_type='dataset',num_workers=4,
                            ignore_patterns=['.cache/**','**/__pycache__/**','**/*.pyc','**/*.tmp'],
                            print_report=True,print_report_every=60)
    remote=Path(hf_hub_download(repo_id,'SHA256SUMS',repo_type='dataset',token=token,force_download=True))
    assert digest(remote)==ready['checksums_sha256']
    info=api.dataset_info(repo_id)
    remote_files={item.path:item.size for item in api.list_repo_tree(repo_id,repo_type='dataset',revision=info.sha,recursive=True)
                  if hasattr(item,'size')}
    assert all(remote_files.get(name)==size for name,size in expected_files.items()), 'Remote release file missing or wrong size'
    state.update(status='complete',revision=info.sha,papers=736,verified_remote_files=len(expected_files),
                 completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(receipt,state)
    print(json.dumps({k:v for k,v in state.items() if k in ['status','url','revision','papers']}),flush=True)


if __name__=='__main__':main()
