"""Pack small Python import files once, for extraction into compute-node RAM.

Large native libraries and compiler headers remain symlinks to their unchanged
environment files. No package is installed or upgraded in the shared environment.
"""
import json,os,tarfile,time
from pathlib import Path
from common import ROOT,write
SITE=Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/lib/python3.11/site-packages')
STDLIB=Path('/e/home/jusers/schuhmann1/jupiter/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib/python3.11')
PACKAGES=['torch','transformers','vllm','sympy','networkx','numpy','triton','flashinfer','msgspec','huggingface_hub','pydantic','pydantic_core',
 'packaging','tokenizers','jinja2','requests','urllib3','tqdm','yaml','regex','safetensors','einops','psutil','sentencepiece','typing_extensions.py']

def main():
 target=ROOT/'inputs/python_import_cache.tar';temporary=target.with_suffix('.tar.part');started=time.monotonic();counts={'copied_files':0,'symlinks':0,'bytes':0}
 with tarfile.open(temporary,'w',dereference=False) as archive:
  def add(path,name):
   if path.is_symlink():
    info=tarfile.TarInfo(name);info.type=tarfile.SYMTYPE;info.linkname=str(path.resolve());archive.addfile(info);counts['symlinks']+=1
   elif path.is_dir():
    if path.name in {'include','csrc','test','tests','docs','benchmarks','site-packages','lib-dynload'}:
     info=tarfile.TarInfo(name);info.type=tarfile.SYMTYPE;info.linkname=str(path);archive.addfile(info);counts['symlinks']+=1
    else:
     for child in sorted(path.iterdir()):add(child,name+'/'+child.name)
   elif path.is_file():
    size=path.stat().st_size
    if path.suffix in {'.py','.pyc','.json','.jinja','.yaml','.yml','.txt'} and size<=4*1024*1024:
     archive.add(path,arcname=name,recursive=False);counts['copied_files']+=1;counts['bytes']+=size
    else:
     info=tarfile.TarInfo(name);info.type=tarfile.SYMTYPE;info.linkname=str(path);archive.addfile(info);counts['symlinks']+=1
  for name in PACKAGES:
   path=SITE/name
   if path.exists():add(path,name)
  # The interpreter and ABI remain unchanged; only pure stdlib import files are
  # cached. Do not shadow the environment's site-package discovery with a copied
  # site.py or sysconfig.py.
  for name in ['dataclasses.py','typing.py','inspect.py','ast.py','textwrap.py','contextlib.py','functools.py','enum.py','linecache.py',
   'collections','concurrent','asyncio','multiprocessing','json','logging','urllib','email','http','importlib']:
   path=STDLIB/name
   if path.exists():add(path,name)
 temporary.replace(target);write(ROOT/'outputs/python_import_cache.json',dict(complete=True,archive=str(target),archive_bytes=target.stat().st_size,
  seconds=time.monotonic()-started,source_environment=str(SITE),counts=counts,policy='Exact existing Python sources/bytecode; large libraries and headers symlink to original environment.'))
 print('IMPORT_CACHE_READY',counts,target.stat().st_size,round(time.monotonic()-started,2),flush=True)
if __name__=='__main__':main()
