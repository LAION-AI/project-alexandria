"""Keep package-version discovery in node RAM too, with unchanged packages."""
import os,tarfile,json
from pathlib import Path
from common import ROOT,write
SITE=Path('/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/lib/python3.11/site-packages')
def main():
 archive_path=ROOT/'inputs/python_import_cache.tar'
 with tarfile.open(archive_path,'r') as source:roots={m.name.split('/')[0] for m in source.getmembers()}
 counts={'metadata_files':0,'root_links':0}
 with tarfile.open(archive_path,'a') as target:
  for item in sorted(SITE.iterdir()):
   if item.name.endswith('.dist-info') and item.is_dir():
    for file in item.iterdir():
     if file.is_file():target.add(file,arcname=item.name+'/'+file.name,recursive=False);counts['metadata_files']+=1
   elif item.name not in roots and item.name!='sitecustomize.py':
    info=tarfile.TarInfo(item.name);info.type=tarfile.SYMTYPE;info.linkname=str(item);target.addfile(info);counts['root_links']+=1
  # Both venv spellings refer to the same immutable original environment. Keep
  # all package roots available through local copies or explicit root symlinks,
  # then remove the duplicate shared package scan from sys.path.
  aliases=[str(SITE),str(SITE.resolve())]
  text='import sys\nSOURCE_SITE_ALIASES='+repr(aliases)+'\nsys.path[:]=[p for p in sys.path if p not in SOURCE_SITE_ALIASES]\n'
  original=SITE/'sitecustomize.py'
  if original.exists():text+='import runpy\nrunpy.run_path('+repr(str(original))+')\n'
  import io
  blob=text.encode();info=tarfile.TarInfo('sitecustomize.py');info.size=len(blob);target.addfile(info,io.BytesIO(blob))
 write(ROOT/'outputs/import_metadata_cache.json',dict(complete=True,archive_bytes=archive_path.stat().st_size,counts=counts,removed_duplicate_shared_sites=aliases,
  purpose='Avoid repeated GPFS dist-info/package discovery during four simultaneous imports; model code and package versions unchanged.'))
 print('METADATA_CACHE_COMPLETE',counts,flush=True)
if __name__=='__main__':main()
