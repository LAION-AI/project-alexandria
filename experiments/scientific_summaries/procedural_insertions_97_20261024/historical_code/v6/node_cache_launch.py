"""Launch with an automatically managed, job-owned node-memory import cache."""
import os,signal,subprocess,sys,tempfile
from common import ROOT
def interrupted(signum,frame):raise InterruptedError('Allocation signal '+str(signum))
def main():
 signal.signal(signal.SIGTERM,interrupted)
 with tempfile.TemporaryDirectory(prefix='alexandria-hybrid-imports-'+os.environ['SLURM_JOB_ID']+'-',dir='/dev/shm') as cache:
  subprocess.run(['tar','-xf',str(ROOT/'inputs/python_import_cache.tar'),'-C',cache],check=True)
  env=dict(os.environ);env['PYTHONPATH']=cache+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
  interpreter=sys.argv[1] if len(sys.argv)>1 else sys.executable
  subprocess.run([interpreter,'-u',str(ROOT/'code/supervisor.py')],env=env,check=True)
if __name__=='__main__':main()
