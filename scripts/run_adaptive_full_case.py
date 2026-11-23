"""One authorized L6 convergence test, with bounded resources and terminal status."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--model-cache',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not 1<=args.threads<=16: raise ValueError('Threads must be in 1..16')
    root=args.root.resolve()
    if not args.root.is_absolute() or any(x in root.parts for x in ('Dropbox','CloudStorage')):
        raise ValueError('Explicit non-synced output required')
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'code'))
    from rydberg_kagome.adaptive_memory import snapshot,GIB
    from rydberg_kagome.results import source_hashes
    if snapshot()['available_bytes']<80*GIB:
        raise RuntimeError('Full-case test needs 64-GiB cap plus 16-GiB live headroom')
    if shutil.disk_usage(root.parent).free<64*GIB:
        raise RuntimeError('Full-case test requires 64 GiB free storage')
    previous=None
    if args.resume:
        previous=json.loads((root/'status.json').read_text())
        if not previous.get('finished'): raise RuntimeError('Previous controller is not finished')
        if not (root/'case/resume.json').exists(): raise RuntimeError('No committed restart')
        if previous['source_hashes']!=source_hashes(): raise RuntimeError('Source mismatch')
        stamp=str(time.time_ns())
        for name in ('status.json','worker.log','worker-resources.json'):
            path=root/name
            if path.exists(): path.rename(root/f'{stamp}-{name}')
    else: root.mkdir(exist_ok=False)
    env=os.environ.copy(); env['PYTHONPATH']=str(repo/'code')
    env['PYTHONDONTWRITEBYTECODE']='1'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]=str(args.threads)
    command=['case','--adaptive','--no-exact','--output',str(root/'case'),
        '--L','6','--sample','0','--model','full','--delta','3.15',
        '--omega','0.9037538790536807','--chi','12',
        '--dimensions','128','256','384','512','--pairs','20000',
        '--max-sweeps','120','--memory-gib','64','--model-cache',str(args.model_cache)]
    bootstrap='''import json,resource,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(64*2**30,64*2**30))
from rydberg_kagome.__main__ import main
started=time.monotonic()
try:
    main()
finally:
    p=Path(sys.argv[sys.argv.index('--output')+1]).parent/'worker-resources.json'
    p.write_text(json.dumps(dict(seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\\n')
'''
    status=dict(controller=os.getpid(),started=time.time(),production_campaign=False,
        purpose='single_full_case_convergence_gate',hard_seconds=72*3600,
        memory_gib=64,threads=args.threads,source_hashes=source_hashes(),
        retention='Retain all immutable generations; no automatic deletion',finished=False)
    def save():
        tmp=root/'status.tmp'; tmp.write_text(json.dumps(status,indent=2)+'\n')
        os.replace(tmp,root/'status.json')
    with (root/'worker.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-c',bootstrap,*command],
            cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    status.update(worker=child.pid,state='running'); save()
    if previous:
        status['original_started']=previous.get('original_started',previous['started'])
        status['resumed_from_worker']=previous['worker']
    deadline=time.monotonic()+max(0,status['hard_seconds']-(time.time()-status.get('original_started',status['started'])))
    while child.poll() is None:
        if time.monotonic()>=deadline:
            status['state']='hard_time_limit'
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
            break
        progress=root/'case/adaptive-status.json'
        if progress.exists(): status['progress']=json.loads(progress.read_text())
        status['updated']=time.time(); save()
        try: child.wait(timeout=30)
        except subprocess.TimeoutExpired: pass
    result=root/'case/result.json'
    if result.exists():
        data=json.loads(result.read_text())
        status['flags']={k:data[k] for k in ('solver_pass','sampling_moment_check','bond_dimension_verified','rg_converged','production_qualified')}
        status['full_case_gate_passed']=bool(child.returncode==0 and data['solver_pass'] and data['sampling_moment_check'])
        status['state']='converged' if status['full_case_gate_passed'] else 'completed_gate_failed'
    else:
        status['full_case_gate_passed']=False
        if status['state']=='running': status['state']='incomplete' if child.returncode==0 else 'failed'
    status.update(finished=True,exit_code=child.returncode,updated=time.time(),seconds=time.time()-status['started'])
    progress=root/'case/adaptive-status.json'
    if progress.exists(): status['progress']=json.loads(progress.read_text())
    save(); print(json.dumps(status),flush=True)


if __name__=='__main__': main()
