"""Owned, bounded supervisor for exactly two non-production pilot probes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n'); os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-case',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['reduced','target'],default=['reduced'])
    p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args(); root=a.root.resolve(); root.mkdir(parents=True,exist_ok=False)
    if not 120<=a.seconds<=900 or len(a.modes)!=len(set(a.modes)):
        raise ValueError('Require 120..900 seconds and unique probe modes')
    started=time.monotonic(); jobs={}; state=dict(controller=os.getpid(),owner='adaptive-d-demo-20261123',
        production_campaign=False,hard_seconds=a.seconds,cases={})
    for mode in a.modes:
        env=os.environ.copy()
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name]='1'
        env['PYTHONDONTWRITEBYTECODE']='1'
        cmd=[sys.executable,'-B','-u',str(Path(__file__).with_name('demo_adaptive_d.py')),
            '--output',str(root/mode),'--source-case',str(a.source_case),'--mode',mode,'--seconds',str(a.seconds)]
        with (root/f'{mode}.log').open('xb') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        jobs[mode]=child
        birth=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]
        state['cases'][mode]=dict(pid=child.pid,birth=birth,state='running',command=cmd)
    while any(p.poll() is None for p in jobs.values()):
        elapsed=time.monotonic()-started
        for mode,child in jobs.items():
            if child.poll() is None and elapsed>=a.seconds:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL); child.wait()
                state['cases'][mode]['state']='hard_time_limit'
            elif child.poll() is not None and state['cases'][mode]['state']=='running':
                state['cases'][mode]['state']='finished' if child.returncode==0 else 'failed'
            if child.poll() is not None: state['cases'][mode]['exit_code']=child.returncode
        state.update(updated=time.time(),elapsed_seconds=elapsed)
        write(root/'status.json',state)
        if any(p.poll() is None for p in jobs.values()): time.sleep(5)
    for mode,child in jobs.items():
        state['cases'][mode]['exit_code']=child.returncode
        if state['cases'][mode]['state']=='running':
            state['cases'][mode]['state']='finished' if child.returncode==0 else 'failed'
        # Terminal sidecar supersedes a stale last-phase worker status after a signal.
        write(root/f'{mode}-termination.json',dict(state=state['cases'][mode]['state'],
              exit_code=child.returncode,updated=time.time(),result_exists=(root/mode/'result.json').exists()))
    state.update(updated=time.time(),finished=True,elapsed_seconds=time.monotonic()-started)
    write(root/'status.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__': main()
