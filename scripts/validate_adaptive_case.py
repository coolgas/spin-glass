"""Bounded single-worker L6 D128 checkpoint/resume validation, not production."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-cache',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve(); root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    env=os.environ.copy(); env['PYTHONPATH']=str(repo/'code'); env['PYTHONDONTWRITEBYTECODE']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    args=['case','--adaptive','--no-exact','--output',str(root/'case'),'--L','6','--sample','0',
        '--model','full','--delta','3.15','--omega','0.9037538790536807','--chi','12',
        '--dimensions','128','256','384','512','--pairs','20000','--seconds','360',
        '--invocation-sweeps','1','--memory-gib','8','--model-cache',str(a.model_cache)]
    bootstrap='import resource,sys; resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30)); from rydberg_kagome.__main__ import main; main()'
    report=dict(production_campaign=False,controller=os.getpid(),started=time.time(),attempts=[])
    def save():
        tmp=root/'validation.tmp'; tmp.write_text(json.dumps(report,indent=2)+'\n'); os.replace(tmp,root/'validation.json')
    for attempt in (1,2):
        with (root/f'invocation{attempt}.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B','-u','-c',bootstrap,*args],cwd=repo,env=env,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        row=dict(attempt=attempt,pid=child.pid,state='running'); report['attempts'].append(row); save()
        try: child.wait(timeout=420)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
            row['state']='hard_time_limit'
        else: row['state']='finished' if child.returncode==0 else 'failed'
        row['exit_code']=child.returncode
        resume=root/'case/resume.json'
        if resume.exists():
            d=json.loads(resume.read_text()); row['generation']=d['generation']
            row['sweeps']=[len(h) for h in d['state']['history']]; row['D']=d['state']['D']
        save()
        if child.returncode or not resume.exists(): break
    report['finished']=True; report['seconds']=time.time()-report['started']
    report['resume_demonstrated']=(len(report['attempts'])==2 and
        report['attempts'][0].get('sweeps')==[1,0] and report['attempts'][1].get('sweeps')==[1,1])
    save(); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
