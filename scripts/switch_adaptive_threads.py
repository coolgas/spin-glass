"""Owned checkpoint-boundary thread benchmark and full-case resume."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    root=Path(sys.argv[1]); repo=Path(__file__).resolve().parents[1]
    case=root/'case'; initial=json.loads((case/'resume.json').read_text())
    status=json.loads((root/'status.json').read_text())
    worker=status['worker']; controller=status['controller']
    cmd=Path(f'/proc/{worker}/cmdline').read_bytes()
    if str(case).encode() not in cmd or b'--adaptive' not in cmd: raise RuntimeError('Worker identity mismatch')
    out=root/'thread-benchmark'; out.mkdir(exist_ok=False)
    report=dict(owner=os.getpid(),started=time.time(),old_worker=worker,old_controller=controller,state='awaiting_checkpoint')
    def save():
        tmp=out/'report.tmp'; tmp.write_text(json.dumps(report,indent=2)+'\n'); os.replace(tmp,out/'report.json')
    save(); deadline=time.monotonic()+600
    while json.loads((case/'resume.json').read_text())['generation']==initial['generation']:
        if time.monotonic()>deadline or not Path(f'/proc/{worker}').exists(): raise RuntimeError('No next checkpoint within switch window')
        time.sleep(1)
    # Only the owned worker is signalled, after a newly committed complete pair.
    # Any fraction of the following sweep is replayed from this checkpoint.
    if Path(f'/proc/{worker}/cmdline').read_bytes()!=cmd: raise RuntimeError('Worker identity changed')
    os.kill(worker,signal.SIGTERM)
    deadline=time.monotonic()+60
    while not json.loads((root/'status.json').read_text()).get('finished'):
        if time.monotonic()>deadline: raise RuntimeError('Old controller did not finish')
        time.sleep(1)
    record=json.loads((case/'resume.json').read_text())
    report.update(state='benchmarking',generation=record['generation'],counts=[len(h) for h in record['state']['history']],cases=[]); save()
    from importlib import import_module
    sys.path.insert(0,str(repo/'code'))
    memory=import_module('rydberg_kagome.adaptive_memory')
    if memory.snapshot()['available_bytes']<80*memory.GIB: raise RuntimeError('Insufficient headroom for four 16-GiB probes')
    processes=[]
    for threads in (1,2,4,8):
        env=os.environ.copy()
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): env[name]=str(threads)
        with (out/f't{threads}.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B',str(repo/'scripts/benchmark_adaptive_threads.py'),
                '--case',str(case),'--output',str(out/f't{threads}.json')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((threads,child)); report['cases'].append(dict(threads=threads,pid=child.pid))
    save(); deadline=time.monotonic()+900
    while any(p.poll() is None for _,p in processes) and time.monotonic()<deadline: time.sleep(5)
    rows=[]
    for threads,child in processes:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        path=out/f't{threads}.json'
        row=dict(threads=threads,exit_code=child.returncode)
        if child.returncode==0 and path.exists(): row.update(json.loads(path.read_text()))
        rows.append(row)
    good=[r for r in rows if 'energy' in r]
    if not good or max(r['energy'] for r in good)-min(r['energy'] for r in good)>108e-7:
        report.update(state='benchmark_failed',results=rows); save(); raise RuntimeError('Missing or inconsistent benchmark results')
    candidates=[r for r in good if r['threads']>1]
    if not candidates: report.update(state='benchmark_failed',results=rows); save(); raise RuntimeError('No successful multithreaded probe')
    selected=min(candidates,key=lambda r:r['seconds'])['threads']
    cache=json.loads((case/'cache-provenance.json').read_text())['source']
    with (root/'threaded-supervisor.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(repo/'scripts/run_adaptive_full_case.py'),
            '--root',str(root),'--model-cache',cache,'--resume','--threads',str(selected)],
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    report.update(state='resume_launched',results=rows,selected_threads=selected,new_controller=child.pid,finished=True)
    save(); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
