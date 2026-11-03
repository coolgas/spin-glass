"""Bounded short-SSH monitor, validated download, and automatic notebook refresh."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--remote',required=True);p.add_argument('--hours',type=float,default=6.)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    destination=root/'data/kagome/ltfim_correlation_expanded_v1';destination.mkdir(parents=True,exist_ok=True)
    code="""
from pathlib import Path
import json
root=Path(REMOTE)
status={}
for stage in ['guides','production','audit']:
 p=root/'results'/(stage+'_status.json')
 if p.exists():status[stage]=json.loads(p.read_text())
p=root/'controller_status.json'
if p.exists():status['controller']=json.loads(p.read_text())
pid=json.loads((root/'launch.json').read_text())['pid']
p=Path('/proc')/str(pid)/'cmdline'
status['alive']=p.exists() and b'controller.py' in p.read_bytes()
print(json.dumps(status))
""".replace('REMOTE',repr(a.remote))
    deadline=time.monotonic()+a.hours*3600;errors=0;previous=None
    while time.monotonic()<deadline:
        try:
            r=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=10',
                '-o','ServerAliveCountMax=2','autodl','/root/miniconda3/bin/python -'],input=code,text=True,capture_output=True,timeout=45)
            if r.returncode:raise RuntimeError(r.stderr)
            status=json.loads(r.stdout);errors=0
        except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
            errors+=1;print(f'SSH monitor failure {errors}/3: {error}',flush=True)
            if errors>=3:raise
            time.sleep(30);continue
        (destination/'remote_status.json').write_text(json.dumps(status,indent=2)+'\n')
        if status!=previous:print(json.dumps(status),flush=True);previous=status
        controller=status.get('controller')
        if controller:
            subprocess.run(['rsync','-az','--exclude','*.tmp*','-e',
                'sshpass -e ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4',
                f'autodl:{a.remote}/results/',str(destination)+'/'],check=True,timeout=3600)
            if controller['state']!='completed':raise RuntimeError('Remote campaign failed; partial data downloaded')
            from transformer_qmc.assess_expanded_correlation import assess
            report=assess(destination,require_complete=True)
            subprocess.run([sys.executable,str(root/'code/transformer_qmc/report_correlation_lengths.py')],cwd=root,check=True,timeout=900)
            print(json.dumps(dict(state='downloaded_validated_notebook_updated',runs=report['completed_runs'],
                                  all_checks_pass=report['all_checks_pass'])),flush=True)
            return
        if not status['alive']:raise RuntimeError('Campaign controller exited without completion record')
        time.sleep(30)
    raise TimeoutError('Monitor deadline reached; remote campaign left running')


if __name__=='__main__':main()
