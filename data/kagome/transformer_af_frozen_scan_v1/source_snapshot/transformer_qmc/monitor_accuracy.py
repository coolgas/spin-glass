"""Retrieve extended AutoDL results and refresh reports through campaign completion.

Run detached with SSHPASS in the environment; credentials are never printed.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REMOTE='/root/autodl-tmp/transformer-af-ltfim-20261026-alibi-accuracy-v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    if not os.environ.get('SSHPASS'):
        raise RuntimeError('SSHPASS must be provided through the environment')
    root=Path(__file__).resolve().parents[2]
    local=root/'data/kagome/transformer_af_alibi_accuracy_v1'
    local.mkdir(parents=True,exist_ok=True)
    environment=dict(os.environ,PYTHONPATH=str(root/'code'))
    iteration=0
    while True:
        # Inspect actual PID as well as progress metadata, so a crash cannot
        # leave this monitor waiting forever on a stale running status.
        program=f'''import json,os
from pathlib import Path
root=Path({REMOTE!r})
status=json.loads((root/'results/status.json').read_text())
pid=int((root/'train.pid').read_text())
try:
 os.kill(pid,0)
 alive=Path('/proc/'+str(pid)+'/stat').read_text().split()[2]!='Z'
except (ProcessLookupError,FileNotFoundError):
 alive=False
print(json.dumps(dict(status=status,alive=alive)))
'''
        check=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl',
                              '/root/miniconda3/envs/yjye_project_env/bin/python -B -'],
                             input=program,text=True,capture_output=True,timeout=40)
        if check.returncode:
            print(check.stdout,check.stderr,flush=True)
            raise RuntimeError('Remote monitor failed; no automatic retry loop')
        state=json.loads(check.stdout)
        complete=state['status']['state']=='completed'
        failed=not state['alive'] and not complete
        if iteration%10==0 or complete or failed or not args.watch:
            print('REMOTE',state,flush=True)
            subprocess.run(['rsync','-az','-e','sshpass -e ssh -o ConnectTimeout=15',
                            f'autodl:{REMOTE}/results/',str(local)+'/'],check=True,timeout=180)
            subprocess.run(['rsync','-az','-e','sshpass -e ssh -o ConnectTimeout=15',
                            f'autodl:{REMOTE}/train.log',str(local/'train.log')],check=True,timeout=90)
            subprocess.run([sys.executable,'-B','-m','transformer_qmc.report_accuracy'],
                           cwd=root,env=environment,check=True,timeout=180)
        if failed:
            print((local/'train.log').read_text()[-12000:],flush=True)
            raise RuntimeError('Training process exited before completion; checkpoints retained')
        if complete:
            subprocess.run([sys.executable,'-B','-m','transformer_qmc.qualify_accuracy'],
                           cwd=root,env=environment,check=True,timeout=180)
            print('Completed: artifacts retrieved, executed notebook refreshed, qualification saved.',flush=True)
            break
        if not args.watch:
            break
        iteration+=1
        time.sleep(60)


if __name__=='__main__':
    main()
