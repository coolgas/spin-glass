"""Monitor a detached symmetric run, retrieve artifacts, audit, execute report."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--local',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SSHPASS'):
        raise RuntimeError('SSHPASS must be set')
    root=Path(__file__).resolve().parents[2]
    args.local.mkdir(parents=True,exist_ok=True)
    lock=(args.local/'monitor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment=dict(os.environ,PYTHONPATH=str(root/'code'))
    iteration=0
    while True:
        program=f'''from pathlib import Path
import json,hashlib
r=Path({args.remote!r})
p=Path('/proc')/(r/'train.pid').read_text().strip()/'stat'
s=json.loads((r/'results/status.json').read_text())
digest=None
if s['state']=='completed':
 evaluation=json.loads((r/'results/evaluation.json').read_text())
 digest=hashlib.sha256((r/'results'/evaluation['checkpoint']).read_bytes()).hexdigest()
print(json.dumps(dict(status=s,alive=p.exists() and p.read_text().split()[2]!='Z',sha256=digest)))
'''
        check=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl',
                              '/root/miniconda3/envs/yjye_project_env/bin/python -B -'],
                             input=program,text=True,capture_output=True,timeout=40)
        if check.returncode:
            print(check.stdout,check.stderr,flush=True)
            raise RuntimeError('Remote status failed; training remains detached')
        state=json.loads(check.stdout)
        complete=state['status']['state']=='completed'
        failed=not state['alive'] and not complete
        if iteration%5==0 or complete or failed:
            print('STATUS',state,flush=True)
            subprocess.run(['rsync','-az','-e','sshpass -e ssh -o ConnectTimeout=15',
                            f'autodl:{args.remote}/results/',str(args.local)+'/'],check=True,timeout=180)
            subprocess.run(['rsync','-az','-e','sshpass -e ssh -o ConnectTimeout=15',
                            f'autodl:{args.remote}/train.log',str(args.local/'train.log')],check=True,timeout=90)
            if complete:
                evaluation=json.loads((args.local/'evaluation.json').read_text())
                checkpoint=args.local/evaluation['checkpoint']
                if not (hashlib.sha256(checkpoint.read_bytes()).hexdigest()==evaluation['sha256']==state['sha256']):
                    raise RuntimeError('Final checkpoint remote/local/evaluation hash mismatch')
                subprocess.run([sys.executable,'-B','-m','transformer_qmc.audit_symmetric',
                                '--results',str(args.local)],cwd=root,env=environment,check=True,timeout=180)
                for module,filename in (
                    ('diagnose_low_field','low_field_spectrum_final.json'),
                    ('diagnose_projection','exact_projection_final.json'),
                    ('compare_low_field','correlation_comparison.json')):
                    command=[sys.executable,'-B','-m','transformer_qmc.'+module,
                             '--checkpoint',str(checkpoint),'--output',str(args.local/filename)]
                    if module=='compare_low_field': command.append('--field-scan')
                    subprocess.run(command,cwd=root,env=environment,check=True,timeout=240)
            subprocess.run([sys.executable,'-B','-m','transformer_qmc.report_symmetric',
                            '--results',str(args.local),'--notebook',str(args.notebook)],
                           cwd=root,env=environment,check=True,timeout=180)
        if failed:
            print((args.local/'train.log').read_text()[-8000:],flush=True)
            raise RuntimeError('Training exited before completion')
        if complete:
            print('Completed: downloaded, checkpoint verified, symmetry audited, notebook executed.',flush=True)
            break
        iteration+=1
        time.sleep(60)


if __name__=='__main__':
    main()
