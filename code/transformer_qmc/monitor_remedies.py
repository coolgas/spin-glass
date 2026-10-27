"""Retrieve detached remedy runs and refresh their executed comparison report."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--local', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--wait-for-launches', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SSHPASS'):
        raise RuntimeError('SSHPASS must be set')
    root = Path(__file__).resolve().parents[2]
    args.local.mkdir(parents=True, exist_ok=True)
    lock = (args.local/'monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, PYTHONPATH=str(root/'code'))
    previous = None
    while True:
        script = f'''from pathlib import Path
import json,hashlib
r=Path({args.remote!r})
launches=json.loads((r/'launches.json').read_text()) if (r/'launches.json').exists() else {{'runs':[]}}
if not launches['runs'] and (r.parent/'supervisor.pid').exists():
 supervisor=Path('/proc')/(r.parent/'supervisor.pid').read_text().strip()/'stat'
 if not supervisor.exists() or supervisor.read_text().split()[2]=='Z':
  print((r.parent/'supervisor.log').read_text()[-12000:])
  raise RuntimeError('Overnight supervisor stopped before launching continuations')
rows=[]
for run in launches['runs']:
 p=Path('/proc')/str(run['pid'])/'stat'
 status=r/run['name']/'status.json'
 state=json.loads(status.read_text()) if status.exists() else {{'state':'starting'}}
 digest=None
 if state['state']=='completed':
  digest=hashlib.sha256((r/run['name']/state['checkpoint']).read_bytes()).hexdigest()
 rows.append(dict(name=run['name'],alive=p.exists() and p.read_text().split()[2]!='Z',
                  status=state,sha256=digest))
print(json.dumps(rows))
'''
        check = subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15','autodl',
                                '/root/miniconda3/envs/yjye_project_env/bin/python -B -'],
                               input=script, text=True, capture_output=True, timeout=40)
        if check.returncode:
            print(check.stdout, check.stderr, flush=True)
            raise RuntimeError('Remote status failed; detached training is unaffected')
        rows = json.loads(check.stdout)
        if not rows:
            if not args.wait_for_launches:
                raise RuntimeError('No run launches found')
            time.sleep(60)
            continue
        complete = all(r['status']['state']=='completed' for r in rows)
        failed = [r for r in rows if not r['alive'] and r['status']['state']!='completed']
        signature = [(r['name'], r['status']) for r in rows]
        if signature != previous or failed or complete:
            print('STATUS', json.dumps(rows), flush=True)
            subprocess.run(['rsync','-az','--exclude','code/','--exclude','*.lock','--exclude','*.tmp',
                            '-e','sshpass -e ssh -o ConnectTimeout=15',
                            f'autodl:{args.remote}/',str(args.local)+'/'], check=True, timeout=240)
            for row in rows:
                if row['sha256']:
                    checkpoint = args.local/row['name']/row['status']['checkpoint']
                    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row['sha256']:
                        raise RuntimeError(f'Checkpoint transfer hash mismatch: {checkpoint}')
                    from .experiment import write_json
                    write_json(args.local/row['name']/'transfer_verification.json',
                               dict(checkpoint=checkpoint.name, sha256=row['sha256'], passed=True))
                    audit=args.local/row['name']/'symmetry_audit.json'
                    if not audit.exists():
                        subprocess.run([sys.executable,'-B','-m','transformer_qmc.audit_symmetric',
                                        '--results',str(checkpoint.parent),'--checkpoint',str(checkpoint)],
                                       cwd=root, env=env, check=True, timeout=240)
                    spectrum=args.local/row['name']/'low_field_spectrum_final.json'
                    if not spectrum.exists():
                        subprocess.run([sys.executable,'-B','-m','transformer_qmc.diagnose_low_field',
                                        '--checkpoint',str(checkpoint),'--output',str(spectrum)],
                                        cwd=root, env=env, check=True, timeout=240)
                    projection=args.local/row['name']/'exact_projection_final.json'
                    if not projection.exists():
                        subprocess.run([sys.executable,'-B','-m','transformer_qmc.diagnose_projection',
                                        '--checkpoint',str(checkpoint),'--output',str(projection)],
                                       cwd=root, env=env, check=True, timeout=240)
            if complete:
                command=[sys.executable,'-B','-m','transformer_qmc.compare_low_field',
                         '--output',str(args.local/'correlation_comparison.json'),'--field-scan']
                for row in rows:
                    command += ['--checkpoint',str(args.local/row['name']/row['status']['checkpoint'])]
                subprocess.run(command,cwd=root,env=env,check=True,timeout=600)
            if list(args.local.glob('*/campaign.json')):
                subprocess.run([sys.executable,'-B','-m','transformer_qmc.report_remedies',
                                '--results',str(args.local),'--notebook',str(args.notebook)],
                               cwd=root, env=env, check=True, timeout=240)
            previous = signature
        if failed:
            for row in failed:
                print((args.local/(row['name']+'.log')).read_text()[-12000:], flush=True)
            raise RuntimeError('One or more training runs exited before completion')
        if complete:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
