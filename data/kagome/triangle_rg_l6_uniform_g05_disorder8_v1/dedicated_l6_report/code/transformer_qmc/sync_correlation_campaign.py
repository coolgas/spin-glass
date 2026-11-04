"""Monitor AutoDL through short SSH connections and sync finished campaign data.

Requires SSHPASS in the environment; never prints or stores credentials.
Transient connection errors are logged and retried at most three times in a row.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


REMOTE = '/root/autodl-tmp/correlation-lengths-20261103-v3'
STATUS_CODE = '''
import json
from pathlib import Path
root=Path('/root/autodl-tmp/correlation-lengths-20261103-v3')
status=json.loads((root/'results/gfmc_status.json').read_text())
review=root/'review_status.json'
if review.exists():
    status['review']=json.loads(review.read_text())
pid=json.loads((root/'launch.json').read_text())['pid']
proc=Path(f'/proc/{pid}/cmdline')
status['campaign_alive']=proc.exists() and b'transformer_qmc.correlation_campaign' in proc.read_bytes()
monitor=json.loads((root/'review_monitor_launch.json').read_text())['pid']
proc=Path(f'/proc/{monitor}/cmdline')
status['monitor_alive']=proc.exists() and b'watch_correlation_campaign.py' in proc.read_bytes()
print(json.dumps(status))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours',type=float,default=4.)
    p.add_argument('--escalation',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    version='v4' if args.escalation else 'v3'
    remote=REMOTE.replace('v3',version)
    destination=root/f'data/kagome/ltfim_correlation_lengths_{version}'
    destination.mkdir(parents=True,exist_ok=True)
    status_code=STATUS_CODE
    if args.escalation:
        status_code='''
import json
from pathlib import Path
root=Path('/root/autodl-tmp/correlation-lengths-20261103-v4')
status=json.loads((root/'results/status.json').read_text())
pid=json.loads((root/'launch.json').read_text())['pid']
proc=Path(f'/proc/{pid}/cmdline')
status['campaign_alive']=proc.exists() and b'transformer_qmc.escalate_correlation_campaign' in proc.read_bytes()
status['monitor_alive']=status['campaign_alive']
if status['state']=='completed' and (root/'results/review.json').exists():
    status['review']={'campaign_state':'completed'}
print(json.dumps(status))
'''
    deadline=time.monotonic()+args.hours*3600
    errors=0
    previous=None
    while time.monotonic()<deadline:
        try:
            result=subprocess.run(['sshpass','-e','ssh','-o','ConnectTimeout=15',
                                   '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
                                   'autodl','/root/miniconda3/bin/python -'],
                                  input=status_code,text=True,capture_output=True,timeout=45)
            if result.returncode:
                raise RuntimeError(result.stderr)
            status=json.loads(result.stdout)
            errors=0
        except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
            errors+=1
            print(f'Status connection failed ({errors}/3): {error}',flush=True)
            if errors>=3:
                raise
            time.sleep(30)
            continue
        if status!=previous:
            print(json.dumps(status),flush=True)
            previous=status
        if 'review' in status:
            subprocess.run(['rsync','-az','--exclude','*.tmp*','-e',
                            'sshpass -e ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4',
                            f'autodl:{remote}/results/',str(destination)+'/'],check=True,timeout=900)
            if status['review']['campaign_state']!='completed':
                raise RuntimeError('Campaign stopped early; partial results downloaded for diagnosis')
            print('Campaign and review complete; results downloaded.',flush=True)
            return
        if not status['monitor_alive']:
            raise RuntimeError('Remote review monitor exited without completion record')
        if not status['campaign_alive'] and status['state']!='completed':
            raise RuntimeError('Campaign exited before completion')
        time.sleep(30)
    raise TimeoutError('Monitoring deadline reached; remote jobs left running')


if __name__=='__main__':
    main()
